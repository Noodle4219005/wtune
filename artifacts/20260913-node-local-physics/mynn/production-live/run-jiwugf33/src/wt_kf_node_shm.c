#define _POSIX_C_SOURCE 200809L
#include "wt_kf_node_shm.h"
#include "wt_kf_node.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef WT_KF_NODE_SHM_DIRECTORY
#define WT_KF_NODE_SHM_DIRECTORY "/dev/shm"
#endif
#define KFS_HEADER 4096u
#define KFS_MAGIC UINT64_C(0x4b4653484d303032)
#define KFS_MAX_RANKS 512u
struct kfs_header {
    uint64_t magic,bytes,phase;
    uint32_t ranks,threads;
    size_t payload;
    _Atomic unsigned ready,active,ever;
    _Atomic unsigned pid[KFS_MAX_RANKS];
};
_Static_assert(sizeof(struct kfs_header)<=KFS_HEADER,"header page too small");
static int decimal(const char *s,uint32_t *out)
{
    uint64_t value=0;
    if (!s || !*s) return 0;
    for (;*s;++s) { if (*s<'0' || *s>'9') return 0; value=value*10+(unsigned)(*s-'0'); if (value>UINT32_MAX) return 0; }
    *out=(uint32_t)value; return 1;
}
static int count(const char *s,uint32_t *out)
{
    char number[16]; size_t n=0; uint32_t repeat;
    if (!s) return 0;
    while (s[n]>='0' && s[n]<='9' && n<sizeof(number)-1) { number[n]=s[n]; ++n; }
    number[n]=0;
    if (!decimal(number,out) || !*out) return 0;
    s+=n;
    if (!*s) return 1;
    if (*s++!='(' || *s++!='x') return 0;
    n=0;
    while (s[n]>='0' && s[n]<='9' && n<sizeof(number)-1) { number[n]=s[n]; ++n; }
    number[n]=0;
    return s[n]==')' && !s[n+1] && decimal(number,&repeat) && repeat;
}
int wt_kf_node_shm_configure_phase(struct wt_kf_node_shm *m,uint64_t phase,
                                  uint32_t threads,size_t payload,size_t budget)
{
    uint32_t job,step,ranks,rank;
    const char *counts=getenv("SLURM_STEP_TASKS_PER_NODE");
    size_t bytes;
    if (!m || !phase) return 0;
    memset(m,0,sizeof(*m)); m->fd=-1;
    if (!counts) counts=getenv("SLURM_TASKS_PER_NODE");
    if (!counts) counts=getenv("SLURM_NTASKS_PER_NODE");
    if (!decimal(getenv("SLURM_JOB_ID"),&job) || !job ||
        !decimal(getenv("SLURM_STEP_ID"),&step) ||
        !decimal(getenv("SLURM_LOCALID"),&rank) || !count(counts,&ranks) ||
        ranks<2 || ranks>KFS_MAX_RANKS || rank>=ranks || !threads || threads>64) return 0;
    m->config=(struct wt_nl_config){ranks,threads,payload}; m->rank=rank;
    if (wt_nl_required_bytes(&m->config,&bytes)!=WT_NL_OK ||
        bytes>SIZE_MAX-KFS_HEADER || bytes+KFS_HEADER>budget ||
        bytes+KFS_HEADER>(size_t)INT64_MAX) return 0;
    m->bytes=bytes+KFS_HEADER; m->phase=phase;
    if (phase==WT_KF_NODE_PHASE) {
        if (snprintf(m->path,sizeof(m->path),WT_KF_NODE_SHM_DIRECTORY "/wtune-kf-%lu-%u-%u",
                (unsigned long)getuid(),job,step)>=(int)sizeof(m->path)) return 0;
    } else if (snprintf(m->path,sizeof(m->path),WT_KF_NODE_SHM_DIRECTORY "/wtune-physics-%llx-%lu-%u-%u",
                (unsigned long long)phase,(unsigned long)getuid(),job,step)>=(int)sizeof(m->path)) return 0;
    return 1;
}
int wt_kf_node_shm_configure(struct wt_kf_node_shm *m,uint32_t threads,
                            size_t payload,size_t budget)
{ return wt_kf_node_shm_configure_phase(m,WT_KF_NODE_PHASE,threads,payload,budget); }
static void unlink_owned(const char *path,int fd)
{
    struct stat current,owned;
    if (path && *path && !fstat(fd,&owned) && !lstat(path,&current) &&
        current.st_dev==owned.st_dev && current.st_ino==owned.st_ino) (void)unlink(path);
}
int wt_kf_node_shm_attach(struct wt_kf_node_shm *m)
{
    struct stat st; struct kfs_header *h;
    int fd,creator,private_owned=0,final_owned=0;
    enum wt_nl_status status;
    unsigned expected=0;
    if (!m || !m->bytes || m->refused) return -1;
    if (m->registered) return 1;
    creator=m->rank==0;
    if (creator) {
        if (lstat(m->path,&st)==0 || errno!=ENOENT) { m->refused=1; return -1; }
        if (snprintf(m->private_path,sizeof(m->private_path),"%s.creating-%lu",m->path,
                (unsigned long)getpid())>=(int)sizeof(m->private_path)) { m->refused=1; return -1; }
    }
    fd=open(creator ? m->private_path : m->path,O_RDWR|O_CLOEXEC|O_NOFOLLOW|
        (creator ? O_CREAT|O_EXCL : 0),0600);
    if (fd<0) {
        if (!creator && errno==ENOENT) return 0;
        m->refused=1; return -1;
    }
    private_owned=creator;
    if (creator && ftruncate(fd,(off_t)m->bytes)) goto reject;
    if (fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_uid!=getuid() ||
        (st.st_mode&077)!=0) goto reject;
    if (st.st_size!=(off_t)m->bytes) {
        goto reject;
    }
    m->mapping=mmap(NULL,m->bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if (m->mapping==MAP_FAILED) { m->mapping=NULL; goto reject; }
    h=m->mapping;
    if (creator) {
        memset(h,0,KFS_HEADER);
        atomic_init(&h->ready,0); atomic_init(&h->active,0); atomic_init(&h->ever,0);
        for (unsigned i=0;i<KFS_MAX_RANKS;++i) atomic_init(&h->pid[i],0);
        h->magic=KFS_MAGIC; h->bytes=m->bytes; h->phase=m->phase;
        h->ranks=m->config.max_ranks; h->threads=m->config.max_threads; h->payload=m->config.slot_payload_bytes;
        status=wt_nl_init_mapping((unsigned char *)m->mapping+KFS_HEADER,
            m->bytes-KFS_HEADER,&m->config,&m->service);
        if (status!=WT_NL_OK) goto reject_mapping;
        atomic_store_explicit(&h->ready,1,memory_order_release);
        /* Only the fully initialized inode becomes visible at the final name.
         * link is exclusive: a stale/existing final name is never overwritten. */
        if (link(m->private_path,m->path)) goto reject_mapping;
        final_owned=1;
        if (!unlink(m->private_path)) { private_owned=0; m->private_path[0]=0; }
        /* An unlink failure leaves a redundant owned name, not a half-ready
         * public service. Quiescent close tries its inode-checked cleanup. */
    } else {
        if (atomic_load_explicit(&h->ready,memory_order_acquire)!=1 ||
            h->magic!=KFS_MAGIC || h->phase!=m->phase || h->bytes!=m->bytes || h->ranks!=m->config.max_ranks ||
            h->threads!=m->config.max_threads || h->payload!=m->config.slot_payload_bytes) goto reject_mapping;
        status=wt_nl_attach_mapping((unsigned char *)m->mapping+KFS_HEADER,m->bytes-KFS_HEADER,&m->service);
        if (status!=WT_NL_OK) goto reject_mapping;
    }
    if (!atomic_compare_exchange_strong(&h->pid[m->rank],&expected,(unsigned)getpid())) goto reject_mapping;
    atomic_fetch_add(&h->active,1); atomic_fetch_add(&h->ever,1);
    m->fd=fd; m->registered=1; return 1;
 reject_mapping:
    munmap(m->mapping,m->bytes); m->mapping=NULL; memset(&m->service,0,sizeof(m->service));
 reject:
    if (final_owned) unlink_owned(m->path,fd);
    if (private_owned) unlink_owned(m->private_path,fd);
    m->private_path[0]=0;
    close(fd); m->refused=1; return -1;
}
void wt_kf_node_shm_close(struct wt_kf_node_shm *m)
{
    if (!m) return;
    if (m->registered) {
        struct kfs_header *h=m->mapping;
        unsigned expected=(unsigned)getpid();
        if (!atomic_compare_exchange_strong(&h->pid[m->rank],&expected,UINT_MAX)) abort();
        if (atomic_fetch_sub(&h->active,1)==1 && atomic_load(&h->ever)==h->ranks) {
            struct stat pathst,fdst;
            if (!fstat(m->fd,&fdst) && !lstat(m->path,&pathst) &&
                pathst.st_dev==fdst.st_dev && pathst.st_ino==fdst.st_ino) (void)unlink(m->path);
        }
        unlink_owned(m->private_path,m->fd);
        munmap(m->mapping,m->bytes); close(m->fd);
    }
    memset(m,0,sizeof(*m)); m->fd=-1;
}
