#ifndef WT_KF_NODE_SHM_H
#define WT_KF_NODE_SHM_H
#include "wt_node_local.h"
struct wt_kf_node_shm {
    struct wt_nl_service service;
    struct wt_nl_config config;
    void *mapping;
    size_t bytes;
    uint64_t phase;
    uint32_t rank;
    int fd, registered, refused;
    char path[192], private_path[224];
};
/* Slurm step counts take precedence; only homogeneous counts are admitted.
 * Creator is local rank zero. Existing creator names are terminally refused. */
/* Nonzero phase namespaces both names and immutable mapping identity. */
int wt_kf_node_shm_configure_phase(struct wt_kf_node_shm *, uint64_t phase,
    uint32_t threads, size_t slot_bytes, size_t total_budget);
/* Compatibility entry uses the KF phase and retains its existing filename. */
int wt_kf_node_shm_configure(struct wt_kf_node_shm *, uint32_t threads,
    size_t slot_bytes, size_t total_budget);
/* One nonblocking attempt, outer-scope control thread only. 1 ready, 0 pending,
 * -1 refused. Creator initializes a private inode then atomically links the
 * complete READY mapping at its final exclusive name; partial files are hidden. */
int wt_kf_node_shm_attach(struct wt_kf_node_shm *);
/* Quiescent only. The last clean rank removes only its exact filename after
 * every expected rank registered. Crashed/incomplete runs leave stale evidence. */
void wt_kf_node_shm_close(struct wt_kf_node_shm *);
#endif
