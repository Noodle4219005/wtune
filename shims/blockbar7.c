/* blockbar7: DYNAMIC re-pin. The optimal rank->core placement changes as the
   terminator sweeps (measured: 84-98% of ranks change core between windows), so a
   static pin is 89-96% wrong over a full run. This reads BLOCKBAR_SCHEDULE
   ("grank c0 c1 ... c[nwin-1]" per line) and re-pins each rank to the CURRENT
   time-window's core, switching on elapsed wall time (BLOCKBAR_WINDOW_SEC).
   Still releases the core on Wait/Waitall/Barrier/Allreduce (over-sub needs it). */
#define _GNU_SOURCE
#include <mpi.h>
#include <sched.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static long sleep_ns=50000; static int inited=0; static int grank,gn;
static int *sched=NULL; static int nwin=1; static double win_sec=300.0;
static int cur_win=-1; static double t0=0;
static void pin(int c){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(c,&s); sched_setaffinity(0,sizeof(s),&s); }
static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static void initcfg(void){
  if(inited)return; inited=1;
  char*e=getenv("BLOCKBAR_SLEEP_NS"); if(e)sleep_ns=atol(e);
  e=getenv("BLOCKBAR_WINDOW_SEC"); if(e)win_sec=atof(e);
  e=getenv("BLOCKBAR_NWIN"); if(e)nwin=atoi(e); if(nwin<1)nwin=1;
  PMPI_Comm_rank(MPI_COMM_WORLD,&grank); PMPI_Comm_size(MPI_COMM_WORLD,&gn);
  sched=(int*)calloc(nwin,sizeof(int));
  char*fn=getenv("BLOCKBAR_SCHEDULE");
  if(fn){ FILE*f=fopen(fn,"r");
    if(f){ char line[16384];
      while(fgets(line,sizeof(line),f)){
        if(line[0]=='#') continue;
        char*tok=strtok(line," \t\n"); if(!tok) continue;
        int gg=atoi(tok); if(gg!=grank) continue;
        for(int w=0; w<nwin; w++){ tok=strtok(NULL," \t\n"); sched[w]= tok?atoi(tok):0; }
        break;
      }
      fclose(f);
    }
  }
  t0=now(); cur_win=-1;
  if(grank==0) fprintf(stderr,"[blockbar7] dynamic re-pin: nwin=%d win_sec=%.1f sleep_ns=%ld\n",nwin,win_sec,sleep_ns);
}
static void maybe_repin(void){
  int w=(int)((now()-t0)/win_sec); if(w>=nwin)w=nwin-1; if(w<0)w=0;
  if(w!=cur_win){ cur_win=w; pin(sched[w]); if(grank==0)fprintf(stderr,"[blockbar7] rank0 -> window %d core %d\n",w,sched[w]); }
}
static inline void bo(void){ struct timespec ts={0,sleep_ns}; nanosleep(&ts,NULL); }
int MPI_Wait(MPI_Request*r,MPI_Status*st){ initcfg(); maybe_repin(); int d=0;while(1){PMPI_Test(r,&d,st);if(d)break;bo();} return MPI_SUCCESS; }
int MPI_Waitall(int n,MPI_Request r[],MPI_Status st[]){ initcfg(); maybe_repin(); int d=0;while(1){PMPI_Testall(n,r,&d,st);if(d)break;bo();} return MPI_SUCCESS; }
int MPI_Barrier(MPI_Comm c){ initcfg(); maybe_repin(); MPI_Request r;int d=0;PMPI_Ibarrier(c,&r);while(1){PMPI_Test(&r,&d,MPI_STATUS_IGNORE);if(d)break;bo();} return MPI_SUCCESS; }
int MPI_Allreduce(const void*sb,void*rb,int c,MPI_Datatype dt,MPI_Op o,MPI_Comm m){ initcfg(); maybe_repin(); MPI_Request r;int f=0;PMPI_Iallreduce(sb,rb,c,dt,o,m,&r);while(1){PMPI_Test(&r,&f,MPI_STATUS_IGNORE);if(f)break;bo();} return MPI_SUCCESS; }
