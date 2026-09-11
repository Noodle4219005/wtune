/* PASS 1 profiler: wrap blocking MPI (releasing core via nanosleep) AND measure each rank's
   wait-fraction (time in Wait/Waitall/Barrier/Allreduce over wall since first call).
   At MPI_Finalize, Gather wait_frac + node-local id to rank 0, which writes a load map file
   BLOCKBAR_PROF_OUT (default ./blockbar_load.txt): "grank lrank wait_frac". */
#define _GNU_SOURCE
#include <mpi.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
static long sleep_ns=200000; static int inited=0; static double t_wait=0,t0=0; static int grank,gn,lrank,lsize;
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void initcfg(void){
  if(inited) return; inited=1;
  char*e=getenv("BLOCKBAR_SLEEP_NS"); if(e) sleep_ns=atol(e);
  PMPI_Comm_rank(MPI_COMM_WORLD,&grank); PMPI_Comm_size(MPI_COMM_WORLD,&gn);
  MPI_Comm loc; PMPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&loc);
  PMPI_Comm_rank(loc,&lrank); PMPI_Comm_size(loc,&lsize); PMPI_Comm_free(&loc);
  t0=now();
  if(grank==0) fprintf(stderr,"[blockbar_prof] active sleep_ns=%ld local_ranks=%d\n",sleep_ns,lsize);
}
static inline void bo(void){ struct timespec ts={0,sleep_ns}; nanosleep(&ts,NULL); }
int MPI_Wait(MPI_Request*rq,MPI_Status*st){ initcfg(); double s=now();int d=0; while(1){PMPI_Test(rq,&d,st);if(d)break;bo();} t_wait+=now()-s; return MPI_SUCCESS; }
int MPI_Waitall(int n,MPI_Request rq[],MPI_Status st[]){ initcfg(); double s=now();int d=0; while(1){PMPI_Testall(n,rq,&d,st);if(d)break;bo();} t_wait+=now()-s; return MPI_SUCCESS; }
int MPI_Barrier(MPI_Comm c){ initcfg(); double s=now();MPI_Request rq;int d=0;PMPI_Ibarrier(c,&rq);while(1){PMPI_Test(&rq,&d,MPI_STATUS_IGNORE);if(d)break;bo();} t_wait+=now()-s; return MPI_SUCCESS; }
int MPI_Allreduce(const void*sb,void*rb,int c,MPI_Datatype dt,MPI_Op o,MPI_Comm m){ initcfg(); double s=now();MPI_Request rq;int f=0;PMPI_Iallreduce(sb,rb,c,dt,o,m,&rq);while(1){PMPI_Test(&rq,&f,MPI_STATUS_IGNORE);if(f)break;bo();} t_wait+=now()-s; return MPI_SUCCESS; }
int MPI_Finalize(void){
  double wall=now()-t0, frac= wall>0? t_wait/wall : 0;
  double *fr=NULL; int *ll=NULL;
  if(grank==0){ fr=malloc(8*gn); ll=malloc(4*gn); }
  PMPI_Gather(&frac,1,MPI_DOUBLE,fr,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  PMPI_Gather(&lrank,1,MPI_INT,ll,1,MPI_INT,0,MPI_COMM_WORLD);
  if(grank==0){ char*o=getenv("BLOCKBAR_PROF_OUT"); FILE*f=fopen(o?o:"blockbar_load.txt","w");
    if(f){ for(int r=0;r<gn;r++) fprintf(f,"%d %d %.4f\n",r,ll[r],fr[r]); fclose(f);
           fprintf(stderr,"[blockbar_prof] wrote load map (%d ranks) local_ranks/node=%d\n",gn,lsize);} }
  return PMPI_Finalize();
}
