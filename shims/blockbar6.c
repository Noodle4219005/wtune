/* blockbar4 + REORDER=file: read BLOCKBAR_PAIRFILE ("grank core" per line), pin this rank
   to its assigned core (within-node). Set I_MPI_PIN=0. Wait/Waitall/Barrier/Allreduce release core. */
#define _GNU_SOURCE
#include <mpi.h>
#include <sched.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static long sleep_ns=50000; static int inited=0; static int grank,gn,lrank,lsize;
static void pin(int c){ cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);sched_setaffinity(0,sizeof(s),&s); }
static void initcfg(void){
  if(inited)return; inited=1;
  char*e=getenv("BLOCKBAR_SLEEP_NS"); if(e)sleep_ns=atol(e);
  PMPI_Comm_rank(MPI_COMM_WORLD,&grank); PMPI_Comm_size(MPI_COMM_WORLD,&gn);
  MPI_Comm loc;PMPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&loc);
  PMPI_Comm_rank(loc,&lrank);PMPI_Comm_size(loc,&lsize);PMPI_Comm_free(&loc);
  char*m=getenv("REORDER");
  if(m&&!strcmp(m,"file")){
    char*fn=getenv("BLOCKBAR_PAIRFILE"); int core=lrank%(lsize/2);
    if(fn){FILE*f=fopen(fn,"r"); if(f){int g,c; while(fscanf(f,"%d %d",&g,&c)==2){ if(g==grank){core=c;break;} } fclose(f);} }
    pin(core);
  } else if(m&&!strcmp(m,"ew")){int NX=getenv("REORDER_NX")?atoi(getenv("REORDER_NX")):28;int h=NX/2,iy=lrank/NX,ix=lrank%NX;pin(iy*h+ix%h);}
  if(grank==0)fprintf(stderr,"[blockbar6] active sleep_ns=%ld reorder=%s\n",sleep_ns,m?m:"none");
}
static inline void bo(void){ struct timespec ts={0,sleep_ns}; nanosleep(&ts,NULL); }
int MPI_Wait(MPI_Request*r,MPI_Status*st){ initcfg(); int d=0;while(1){PMPI_Test(r,&d,st);if(d)break;bo();} return MPI_SUCCESS; }
int MPI_Waitall(int n,MPI_Request r[],MPI_Status st[]){ initcfg(); int d=0;while(1){PMPI_Testall(n,r,&d,st);if(d)break;bo();} return MPI_SUCCESS; }
int MPI_Barrier(MPI_Comm c){ initcfg(); MPI_Request r;int d=0;PMPI_Ibarrier(c,&r);while(1){PMPI_Test(&r,&d,MPI_STATUS_IGNORE);if(d)break;bo();} return MPI_SUCCESS; }
int MPI_Allreduce(const void*sb,void*rb,int c,MPI_Datatype dt,MPI_Op o,MPI_Comm m){ initcfg(); MPI_Request r;int f=0;PMPI_Iallreduce(sb,rb,c,dt,o,m,&r);while(1){PMPI_Test(&r,&f,MPI_STATUS_IGNORE);if(f)break;bo();} return MPI_SUCCESS; }
