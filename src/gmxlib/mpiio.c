/*
 * $Id$
 * 
 *       This source code is part of
 * 
 *        G   R   O   M   A   C   S
 * 
 * GROningen MAchine for Chemical Simulations
 * 
 *               VERSION 2.0
 * 
 * Copyright (c) 1991-1999
 * BIOSON Research Institute, Dept. of Biophysical Chemistry
 * University of Groningen, The Netherlands
 * 
 * Please refer to:
 * GROMACS: A message-passing parallel molecular dynamics implementation
 * H.J.C. Berendsen, D. van der Spoel and R. van Drunen
 * Comp. Phys. Comm. 91, 43-56 (1995)
 * 
 * Also check out our WWW page:
 * http://md.chem.rug.nl/~gmx
 * or e-mail to:
 * gromacs@chem.rug.nl
 * 
 * And Hey:
 * Green Red Orange Magenta Azure Cyan Skyblue
 */
static char *SRCID_mpiio_c = "$Id$";

#include <string.h>
#include "mpiio.h"
#include "fatal.h"
#include "main.h"
#include "smalloc.h"

static int  mpi_num_nodes=0;
static int  mpi_my_rank=-1;
static char mpi_hostname[MPI_MAX_PROCESSOR_NAME];

static MPI_Request mpi_req_tx=MPI_REQUEST_NULL,mpi_req_rx;

/*#define DEBUG*/

/* Try setting MPI_TEST when you experience unexplainable crashes, *
 * up til now these crashes have only occured with IRIX 6.5        */
/* #define MPI_TEST */

void mpiio_tx(int nodeid,void *buf,int bufsize)
{
  int        tag,flag;
  MPI_Status status;
  
#ifdef DEBUG
  fprintf(stderr,"mpiio_tx: nodeid=%d, buf=%x, bufsize=%d\n",nodeid,buf,bufsize);
#endif
#ifdef MPI_TEST
  /* workaround for crashes encountered with MPI on IRIX 6.5 */
  if (mpi_req_tx != MPI_REQUEST_NULL) {
    MPI_Test(&mpi_req_tx,&flag,&status);
    if (flag==FALSE) {
      fprintf(stdlog,"mpiio_tx called before previous send was complete: nodeid=%d, buf=%x, bufsize=%d\n",
	      nodeid,buf,bufsize);
      mpiio_tx_wait(nodeid);
    }
  }
#endif
  tag = 0;
  if (MPI_Isend(buf,bufsize,MPI_BYTE,nodeid,tag,MPI_COMM_WORLD,&mpi_req_tx) != 0)
    fatal_error(0,"MPI_Isend Failed !");
}

void mpiio_tx_wait(int nodeid)
{
  MPI_Status  status;
  int mpi_result;
  
  if ((mpi_result=MPI_Wait(&mpi_req_tx,&status)) != 0)
    fatal_error(0,"MPI_Wait: result=%d",mpi_result);
}

void mpiio_txs(int nodeid,void *buf,int bufsize)
{
  int tag;

#ifdef DEBUG
  fprintf(stderr,"mpiio_txs: nodeid=%d, buf=%x, bufsize=%d\n",nodeid,buf,bufsize);
#endif
  tag = 0;
  if (MPI_Send(buf,bufsize,MPI_BYTE,nodeid,tag,MPI_COMM_WORLD) != 0)
    fatal_error(0,"MPI_Send Failed !");
  /* mpiio_tx(nodeid,buf,bufsize);
     mpiio_tx_wait(nodeid);*/
}

void mpiio_rx(int nodeid,void *buf,int bufsize)
{
  int        tag;

#ifdef DEBUG
  fprintf(stderr,"mpiio_rx: nodeid=%d, buf=%x, bufsize=%d\n",nodeid,buf,bufsize);
#endif
  tag = 0;
  if (MPI_Irecv( buf, bufsize, MPI_BYTE, nodeid, tag, MPI_COMM_WORLD, &mpi_req_rx) != 0 )
    fatal_error(0,"MPI_Recv Failed !");
}

void mpiio_rx_wait(int nodeid)
{
  MPI_Status  status;
  int mpi_result;
  
  if ((mpi_result=MPI_Wait(&mpi_req_rx,&status)) != 0)
    fatal_error(0,"MPI_Wait: result=%d",mpi_result);
}

int mpiio_rx_probe(int nodeid)
{
  MPI_Status  status;
  int mpi_result,flag=0;
  
  if ((mpi_result = MPI_Test(&mpi_req_rx,&flag,&status)) != MPI_SUCCESS)
    fatal_error(0,"MPI_Test: result=%d",mpi_result);
    
  return flag;
}

void mpiio_rxs(int nodeid,void *buf,int bufsize)
{
  MPI_Status stat;
  int        tag;

#ifdef DEBUG
  fprintf(stderr,"mpiio_rxs: nodeid=%d, buf=%x, bufsize=%d\n",nodeid,buf,bufsize);
#endif
  tag = 0;
  if (MPI_Recv( buf, bufsize, MPI_BYTE, nodeid, tag, MPI_COMM_WORLD, &stat) != 0 )
    fatal_error(0,"MPI_Recv Failed !");
  /* mpiio_rx(nodeid,buf,bufsize);
     mpiio_rx_wait(nodeid);*/
}

int mpiio_setup(int *argc,char **argv,int *nnodes)
{
  char   buf[256];
  int    resultlen;               /* actual length of node name      */
  int    i,flag;

  /* Call the MPI routines */
  (void) MPI_Init(argc,&argv);
  (void) MPI_Comm_size( MPI_COMM_WORLD, &mpi_num_nodes );
  (void) MPI_Comm_rank( MPI_COMM_WORLD, &mpi_my_rank );
  (void) MPI_Get_processor_name( mpi_hostname, &resultlen );

  fprintf(stderr,"NNODES=%d, MYRANK=%d, HOSTNAME=%s\n",
	  mpi_num_nodes,mpi_my_rank,mpi_hostname);
  
  *nnodes=mpi_num_nodes;
  
  return mpi_my_rank;
}

/* If this is the master, spawn some kids. Return nodeid */

void mpiio_stat(FILE *fp,char *msg)
{
  ;
}

int  mpinodecount(void)
{
  return mpi_num_nodes;
}

int mpinodenumber(void)
{
  return mpi_my_rank;
}

int mpi_idle_send(void)
{
  return 0;
}

int mpi_idle_rec(void)
{
  return 0;
}

void mpi_left_right(int nnodes,int nodeid,int *left,int *right)
{
  *left  = (nnodes+nodeid-1) % nnodes;
  *right = (nodeid+1) % nnodes;
}

void mpiio_tx_rx(int send_nodeid,void *send_buf,int send_bufsize,
		 int rec_nodeid,void *rec_buf,int rec_bufsize)
{
  fatal_error(0,"mpiio_tx_rx not implemented!");
}
		 
void mpiio_wait(int left,int right)
{
  mpiio_tx_wait(left);
  mpiio_rx_wait(right);
}

void mpiio_sync_ring(int nodeid,int nnodes,int left,int right)
{
  fatal_error(0,"mpiio_sync_ring called");;
}

void mpi_reset_idle(void)
{
  ;
}

void mpi_abort(int nodeid,int nnodes,int errorno)
{
  MPI_Abort(MPI_COMM_WORLD,errorno);
}

#ifdef DEBUG
#undef DEBUG
#endif
