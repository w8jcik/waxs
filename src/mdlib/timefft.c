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
static char *SRCID_testfft_c = "$Id$";

#include <math.h>
#include <stdio.h>
#include <time.h>
#include "typedefs.h"
#include "macros.h"
#include "smalloc.h"
#include "xvgr.h"
#include "complex.h"
#include "copyrite.h"
#include "fftgrid.h"
#include "mdrun.h"
#include "main.h"
#include "statutil.h"

int main(int argc,char *argv[])
{
  int       mmm[] = { 8, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40,
		      45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 100 };
  int       nnn[] = { 24, 32, 48, 60, 72, 84, 96 };
#define NNN asize(nnn)
  FILE      *fp;
  int       *niter;
  int       i,j,n,nit,ntot,n3,rsize;
  double    t,nflop,start;
  double    *rt,*ct;
  t_fftgrid *g;
  t_commrec *cr;
  static bool bOptFFT = FALSE;
  static int  nnode    = 1;
  static int  nitfac  = 1;
  t_pargs pa[] = {
    { "-opt",   FALSE, etBOOL, {&bOptFFT}, 
      "Optimize FFT" },
    { "-np",    FALSE, etINT, {&nnode},
      "Number of NODEs" },
    { "-itfac", FALSE, etINT, {&nitfac},
      "Multiply number of iterations by this" }
  };
  static t_filenm fnm[] = {
    { efLOG, "-g", "fft",      ffWRITE },
    { efXVG, "-o", "fft",      ffWRITE }
  };
#define NFILE asize(fnm)
  
  cr = init_par(&argc,&argv);
  if (MASTER(cr))
    CopyRight(stdout,argv[0]);
  parse_common_args(&argc,argv,
		    PCA_CAN_SET_DEFFNM | (MASTER(cr) ? 0 : PCA_QUIET),
		    TRUE,NFILE,fnm,asize(pa),pa,0,NULL,0,NULL);
  open_log(ftp2fn(efLOG,NFILE,fnm),cr);

  snew(niter,NNN);
  snew(ct,NNN);
  snew(rt,NNN);
  rsize = sizeof(real);
  for(i=0; (i<NNN); i++) {
    n  = nnn[i];
    if (n < 16)
      niter[i] = 50;
    else if (n < 26)
      niter[i] = 20;
    else if (n < 51)
      niter[i] = 10;
    else
      niter[i] = 5;
    niter[i] *= nitfac;
    nit = niter[i];
    
    if (MASTER(cr))
      fprintf(stderr,"\r3D FFT (%s precision) %3d^3, niter %3d     ",
	      (rsize == 8) ? "Double" : "Single",n,nit);
    
    g  = mk_fftgrid(stdlog,(nnode > 1),n,n,n,bOptFFT);

    if (PAR(cr))
      start = time(NULL);
    else
      start_time();
    for(j=0; (j<nit); j++) {
      gmxfft3D(g,FFTW_FORWARD,cr);
      gmxfft3D(g,FFTW_BACKWARD,cr);
    }
    if (PAR(cr)) 
      rt[i] = time(NULL)-start;
    else {
      update_time();
      rt[i] = node_time();
    }
    done_fftgrid(g);
    sfree(g);
  }
  if (MASTER(cr)) {
    fprintf(stderr,"\n");
    fp=xvgropen(ftp2fn(efXVG,NFILE,fnm),
		"FFT timings","n^3","t (s)");
    for(i=0; (i<NNN); i++) {
      n3 = 2*niter[i]*nnn[i]*nnn[i]*nnn[i];
      fprintf(fp,"%10d  %10g\n",nnn[i],rt[i]/(2*niter[i]));
    }
    fclose(fp);
  }
  return 0;
}
