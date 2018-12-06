/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.
 * Copyright (c) 2012,2013, by the GROMACS development team, led by
 * David van der Spoel, Berk Hess, Erik Lindahl, and including many
 * others, as listed in the AUTHORS file in the top-level source
 * directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

#ifndef _waxsmd_h
#define _waxsmd_h

#include "visibility.h"
#include "index.h"
#include "types/simple.h"
#include "gmxcomplex.h"
#include "oenv.h"
#include "types/waxsrec.h"
#include "types/waxstop.h"
#include "gmx_wallcycle.h"

/* suggested by Daniel Bowron in his EPSR tutorial (March 29,2010)
   Also see Sorensen et al, J. Chem. Phys. 113,9149 (2000)
   Note: We use NOT the Sorensen et al, formula, but instead
         f'(Q) = [ 1 + alpha*exp(-Q^2/2delta^2) ] * f(Q)
*/
/*#define WAXS_OPT_WATER_ALPHAO  0.12
#define WAXS_OPT_WATER_ALPHAH -0.48
#define WAXS_OPT_WATER_DELTA  22. */

/* Placeholders for deuteratable hydrogen atoms. These MUST correspond to
 * definitions in share/top/neutron-scatt-len-defs.itp
 */
#define NSL_H_DEUTERATABLE            -1000
#define NSL_H_DEUTERATABLE_BACKBONE   -2000

#define NEUTRON_SCATT_LEN_1H        (-3.7406)
#define NEUTRON_SCATT_LEN_2H          6.671
#define NEUTRON_SCATT_LEN_O           5.803

/* According to the Cryson paper, the fraction of deuterated backbone H
 *  is typically 10% below the deuteration of the other polar H
 */
// #define NOT_DEUTERATED_BACKBONE_H_FRACTION 0.1
#define BACKBONE_DEUTERATED_PROB_DEFAULT 0.9

#define CONSTANT_HC 1239.842

#ifdef __cplusplus
extern "C" {
#endif

#if 0
}
/* Hack to make automatic indenting work */
#endif

/* Default settings for J, the number of points for numerical sphereical average
 * In automatic mode,
 *    J(q) = MAX( Jmin , alpha * (D*q)^2 )
 * where D is the maximum diameter of the envelope.
 */
#define GMX_WAXS_JMIN    100
#define GMX_WAXS_J_ALPHA 0.1


/* Default settint for tolerance on state weights */
#define GMX_WAXS_WEIGHT_TOLERANCE_DEFAULT 0.1

#define CROMERMANN_2_NELEC(x) (x.c + x.a[0] + x.a[1] + x.a[2] + x.a[3])

int map_indices_molblock_atoms( gmx_mtop_t *mtop, int *map );

/* This is called near the beginning of MD runs to setup all the variables
 * and allocate all necessary blocks given the conditions of the run.
 * Later functions should not have to allocate anything that lasts
 * for the duration of the simulation once this is invoked. */
void init_waxs_md( t_waxsrec *wr,
                   t_commrec *cr, t_inputrec *ir,
                   gmx_mtop_t *top_global,
                   const output_env_t oenv, double t0,
                   const char *fntpsSolv, const char *fnxtcSolv,const char *fnOut,
                   const char *fnScatt,
                   t_state *state_local, gmx_bool bRerunMD, gmx_bool bWaterOptSet,
                   gmx_bool bReadI);

/* Iterates through various sub-structures of t_waxsrec to ensure safe removal */
void done_waxs_md(t_waxsrec *wr);

/* Checks if global are present, and if not, allocate them. */
void waxs_alloc_globalavs(t_waxsrec *wr);

/* Checks if global are present, and if so, free them. */
void waxs_free_globalavs(t_waxsrec *wr);

t_spherical_map *gen_qvecs_map( real minq, real maxq, int nqabs, int J,
                                gmx_bool bDebug, t_commrec *cr, int ewaxsaniso, real qbeam,
                                gmx_envelope_t env, int Jmin, real Jalpha, gmx_bool bVerbose);

/* int init_waxs_md (FILE *fplog,
          t_commrec *cr,t_inputrec *ir, t_forcerec *fr, const output_env_t oenv,
          gmx_mtop_t *top_global,int nfile,const t_filenm fnm[]); */

/* Header placement obsoleted by done_waxs_md*/
void done_waxs_solvent(t_waxs_solvent ws );

void done_waxs_output(t_waxsrec *wr, output_env_t oenv);

void waxs_ensemble_average(t_waxsrec *wr, t_waxsrecType *wt, t_commrec *cr, int iq, gmx_bool bVerbose);

void do_waxs_md_low(t_commrec *cr, rvec x[], double t, gmx_large_int_t step,
                    t_waxsrec *wr, gmx_mtop_t *mtop, matrix box, int ePBC, gmx_bool bDoLog);

/* Does MPI communication between global averages on master, and its local copies,
 * using a prepared nq_masterlist, qoff_masterlist on master to determine
 * size and offsets of arrays based on q-points handled by the node.
 * Combines dissemination and collection below into one function.
 * Without MPI, switches to an old inefficient method.
 * bCollect TRUE is inwards communication, FALSE is outwards communication */
void
waxsDoMPIComm_qavgs(t_waxsrec *wr, t_commrec *cr, gmx_bool bCollect);

/* The envelope communications are here because node-specific qlists are required. */
void
waxsDoMPIComm_envelope(t_waxsrec *wr, t_commrec *cr, gmx_bool bCollect);

/* Default MPI call after Checkpoint I/O */
void
waxsDoMPIComm_cpt(t_waxsrec *wr, t_commrec *cr, gmx_bool bCollect);

/* WAXS checkpoints contain all relevant averages in A and B.
 * Note that the checkpoints do not contain D at present.
 * The number of bytes written will be reported. */
void
waxs_write_checkpoint(t_waxsrec *wr, const char * fn);

void
waxs_read_checkpoint(t_waxsrec *wr, const char * fn);

/* Debugging function. Allows general access to report internal variables to stderr*/
void
waxs_quick_debug(t_waxsrec *wr);

void
waxsEstimateNumberIndepPoints(t_waxsrec *wr, t_commrec *cr, gmx_bool bWriteACF2File,
                              gmx_bool bScale_varI_now);

#ifdef __cplusplus
}
#endif
#endif
