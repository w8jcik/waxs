/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands
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

/*
 *  This source file was written by Jochen Hub, with contributions from Po-chia Chen.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/* Do we need some #ifdef statement here (such as GMX_THREAD_SHM_FDECOMP) ?? */
#include <pthread.h>

#include <ctype.h>
#include "sysstuff.h"
#include "typedefs.h"
#include "statutil.h"
#include "smalloc.h"
#include "string2.h"
#include "futil.h"
#include "maths.h"
#include "gmx_fatal.h"
#include "vec.h"
#include "macros.h"
#include "index.h"
#include "strdb.h"
#include "copyrite.h"
#include "gmxfio.h"
#include "trnio.h"
#include "tpxio.h"
#include "oenv.h"
#include "xvgr.h"
#include "matio.h"
#include "gmx_ana.h"
#include "names.h"
#include "waxsmd.h"
#include "pbc.h"
#include "rmpbc.h"
#include "confio.h"
#include "ns.h"
#include "nsgrid.h"
#include "xtcio.h"
#include "randframes.h"
#include "sftypeio.h"
#include "mtop_util.h"
#include "mdrun.h"
#include "txtdump.h"
#include "do_fit.h"
#include "waxsmd_utils.h"
#include "gmx_envelope.h"
#include "xdrf.h"
#include "checkpoint.h"
#include <time.h>
#include "string2.h"
#include "gmx_wallcycle.h"
#include "gmx_sort.h"
#include "gmx_omp_nthreads.h"
#include "gmx_omp.h"

#include"cuda_tools/scattering_amplitude_cuda.cuh" /* for calculation of scattering amplitude on GPU */

#include <sys/time.h>
#include<string.h>

/* include the right header file */
#ifdef GMX_DOUBLE

#if   defined(GMX_X86_AVX_256)
#include "gmx_math_x86_avx_256_double.h"
#elif defined(GMX_X86_AVX_128_FMA)
#include "gmx_math_x86_avx_128_fma_double.h"
#elif defined(GMX_X86_SSE4_1)
#include "gmx_math_x86_sse4_1_double.h"
#elif defined(GMX_X86_SSE2)
#include "gmx_math_x86_sse2_double.h"
#endif

#else  /* single prec. */

#if   defined(GMX_X86_AVX_256)
#include "gmx_math_x86_avx_256_single.h"
#elif defined(GMX_X86_AVX_128_FMA)
#include "gmx_math_x86_avx_128_fma_single.h"
#elif defined(GMX_X86_SSE4_1)
#include "gmx_math_x86_sse4_1_single.h"
#elif defined(GMX_X86_SSE2)
#include "gmx_math_x86_sse2_single.h"
#endif
#endif /* GMX_DOUBLE */


/* #define GMX_WAXS_NO_SIMD */

/*
 * Macro names for AVX/SSE instructions
 */
#if defined(GMX_WAXS_NO_SIMD)

/* Generic placeholders, in case we don't use SIMD - only kept for testing */
#define REGISTER real
#define REGISTER_SINCOS waxs_sincos_real
#define REGISTER_MUL(a,b) ((a)*(b))

#elif defined(GMX_X86_AVX_256)

#ifdef GMX_DOUBLE
#define REGISTER __m256d
#define REGISTER_SINCOS gmx_mm256_sincos_pd
#define REGISTER_MUL    _mm256_mul_pd
#else
#define REGISTER __m256
#define REGISTER_SINCOS gmx_mm256_sincos_ps
#define REGISTER_MUL    _mm256_mul_ps
#endif

#elif defined(GMX_X86_AVX_128_FMA) || defined(GMX_X86_SSE4_1) || defined(GMX_X86_SSE2)
/* AVX_128_FMA, SSE2, or SSE4.1 */
#ifdef GMX_DOUBLE
#define REGISTER __m128d
#define REGISTER_SINCOS gmx_mm_sincos_pd
#define REGISTER_MUL    _mm_mul_pd
#else
#define REGISTER __m128
#define REGISTER_SINCOS gmx_mm_sincos_ps
#define REGISTER_MUL    _mm_mul_ps
#endif

#endif /* SIMD level */


#define WAXS_MAX(a,b) ((a)>(b)?(a):(b))
#define WAXS_MIN(a,b) ((a)<(b)?(a):(b))
#define UPDATE_CUMUL_AVERAGE(f, w, fadd) (f = 1.0/(w) * ((w-1.0)*(f) + 1.0*(fadd)))

#define WAXS_WARN_BOX_DIST 0.5

#define WAXS_ENSEMBLE(x) (x->ensemble_nstates > 1)

#define XRAY_NEUTRON_STRING(x) ((x == escatterXRAY) ? "xray" : "neutron")


#define waxs_debug(x)
/*
 * Uncomment some of the following definitions if you want more debugging output
 */
// #define waxs_debug(x) _waxs_debug(x, __FILE__, __LINE__)
// #define RERUN_CalcForces
// #define PRINT_A_dkI

/* Portable version of ctime_r implemented in src/gmxlib/string2.c, but we do not want it declared in public installed headers */
GMX_LIBGMX_EXPORT
char *
gmx_ctime_r(const time_t *clock, char *buf, int n);

//#define GMX_GPU

/* Large structure that keeps all the averages specific to one scattering type (such
   as X-ray scattering, or Neutron scattering with a specific D2O concentration */
struct waxs_datablock {
    t_complex_d *A, *B;  /* Scattering amplitude of the present step,       size qhomenr */

    /* The gradients
       grad[k] A(q) = i . q . f[k](q) . exp(iq.r[k])
          require lots of memory. Therefore, we only store [ i . f[k](q) . exp(iq.r[k]) ]* and
          multipy by q later. */
    t_complex *dkAbar; /* size qhomenr * nind_prot */

    /* NOTE ON PRECISION:
     *       dkAbar is done in real
     *       All averages are done in double prec.
     */

    double normA, normB;    /* Norm for exponential or equally-weighted averaging.
                                   N(n) = 1+f+f^2+...f^n, where f=exp(-deltat/tau)
                                   Use <A(n)> = N(n)^[-1] [ A(n) + f.<A(n-1)> ] */

    t_complex_d  *avA ;             /* <A>, size qhomenr     */
    t_complex_d  *avAcorr;          /* Correction added to <A> to scale I(q=0) to a given value
                                       if waxsrec->bScaleI0, see mdp option waxs-scalei0 */
    t_complex_d  *avB ;             /* <B>, size qhomenr     */
    double       *avAsq ;           /* <|A|^2>, size qhomenr */
    double       *avBsq ;           /* <|B|^2>, size qhomenr */
    double       *re_avBbar_AmB;    /* Re < B* . (A-B) >   */
    double       *D;                /* size qhomenr        */
    double       *DerrSolvDens;     /* Uncertainty of D(q) due to uncertainty in solvent density */

    double *Dglobal;          /* D        of size qvecs->n - for Master for output only */
    t_complex_d *avAglobal;   /* avA      of size qvecs->n - for Master for output only */
    double *avAsqglobal;      /* avAsq    of size qvecs->n - for Master for output only */
    double *avA4global;       /* avA4     of size qvecs->n - for Master for output only */
    double *av_ReA_2global;   /* av_ReA_2 of size qvecs->n - for Master for output only */
    double *av_ImA_2global;   /* av_ImA_2 of size qvecs->n - for Master for output only */
    t_complex_d *avBglobal;   /* avB      of size qvecs->n - for Master for output only */
    double *avBsqglobal;      /* avBsq    of size qvecs->n - for Master for output only */
    double *avB4global;       /* avB4     of size qvecs->n - for Master for output only */
    double *av_ReB_2global;   /* av_ReB_2 of size qvecs->n - for Master for output only */
    double *av_ImB_2global;   /* av_ImB_2 of size qvecs->n - for Master for output only */

    double *avA4;             /* < |A|^4 >    - to compute errors */
    double *avB4;             /* < |B|^4 >    - to compute errors */
    double *av_ReA_2;         /* < (Re A)^2 > - to compute errors */
    double *av_ImA_2;         /* < (Im A)^2 > - to compute errors */
    double *av_ReB_2;         /* < (Re B)^2 > - to compute errors */
    double *av_ImB_2;         /* < (Im B)^2 > - to compute errors */

    /* Orientationa average of Re[ grad[k](A*) (A-<B>) ].
       Explanation: In eq. 13, of Chen and Hub, Biophys J 2015 (http://dx.doi.org/10.1016/j.bpj.2015.03.062), we
       use that <B(q)> does not change after a while any more. Hence, we replace everything within the square brackets by
       < grad[k](A*) (A-<B>) >_t
       -> All operations (Real part, time average, and orientational average) can be done in an arbitrary order.
       -> Do the orientational average immediately, before doing the time average.
       -> No need to save the huge array grad[k](A*)(vec[q])
       -> Same memory, so we can do the calculation on the GPU.
    */
    dvec *Orientational_Av;

    /* Byte blocks and offsets for communications. */
    gmx_large_int_t nByteAlloc; /* Memory allocated to store averages (on this node) */

    int *masterlist_qhomenr;    /* qstart and qhomenr kept on the Master node. Required for checkpointing */
    int *masterlist_qstart;

    /* The SAXS/SANS intensities, size nIvalues */
    int nIvalues;             /* # intensity values stored. nabs with isotropic patterns, otherwise more */
    double *I;                /* SAXS/SANS inentities  - size nIvalues */
    dvec *dkI;                /* grad[k] I(q)          - size nIvalues*nind_prot */


    /*  All the followint intenties, variances, etc. are only for computing errors and for
     *  reporting the contributions to the SAXS/SANS curves, as provided by waxs_contrib.xvg
     */
    double *ensemble_I_Aver;  /* I(q) of the complete ensemble, */
    double *IA, *IB;          /* Intensity from only A or B atoms, respectively */
    double *Ibulk;            /* Intensity from coupling between solute/excluded solvent and bulk water
                                     Ibulk = 2Re [ -<B*>.<A-B> ] */
    double *I_avAmB2;         /* Intensities from the three terms in eq. 26 of Park et al,  */
    double *I_varA;           /* D = |<A-B>|^2 + var(A) - var(B), size nIvalues             */
    double *I_varB;
    double *I_errSolvDens;    /* Uncertainty in I(q) due to uncertainty of the solvent density */
    double *I_scaleI0;        /* Additional intensity due to scaling of I(q=0) */
    double *Nindep;           /* For each |q|: number of statistically independent D(q),
                                     or "effective sample size */

    double *varIA, *varIB;    /* There respective variances - to compute errors */
    double *varIbulk, *varI;
    double *varI_avAmB2, *varI_varA, *varI_varB;
    double *avAqabs_re, *avAqabs_im; /* < A(|q|) >, size nIvalues */
    double *avBqabs_re, *avBqabs_im; /* < B(|q|) >, size nIvalues */

    double *vAver, *vAver2;   /* Average WAXS potential vs q:  < V[waxs] (q) > */

    /* Used to compute the total contrast between A and B system, used
       to test if we have overall positive or negative contrast */
    double avAsum;             /* average total number of electrons/NSLs of A system */
    double avBsum;             /* average total number of electrons/NSLs of B system */
};

/* Structure for on-the-fly calculation of the RMSD. This can be used to scale down the SAXS-derived
   forces after a conformational transition, so we avoid the "overshooting".
   Current status (Dec. 2017): It seems that by gradually switching on the forces (with mdp-option waxs-ttarget)
   we don't need this any more.
*/
struct waxs_eavrmsd {
    double scale;       /* Copy of wr->scale. exp(-1*dt/tau) for both ensemble-x and RMSD */
    double norm;        /* Norm */
    rvec *x_avg;        /* Ensemble average positions. */
    real rmsd_now;      /* RMSD of current snapshot to said ensemble average */
    real rmsd_av;       /* Exponential average of said RMSD. */
    real rmsd_avsq;     /* For standard deviation calculation */
    real sd_now;        /* SD of above exponential average */
};

struct waxs_output {
    /* output that needs only one file, even with multiple scattering types */
    FILE *fpLog, *fpExpRMSD;
    char *fnLog, *fnExpRMSD, *fnDensity;

    /* output that need an extra file for each scattering type (xray, neutron, etc) */
    FILE **fpSpectra, **fpStddevs,                         **fpForces, **fpSpecEns, **fpPot, **fpNindep, **fpD;
    char **fnSpectra, **fnStddevs, **fnFinal, **fnContrib, **fnForces, **fnSpecEns, **fnPot;
    FILE **fpAvgA, **fpAvgB, **fpGibbsEnsW, **fpGibbsSolvDensRelErr;
    char **fnAvgA, **fnAvgB, **fnGibbsEnsW, **fnGibbsSolvDensRelErr, **fnNindep, **fnD;
    t_fileio **xfout;
};

enum {
    mpi_typeINT,
    mpi_typeFLOAT,
    mpi_typeDOUBLE,
    mpi_typeREAL
};

void
_waxs_debug(const char* str, const char *file, int linenr)
{
    fprintf(stderr, "DEGUB: %s, line %4d -- %s\n", file, linenr, str ? str : "");
}

static inline void
waxs_sincos_real(real arg, real *sine, real *cosine)
{
#ifdef GMX_DOUBLE
    *sine   = sin(arg);
    *cosine = cos(arg);
#else
    *sine   = sinf(arg);
    *cosine = cosf(arg);
#endif
}
const char *waxs_simd_string()
{
#if defined (GMX_WAXS_NO_SIMD)
    return "None";
#elif defined(GMX_X86_AVX_256)
    return "AVX-256";
#elif defined(GMX_X86_AVX_128_FMA)
    return "AVX-128-FMA";
#elif defined(GMX_X86_SSE4_1)
    return "SSE-4.1";
#elif defined(GMX_X86_SSE2)
    return "SSE-2";
#endif
}

static void
assert_scattering_type_is_set(int stype, int type, int atomno)
{
    if (stype == NOTSET)
    {
        gmx_fatal(FARGS, "Found a scattering type (%s) that is NOTSET, atom no %d. This should not happen.\n",
                  (type == escatterXRAY) ? "Cromer-Mann type" : "Neutron scatterling length type",
                  atomno + 1);
    }
}


/* Lower function that handles an individual communication step.
 * nPerq is the number of eTypes that will be sent per vector q.
 * e.g. avAglobal is of size nq*dcomplex, which reduces to 1 complex(2 doubles) per q.
 */
void
waxsDoMPIComm_qavgs_low(t_waxsrecType *wt, t_commrec *cr, void *loc_buf, void *glb_buf,
                        int datatype, int nPerq, gmx_bool bCollect)
{
    int nq_loc = wt->qvecs->qhomenr; // Number of q's this node is responsible for.
    int loc_cnts;                    //Size of each package on outer nodes.
    int *glb_cnts = NULL;            //Master array: expected number of incoming bytes from each node.
    int *glb_offs = NULL;            //Master array: offsets to put into destination.
    gmx_bool bLocBufEmpty = FALSE;

    gmx_large_int_t test;
    int i;
    int nnodes = cr->nnodes - cr->npmenodes;
    int limit  = 2147483647; // 2 Gigs, largest possible int?
    const gmx_bool bVerbose = FALSE;

    t_waxs_datablock wd = wt->wd;

    /* Test for size */
    test = wt->qvecs->n * nPerq;

    if (test > limit)
    {
        gmx_fatal(FARGS,"Integer overflow detected in waxsDoMPIComm!\n"
                  "Perhaps your system is too big for WAXS checkpointing.");
    }
    loc_cnts = nq_loc * nPerq;

    if (loc_buf == NULL)
    {
        if (loc_cnts > 0)
        {
            gmx_fatal(FARGS, "Error in waxsDoMPIComm_qavgs_low(): loc_cnts = %d, but loc_buf = NULL (rank %d)\n",
                      loc_cnts, cr->nodeid);
        }
        /* If no q-vectors are stored on a node, loc_buf is zero, leadig to an MPI error below.
           Therefore, in this case, allocate a small arrray and clear it below */
        bLocBufEmpty = TRUE;
        snew(loc_buf, 8);
    }

    /* Construct count arrays on master.
     * This is counting the number of eType that each will send. */
    if (MASTER(cr))
    {
        snew(glb_cnts, nnodes);
        snew(glb_offs, nnodes);

        for (i = 0; i < nnodes; i++)
        {
            glb_cnts[i] = wd->masterlist_qhomenr[i] * nPerq;
            test        = wd->masterlist_qstart [i] * nPerq;
            if (test > limit)
            {
                gmx_fatal(FARGS,"Integer overflow detected in waxsDoMPIComm!\n"
                          "Perhaps your system is too big for WAXS checkpointing, or"
                          "convince the devs to implement multi-step communications.");
            }
            glb_offs[i] = test;
        }
    }
    if (PAR(cr)) gmx_barrier(cr);

    if (bVerbose)
    {
        if (MASTER(cr))
        {
            for (i = 0; i < nnodes; i++)
            {
                fprintf(stderr, "Collecting to master: %2d) glb_cnts = %4d glb_offs = %4d to %p\n",
                        i, glb_cnts[i], glb_offs[i], glb_buf);
            }
        }
        if (PAR(cr)) gmx_barrier(cr);
        for (i = 0; i < nnodes; i++)
        {
            if (cr->nodeid == i)
            {
                fprintf(stderr, "Sending from master: %2d) loc_cnts = %4d from %p\n",
                        i, loc_cnts, loc_buf);
            }
            if (PAR(cr)) gmx_barrier(cr);
        }
    }


    if (datatype ==  mpi_typeREAL)
    {
#ifdef GMX_DOUBLE
        datatype = mpi_typeDOUBLE;
#else
        datatype = mpi_typeFLOAT;
#endif
    }
    switch(datatype)
    {
    case mpi_typeINT:
        if (bCollect)
            MPI_Gatherv(loc_buf, loc_cnts, MPI_INT,
                        glb_buf, glb_cnts, glb_offs, MPI_INT,
                        MASTERRANK(cr), cr->mpi_comm_mygroup);
        else
            MPI_Scatterv(glb_buf, glb_cnts, glb_offs, MPI_INT,
                         loc_buf, loc_cnts, MPI_INT,
                         MASTERRANK(cr), cr->mpi_comm_mygroup);
        break;
    case mpi_typeFLOAT:
        if (bCollect)
            MPI_Gatherv(loc_buf, loc_cnts, MPI_FLOAT,
                        glb_buf, glb_cnts, glb_offs, MPI_FLOAT,
                        MASTERRANK(cr), cr->mpi_comm_mygroup);
        else
            MPI_Scatterv(glb_buf, glb_cnts, glb_offs, MPI_FLOAT,
                         loc_buf, loc_cnts, MPI_FLOAT,
                         MASTERRANK(cr), cr->mpi_comm_mygroup);
        break;
    case mpi_typeDOUBLE:
        if (bCollect)
            MPI_Gatherv(loc_buf, loc_cnts, MPI_DOUBLE,
                        glb_buf, glb_cnts, glb_offs, MPI_DOUBLE,
                        MASTERRANK(cr), cr->mpi_comm_mygroup);
        else
            MPI_Scatterv(glb_buf, glb_cnts, glb_offs, MPI_DOUBLE,
                         loc_buf, loc_cnts, MPI_DOUBLE,
                         MASTERRANK(cr), cr->mpi_comm_mygroup);
        break;
    default:
        gmx_fatal(FARGS,"Undefined datatype given to waxsDoMPIComm_qavgs_low!\n");
    }

    if (MASTER(cr))
    {
        sfree(glb_cnts);
        sfree(glb_offs);
    }
    if (bLocBufEmpty)
    {
        sfree(loc_buf);
    }
}

void
waxsDoMPIComm_qavgs(t_waxsrec *wr, t_commrec *cr, gmx_bool bCollect)
{
#ifdef GMX_MPI
    time_t begt, endt;
    int t, nprot = wr->nindA_prot;
    t_waxs_datablock wd;
    t_waxsrecType *wt;

    if (MASTER(cr))
    {
        time(&begt);
        if (bCollect)
            fprintf(stderr,"\nDoing MPI collection   of q-averages...");
        else
            fprintf(stderr,"\nDoing MPI distribution of q-averages...");
        fflush(stdout);
    }

    /* distribute/collect data in t_waxsrec */
    if (!bCollect) gmx_bcast(sizeof(double),&wr->waxsStep,cr);
    if (!wr->bVacuum)
    {
        if (!bCollect) gmx_bcast(sizeof(double), &wr->solElecDensAv, cr);
        if (!bCollect) gmx_bcast(sizeof(double), &wr->nElecAvB, cr);
    }

    /* distibute/collect data in waxs_datablock, specifit to each scatterin type */
    for (t = 0; t < wr->nTypes; t++)
    {
        wt = &wr->wt[t];
        wd = wt->wd;
        if (!bCollect) gmx_bcast(sizeof(double), &wd->normA, cr);
        waxsDoMPIComm_qavgs_low(wt, cr, wd->D,        wd->Dglobal,        mpi_typeDOUBLE, 1, bCollect);
        //fprintf(stderr, "\nXXX t = %d before %g -  %g\n", t, wd->avA[0].re,      wd->avAglobal[0].re);
        waxsDoMPIComm_qavgs_low(wt, cr, wd->avA,      wd->avAglobal,      mpi_typeDOUBLE, 2, bCollect);
        // fprintf(stderr, "\nXXX t = %d after  %g -  %g\n", t, wd->avA[0].re,      wd->avAglobal[0].re);
        waxsDoMPIComm_qavgs_low(wt, cr, wd->avAsq,    wd->avAsqglobal,    mpi_typeDOUBLE, 1, bCollect);
        waxsDoMPIComm_qavgs_low(wt, cr, wd->avA4,     wd->avA4global,     mpi_typeDOUBLE, 1, bCollect);
        waxsDoMPIComm_qavgs_low(wt, cr, wd->av_ReA_2, wd->av_ReA_2global, mpi_typeDOUBLE, 1, bCollect);
        waxsDoMPIComm_qavgs_low(wt, cr, wd->av_ImA_2, wd->av_ImA_2global, mpi_typeDOUBLE, 1, bCollect);

        if (!wr->bVacuum)
        {
            if (!bCollect) gmx_bcast(sizeof(double), &wd->normB, cr);

            waxsDoMPIComm_qavgs_low(wt, cr, wd->avB,      wd->avBglobal,      mpi_typeDOUBLE, 2, bCollect);
            waxsDoMPIComm_qavgs_low(wt, cr, wd->avBsq,    wd->avBsqglobal,    mpi_typeDOUBLE, 1, bCollect);
            waxsDoMPIComm_qavgs_low(wt, cr, wd->avB4,     wd->avB4global,     mpi_typeDOUBLE, 1, bCollect);
            waxsDoMPIComm_qavgs_low(wt, cr, wd->av_ReB_2, wd->av_ReB_2global, mpi_typeDOUBLE, 1, bCollect);
            waxsDoMPIComm_qavgs_low(wt, cr, wd->av_ImB_2, wd->av_ImB_2global, mpi_typeDOUBLE, 1, bCollect);
        }
    }

    if (MASTER(cr))
    {
        time(&endt);
        printf(" MPI communication took %d seconds.\n", (int)(endt-begt));
    }
#else
    gmx_fatal(FARGS,"MPI Communications required for checkpointing. Don't use the old functions.\n");
#endif
}

static real
droplet_volume(t_waxsrec *wr)
{
    return gmx_envelope_getVolume(wr->wt[0].envelope);
}

void
waxsEstimateNumberIndepPoints(t_waxsrec *wr, t_commrec *cr, gmx_bool bWriteACF2File,
        gmx_bool bScale_varI_now)
{
    t_waxs_datablock  wd;
    t_waxsrecType    *wt;
    double           *D, *Dglob;
    real             *acf, invexp, phi0, phi_exp1, tau_phi, corr_area, invqabs, dphi, normdq, rtmp;
    rvec              tmp;
    t_spherical_map  *qvecs;
    char              buf[1024];
    double            phi, Dav, Dav2, Dvar, tmpd;
    int               nabs, qstart, qhomenr, thisJ, i, j, jj, *ind, k, kk, iphi;
    int              *n=NULL, nbins, iset, Jmax = 0, t, nbinsmax;
    FILE             *fp = NULL;

    for (t = 0; t < wr->nTypes; t++)
    {
        fp       = NULL;
        Dglob    = NULL;
        acf      = NULL;
        n        = NULL;
        nbinsmax = 0;
        Jmax     = 0;
        iset     = 0;
        wt       = &wr->wt[t];

        wd    = wt->wd;
        D     = wd->D;
        qvecs = wt->qvecs;

        ind     = qvecs->ind;
        qstart  = qvecs->qstart;
        qhomenr = qvecs->qhomenr;
        nabs    = qvecs->nabs;
        invexp  = 1./exp(1.);

        if (MASTER(cr))
        {
            printf("\n\nEstimating the number of independent I per |q|, scattering type %d:\n", t);
        }

        if (wd->Nindep == NULL)
        {
            snew(wd->Nindep, wt->nq);
        }

        if (bWriteACF2File && MASTER(cr))
        {
            sprintf(buf, "acf_step_%d.dat", wr->waxsStep);
            /* avoid use of oenv here - so we don't have to pass it across the functions */
            fp = ffopen(buf, "w");
            fprintf(fp, "@    title \"ACF of D(q) vs. angle\"\n"
                    "@    xaxis  label \"\xf\f{} [Rad]\"\n"
                    "@    yaxis  label \"ACF\"\n@type xy\n");
        }

        for(i=0; i<nabs; i++)
        {
            thisJ    = ind[i+1]-ind[i];
            nbins    = (int) floor(M_PI/(2*sqrt(M_PI/thisJ)));
            Jmax     = (thisJ > Jmax) ? thisJ : Jmax;
            nbinsmax = (nbins > nbinsmax) ? nbins : nbinsmax;
        }
        snew(Dglob, Jmax);
        snew(acf, nbinsmax);
        snew(n,   nbinsmax);

        /* loop over absolut q values */
        for (i = 0; i < nabs; i++)
        {
            thisJ = ind[i+1]-ind[i];
            for (j = 0; j < Jmax; j++)
            {
                Dglob[j] = 0.;
            }

            /* Loop over q-vectors with fixed |q|
               Only over overlap of [qstart,qend) and [ind[i],ind[i+1]) */
            for (j = ind[i]; j < ind[i+1]; j++)
            {
                if (j >= qstart && j < (qstart+qhomenr))
                {
                    jj = j - qstart;
                    k  = j - ind[i];
                    if (k < 0 || k >= thisJ)
                    {
                        gmx_fatal(FARGS, "Error in waxsEstimateNumberIndepPoints(). Found k = %d (thisJ = %d)",
                                k, thisJ);
                    }
                    Dglob[k] = D[jj];
                }
            }

            if (PAR(cr))
            {
                gmx_sumd(thisJ, Dglob, cr);
            }

            if (MASTER(cr))
            {
                if (thisJ == 1)
                {
                    wd->Nindep[i] = 1.;
                }
                else
                {
                    if (thisJ < 20)
                    {
                        fprintf(stderr,"\n\nWARNING while estimating number of independent points.\n"
                                "have only J = %d points. This may get inaccurate.\n\n", thisJ);
                    }

                    /* 2*sqrt(pi/J) is roughly the angle betwen adjacent q-vectors */
                    nbins = (int) floor(M_PI/(2*sqrt(M_PI/thisJ)));
                    if (nbins > nbinsmax || thisJ > Jmax)
                    {
                        gmx_fatal(FARGS, "Error - this should not happen\n");
                    }
                    dphi = M_PI/nbins;

                    Dav  = 0.;
                    Dav2 = 0.;
                    Dvar = 0.;
                    for (k = 0; k < thisJ; k++)
                    {
                        Dav  += Dglob[k];
                        /*if (i==1 || i==2)
                          printf("D%d =  %d  %g\n", i, k, Dglob[k]);*/
                    }
                    Dav  /= thisJ;
                    for (k = 0; k < thisJ; k++)
                    {
                        Dvar += dsqr(Dglob[k]-Dav);
                    }
                    Dvar /= thisJ;
                    // printf("Dav  = %g  %g\n", Dav, Dvar);

                    for (iphi = 0; iphi<nbins; iphi++)
                    {
                        acf[iphi] = 0.;
                        n  [iphi] = 0;
                    }
                    invqabs = 1./qvecs->abs[i];
                    for (k = 0; k < thisJ; k++)
                    {
                        //fprintf(stderr, "iphi %d nbins %d, nbinsmax %d, k %d kk %d\n", iphi, nbins, nbinsmax, k, kk);
                        for (kk = k; kk < thisJ; kk++)
                        {
                            /* Get angle between vector k and kk */
                            /*rvec_sub(qvecs->q[ind[i]+k], qvecs->q[ind[i]+kk], tmp);
                              normdq = norm(tmp);
                              phi = 2*asin(0.5*normdq*invqabs);
                             */
                            tmpd = iprod(qvecs->q[ind[i]+k], qvecs->q[ind[i]+kk]) * invqabs * invqabs;
                            /* Catch rounding errors */
                            if (tmpd > 1)
                                phi = 0.;
                            else if (tmpd < -1)
                                phi = M_PI;
                            else
                                phi = acos(tmpd);
                            iphi = (int) (round(phi/dphi));
                            if (iphi < nbins)
                            {
                                acf[iphi] += (Dglob[k]-Dav)*(Dglob[kk]-Dav);
                                n  [iphi] ++;
                            }
                        }
                    }
                    for (iphi = 0; iphi<nbins; iphi++)
                    {
                        /* Normalize to one at phi == 0 */
                        acf[iphi] /= (n[iphi]*Dvar);
                    }
                    /* Check where acf drops below 1/e, and interpolate with prev. point */
                    iphi = 0;
                    while (acf[++iphi] > invexp);
                    phi0 = (iphi-1)*dphi;
                    phi_exp1 = phi0 + dphi * (invexp-acf[iphi-1])/(acf[iphi]-acf[iphi-1]);

                    /* Assume that acf = exp(-phi/tau) * 0.5*(3cos^2(x) - 1) */
                    rtmp = 2*invexp / (3*sqr(cos(phi_exp1))-1);
                    tau_phi = (rtmp > 1e-20) ? -phi_exp1/log(rtmp) : 50.;

                    /* - autocorrelation area =
                       2pi * int_0^Pi/2 dtheta sin(theta) * exp(-theta/tau)
                       = 2pi tau * (tau-exp(-pi/2tau))/(1+tau^2)
                       - muliply by 2 because there is additional correlation on the opposite
                       side of the sphere
                     */
                    corr_area = 2 * 2*M_PI * tau_phi*(tau_phi-exp(-M_PI/(2*tau_phi)))/(1+sqr(tau_phi));
                    if (corr_area > 4*M_PI)
                    {
                        corr_area = 4*M_PI;
                    }
                    else if (corr_area < 4*M_PI/thisJ)
                    {
                        corr_area = 4*M_PI/thisJ;
                    }
                    wd->Nindep[i] = 4*M_PI / corr_area;

                    if (bWriteACF2File)
                    {
                        fprintf(fp, "@type xy\n@   s%d legend  \"i = %d, q = %4g, \\xt\\f{}=%g, N\\sind\\N=%.1f\"\n",
                                iset++, i, qvecs->abs[i], tau_phi, wd->Nindep[i]);
                        for (iphi = 0; iphi < nbins; iphi++)
                        {
                            fprintf(fp, "%g  %g   %d\n", iphi*dphi, acf[iphi], n[iphi]);
                        }
                    }
                }
                printf("\tACF %2d of %d (q = %8g): %6.1f of %d points independent\n", i, nabs, qvecs->abs[i],
                        wd->Nindep[i], thisJ);
            }
        }
        if (MASTER(cr))
        {
            printf("\n");
        }
        sfree(Dglob);
        sfree(acf);
        sfree(n);

        if (fp && MASTER(cr))
        {
            ffclose(fp);
            printf("Wrote autocorrelation functions of D(q) to %s for each |q|\n", buf);
        }

        if (PAR(cr))
        {
            gmx_bcast(wt->nq*sizeof(double), wd->Nindep, cr);
        }

        if (bScale_varI_now)
        {
            if (MASTER(cr))
            {
                printf("Scaling the variance of I(q) by 1/# independent data points\n");
            }
            for (i = 0; i < nabs; i++)
            {
                /* devide variance by # independent q-vectors */
                wd->varI[i] /= wd->Nindep[i];
            }
        }
    }

    wr->bHaveNindep = TRUE;
}


static void init_waxs_output(t_waxsrec *wr, const char *fnOut, output_env_t oenv)
{
    int len, i, t;
    char *base, typeStr[STRLEN], title[STRLEN];
    real dq;
    const int lenNameExt = 50;


    /* Get base name of output (without .dat/.xvg extension) */
    len = strlen(fnOut);
    snew(base, len);
    strncpy(base, fnOut, len-4);
    base[len-4] = '\0';
    fprintf(stderr, "WAXS-MD: Base file name for output: %s\n", base);

    /*
     * Output that needs only one output file for all scattering types
     */
    snew(wr->wo, 1);
    snew(wr->wo->fnLog,     len+lenNameExt);
    snew(wr->wo->fnExpRMSD, len+lenNameExt);
    snew(wr->wo->fnDensity, len+lenNameExt);

    sprintf(wr->wo->fnLog,     "%s.log",         base);
    sprintf(wr->wo->fnDensity, "%s_density.dat", base);

    wr->wo->fpLog = ffopen(wr->wo->fnLog, "w");
    if (wr->bRotFit )
    {
        fprintf(wr->wo->fpLog,"NB: WAXS-rotational fit has been turned on.\n");
    }

    /* Writing exponential average RMSD at each time point into xvg */
    if (wr->bDampenForces )
    {
        sprintf(wr->wo->fnExpRMSD, "%s_exprmsd.xvg", base);
        fprintf(stderr,"Writing RMSD exponential averages into %s \n", wr->wo->fnExpRMSD);
        wr->wo->fpExpRMSD = xvgropen(wr->wo->fnExpRMSD,
                "RMSD to past ensemble and its running exp. avg.",
                "t [ps]", "RMSD [nm]", oenv);
        fprintf(wr->wo->fpExpRMSD, "@ s%d legend \"RMSD to average position\"\n", 0);
        fprintf(wr->wo->fpExpRMSD, "@ s%d legend \"exp. average RMSD\"\n", 1);
        fprintf(wr->wo->fpExpRMSD, "@ s%d legend \"exp. std. dev. RMSD\"\n", 2);
    }
    else
    {
        wr->wo->fpExpRMSD = NULL;
    }


    /*
     * Output that needs one output file for each scattering type
     */
    snew(wr->wo->fnSpectra,              wr->nTypes);
    snew(wr->wo->fnStddevs,              wr->nTypes);
    snew(wr->wo->fnFinal,                wr->nTypes);
    snew(wr->wo->fnContrib,              wr->nTypes);
    snew(wr->wo->fnNindep,               wr->nTypes);
    snew(wr->wo->fnD,                    wr->nTypes);
    snew(wr->wo->fnForces,               wr->nTypes);
    snew(wr->wo->fnSpecEns,              wr->nTypes);
    snew(wr->wo->fnAvgA,                 wr->nTypes);
    snew(wr->wo->fnAvgB,                 wr->nTypes);
    snew(wr->wo->fnPot,                  wr->nTypes);
    snew(wr->wo->fnGibbsEnsW,            wr->nTypes);
    snew(wr->wo->fnGibbsSolvDensRelErr,  wr->nTypes);

    snew(wr->wo->fpSpectra,              wr->nTypes);
    snew(wr->wo->fpStddevs,              wr->nTypes);
    snew(wr->wo->fpNindep,               wr->nTypes);
    snew(wr->wo->fpD,                    wr->nTypes);
    snew(wr->wo->fpForces,               wr->nTypes);
    snew(wr->wo->fpSpecEns,              wr->nTypes);
    snew(wr->wo->fpAvgA,                 wr->nTypes);
    snew(wr->wo->fpAvgB,                 wr->nTypes);
    snew(wr->wo->fpPot,                  wr->nTypes);
    snew(wr->wo->xfout,                  wr->nTypes);
    snew(wr->wo->fpGibbsEnsW,            wr->nTypes);
    snew(wr->wo->fpGibbsSolvDensRelErr,  wr->nTypes);

    for (t = 0; t<wr->nTypes; t++)
    {
        if (wr->nTypes > 1)
        {
            sprintf(typeStr, "_%d", t);
        }
        else
        {
            strcpy(typeStr, "");
        }

        snew(wr->wo->fnSpectra[t], len+lenNameExt);
        snew(wr->wo->fnStddevs[t], len+lenNameExt);
        snew(wr->wo->fnFinal  [t], len+lenNameExt);
        snew(wr->wo->fnContrib[t], len+lenNameExt);
        snew(wr->wo->fnNindep [t], len+lenNameExt);
        snew(wr->wo->fnD      [t], len+lenNameExt);
        snew(wr->wo->fnForces [t], len+lenNameExt);
        snew(wr->wo->fnSpecEns[t], len+lenNameExt);
        snew(wr->wo->fnAvgA   [t], len+lenNameExt);
        snew(wr->wo->fnAvgB   [t], len+lenNameExt);
        snew(wr->wo->fnPot    [t], len+lenNameExt);
        snew(wr->wo->fnGibbsEnsW[t], len+lenNameExt);
        snew(wr->wo->fnGibbsSolvDensRelErr[t], len+lenNameExt);

        sprintf(wr->wo->fnSpectra            [t], "%s_spectra%s.xvg",          base, typeStr);
        sprintf(wr->wo->fnSpecEns            [t], "%s_spectra_ensemlbe%s.xvg", base, typeStr);
        sprintf(wr->wo->fnStddevs            [t], "%s_stddevs%s.xvg",          base, typeStr);
        sprintf(wr->wo->fnFinal              [t], "%s_final%s.xvg",            base, typeStr);
        sprintf(wr->wo->fnContrib            [t], "%s_contrib%s.xvg",          base, typeStr);
        sprintf(wr->wo->fnNindep             [t], "%s_nindep%s.xvg",           base, typeStr);
        sprintf(wr->wo->fnD                  [t], "%s_SF%s.xvg",               base, typeStr);
        sprintf(wr->wo->fnForces             [t], "%s_forces%s.trr",           base, typeStr);
        sprintf(wr->wo->fnAvgA               [t], "%s_averageA%s.xvg",         base, typeStr);
        sprintf(wr->wo->fnAvgB               [t], "%s_averageB%s.xvg",         base, typeStr);
        sprintf(wr->wo->fnPot                [t], "%s_pot%s.xvg",              base, typeStr);
        sprintf(wr->wo->fnGibbsEnsW          [t], "%s_ensembleWeights%s.xvg",  base, typeStr);
        sprintf(wr->wo->fnGibbsSolvDensRelErr[t], "%s_solvDensUncert%s.xvg",   base, typeStr);

        sprintf(title, "%s curves", wr->wt[t].saxssansStr);
        wr->wo->fpSpectra[t] = xvgropen(wr->wo->fnSpectra[t], title, "q [nm\\S-1\\N]", "I [e\\S2\\N]", oenv);

        if (wr->bPrintForces)
        {
            wr->wo->xfout[t]     = open_trn(wr->wo->fnForces[t], "w");
            fprintf(wr->wo->fpLog,"NB: Printing coordinates and forces at each waxs-step (scattering type %d, %s).\n", t, wr->wt[t].saxssansStr);
        }
        if (WAXS_ENSEMBLE(wr))
        {
            sprintf(title, "%s curves of ensemble", wr->wt[t].saxssansStr);
            wr->wo->fpSpecEns[t] = xvgropen(wr->wo->fnSpecEns[t], title, "q [nm\\S-1\\N]", "I [e\\S2\\N]", oenv);
        }
        if (wr->ewaxs_ensemble_type == ewaxsEnsemble_BayesianOneRefined)
        {
            wr->wo->fpGibbsEnsW[t] = xvgropen(wr->wo->fnGibbsEnsW[t], "Ensemble weights", "t [ps]", "weights", oenv);
        }
        if (wr->bBayesianSolvDensUncert)
        {
            sprintf(title, "Relative solvent density uncertainty");
            wr->wo->fpGibbsSolvDensRelErr[t] = xvgropen(wr->wo->fnGibbsSolvDensRelErr[t],
                                                         "Relative solvent density uncertainty", "t [ps]",
                                                        "\\xdr\\f{}\\sbuf\\N / \\xr\\f{}\\sbuf", oenv);
        }

        if (wr->bCalcPot )
        {
            fprintf(stderr, "Writing %s potentials for each q into %s (scattering type %d) \n", wr->wt[t].saxssansStr, wr->wo->fnPot[t], t);
            sprintf(title, "%s potentials at individual q's", wr->wt[t].saxssansStr);
            wr->wo->fpPot[t] = xvgropen(wr->wo->fnPot[t], title, "t [ps]", "E [kJ mol\\S-1\\N]", oenv);

            dq = (wr->wt[t].nq > 1) ? (wr->wt[t].maxq-wr->wt[t].minq)/(wr->wt[t].nq - 1) : 0.0 ;
            for (i = 0; i < wr->wt[t].nq; i++)
            {
                fprintf(wr->wo->fpPot[t], "@ s%d legend \"q = %.2f\"\n", i, wr->wt[t].minq + dq*i);
            }
        }
        if (wr->weightsType != ewaxsWeightsUNIFORM)
        {
            /* Open file with standard deviations of intensities */
            sprintf(title, "Stddevs entering %s potential", wr->wt[t].saxssansStr);
            wr->wo->fpStddevs[t] = xvgropen(wr->wo->fnStddevs[t], title, "q [nm\\S-1\\N]", "\\xs\\f{} [e\\S2\\N]", oenv);
            i = 0;
            fprintf(wr->wo->fpStddevs[t], "@ s%d legend \"\\xs\\f{}(I\\scalc\\N)\"\n", i++);
            if (wr->wt[t].Iexp_sigma)
            {
                fprintf(wr->wo->fpStddevs[t], "@ s%d legend \"\\xs\\f{}(I\\sexp\\N)\"\n", i++);
            }
            if (wr->solventDensRelErr)
            {
                fprintf(wr->wo->fpStddevs[t], "@ s%d legend \"\\xs\\f{}(I\\sbuf\\N)\"\n", i++);
            }
        }
    }
    sfree(base);
}

static void
write_averages(t_waxsrecType *wt, const char *fnAvgA, const char *fnAvgB)
{
    int i, j;
    FILE *fp;

    /* Write Average Scattering amplitude A(vec{q}) */
    fp = ffopen(fnAvgA, "w");
    fprintf(fp,"# %10s = %d\n", "nabs",wt->qvecs->nabs);
    fprintf(fp,"# %10s %12s %12s  %12s %12s %12s %12s %12s %12s\n", "qx", "qy", "qz",
            "< Re A >", "< Im A >", "< |A|^2 >", "< |A|^4 >", "< (Re A)^2 >", "< (Im A)^2 >");

    for (i = 0; i < wt->qvecs->nabs; i++)
    {
        fprintf(fp, "# %12s = %d\n",   "indexofqabs", i);
        fprintf(fp, "# %12s = %10g\n", "qabs",        wt->qvecs->abs[i]);
        fprintf(fp, "# %12s = %d\n",   "nofqvec",     wt->qvecs->ind[i+1] - wt->qvecs->ind[i]);

        for (j = wt->qvecs->ind[i]; j < wt->qvecs->ind[i+1]; j++)
        {
            fprintf(fp,"%12g %12g %12g  %12g %12g %12g %12g %12g %12g\n", wt->qvecs->q[j][XX], wt->qvecs->q[j][YY],
                    wt->qvecs->q[j][ZZ], wt->wd->avAglobal[j].re, wt->wd->avAglobal[j].im,wt->wd->avAsqglobal[j],
                    wt->wd->avA4global[j],wt->wd->av_ReA_2global[j],wt->wd->av_ImA_2global[j]);
        }
        fprintf(fp,"\n\n");
    }
    printf("Wrote all averages of scattering amplitudes of System A to %s\n", fnAvgA);
    ffclose(fp);

    if (fnAvgB)
    {
        fp = ffopen(fnAvgB, "w");
        fprintf(fp,"# %10s = %d\n", "nabs",wt->qvecs->nabs);
        fprintf(fp,"# %8s %10s %10s  %10s %10s %10s %10s %10s %10s\n", "qx", "qy", "qz",
                "< Re B >", "< Im B >", "< |B|^2 >", "< |B|^4 >", "< (Re B)^2 >", "< (Im B)^2 >");

        for (i = 0; i < wt->qvecs->nabs; i++)
        {
            fprintf(fp,"# %10s = %d\n",   "indexofqabs", i);
            fprintf(fp,"# %10s = %10g\n", "qabs",        wt->qvecs->abs[i]);
            fprintf(fp,"# %10s = %d\n",   "nofqvec",     wt->qvecs->ind[i+1] - wt->qvecs->ind[i]);

            for (j = wt->qvecs->ind[i]; j < wt->qvecs->ind[i+1]; j++)
            {
                fprintf(fp,"%12g %12g %12g  %10g %10g %10g %10g %10g %10g\n", wt->qvecs->q[j][XX], wt->qvecs->q[j][YY],
                        wt->qvecs->q[j][ZZ], wt->wd->avBglobal[j].re, wt->wd->avBglobal[j].im,wt->wd->avBsqglobal[j],
                        wt->wd->avB4global[j],wt->wd->av_ReB_2global[j],wt->wd->av_ImB_2global[j]);
            }
            fprintf(fp,"\n\n");
        }
        printf("Wrote all averages of scattering amplitudes of System B to %s\n", fnAvgB);
        ffclose(fp);
    }
}

static void
write_stddevs(t_waxsrec *wr, gmx_large_int_t step, int t)
{
    int           i;
    t_waxsrecType *wt;
    FILE          *fp;

    wt = &wr->wt[t];
    fp = wr->wo->fpStddevs[t];

    fprintf(fp, "\n# Std deviations %d simulation step ", wr->waxsStep);
    fprintf(fp, gmx_large_int_pfmt, step);
    fprintf(fp, "\n");

    for (i = 0; i < wt->nq; i++)
    {
        fprintf(fp, "%8g  %12g", wt->qvecs->abs[i], sqrt(wt->wd->varI[i]));
        if (wt->Iexp_sigma)
        {
            fprintf(fp, "  %12g", wt->f_ml * wt->Iexp_sigma[i]);
        }
        if (wt->wd->I_errSolvDens)
        {
            fprintf(fp, "  %12g", fabs(wr->epsilon_buff * wt->wd->I_errSolvDens[i]));
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    fflush(fp);
}

void
write_intensity(FILE *fp, t_waxsrec *wr, int type)
{
    int i, t;
    t_waxsrecType *wt = &wr->wt[type];

    switch(wr->ewaxsaniso)
    {
    case ewaxsanisoNO:
        fprintf(fp, "\n@type xydy\n");
        for (i = 0; i < wt->nq; i++)
        {
            fprintf(fp, "%8g  %12g %12g %12g %12g %12g\n", wt->qvecs->abs[i], wt->wd->I[i],
                    sqrt(wt->wd->varI[i]),
                    !wr->bVacuum ? wt->wd->I_avAmB2[i] : 0.0,
                    !wr->bVacuum ? wt->wd->I_varA  [i] : 0.0,
                    !wr->bVacuum ? wt->wd->I_varB  [i] : 0.0);
        }
        break;
    case ewaxsanisoYES:
        printf("nIvalues = %d\n", wt->wd->nIvalues);
        /* We assume that the beam comes from x - so write as I(qy, qz) */
        for (i = 0; i < wt->wd->nIvalues; i++)
        {
            fprintf(fp, "%8g %8g %12g\n", wt->qvecs->q[i][YY], wt->qvecs->q[i][ZZ], wt->wd->I[i]);
        }
        break;
    default:
        gmx_fatal(FARGS, "This anisotropy (%d) is not supported (in write_intensity)\n", wr->ewaxsaniso);
    }
}

/* wrinte I vs. q to grace file, and optionally the error.
   With bVariance == TRUE, take sqrt of error before writing the error column.
   fact ( = 1/sqrt(N) ) allows to switch from stddev to stddev/sqrt(N)
 */
void print_Ivsq(FILE *fp, t_waxsrecType *wt, double *I, double *Ierror, gmx_bool bVariance, double fact, char *type)
{
    int i;
    real dq = (wt->nq > 1) ? (wt->maxq - wt->minq)/(wt->nq - 1) : 0.0;
    real err;

    if (type)
    {
        fprintf(fp, "%s\n", type);
    }
    for (i = 0; i < wt->nq; i++)
    {
        fprintf(fp, "%12g  %12g", wt->minq + i*dq, fact*I[i]);
        if (Ierror)
        {
            err = bVariance ? sqrt(Ierror[i]) : Ierror[i];
            fprintf(fp, "  %12g\n", err);
        }
        else
        {
            fprintf(fp, "\n");
        }
    }
    fprintf(fp, "&\n");
}

void
done_waxs_output(t_waxsrec *wr, output_env_t oenv)
{
    int              i, j, s, t;
    real             dq, compwater, r2;
    double           stddev, RgGuinier;
    t_waxs_datablock wd;
    t_waxs_output    wo = wr->wo;
    t_waxsrecType   *wt;
    FILE            *fp;
    rvec             cent;
    char             title[STRLEN];

    fprintf(stderr,"\nClosing WAXS-MD output\n\n");

    for (t = 0; t<wr->nTypes; t++)
    {
        wt = &wr->wt[t];
        wd = wt->wd;
        dq = (wt->nq > 1) ? (wt->maxq - wt->minq)/(wt->nq - 1) : 0.0 ;

        ffclose(wo->fpSpectra[t]);
        if (wo->fpStddevs[t])
        {
            ffclose(wo->fpStddevs[t]);
        }
        if (wo->fpSpecEns[t])
        {
            ffclose(wo->fpSpecEns[t]);
        }

        if (wr->bCalcPot)
        {
            fprintf(stderr,"\nWriting average %s potentials into %s, group %d\n", wt->saxssansStr, wo->fnLog, t);
            fprintf(wo->fpLog, "\n\n### Average %s-%d Potentials\n", wt->saxssansStr, t);
            fprintf(wo->fpLog, "@    title \"%s potentials: average and stddev\"\n", wt->saxssansStr);
            fprintf(wo->fpLog, "@    xaxis  label \"q [nm\\S-1\\N]\"\n");
            fprintf(wo->fpLog, "@    yaxis  label \"E [kJ mol\\S-1\\N]\"\n");
            fprintf(wo->fpLog,"@type xydy\n");
            for (i=0; i < wt->nq; i++)
            {
                /* Get standard deviation */
                stddev = wd->vAver2[i] - sqr(wd->vAver[i]);
                stddev = ( (stddev>0.) ? sqrt(stddev) : 0.);
                fprintf(wo->fpLog,"%8g  %10g  %10g\n", wt->minq + i*dq, wd->vAver[i], stddev);
            }
            fprintf(wo->fpLog, "### Use: sed -n '/Average %s-%d Pot/,/End average %s-%d/p' < %s\n",
                    wt->saxssansStr, t, wt->saxssansStr, t, wo->fnLog);
            fprintf(wo->fpLog, "### End average %s-%d Potentials\n\n", wt->saxssansStr, t);
        }

        /* Write final spectrum and target spectrum to xxx_final.xvg */
        fprintf(stderr,"Writing final spectrum to %s (group %s %d)\n", wo->fnFinal[t], wt->saxssansStr, t);
        switch(wr->ewaxsaniso)
        {
        case ewaxsanisoNO:
            sprintf(title, "Final %s intensity", wt->saxssansStr);
            fp = xvgropen(wo->fnFinal[t], title, "q [nm\\S-1\\N]","Intensity [e\\S2\\N]", oenv);
            if (wt->type == escatterNEUTRON)
            {
                fprintf(fp, "@    subtitle \"Deuterium concentration %g %%\"\n", wt->deuter_conc*100);
            }
            fprintf(fp, "@    s0 legend  \"Final intensity\"\n");
            print_Ivsq(fp, wt, wd->I, wd->varI, TRUE, 1., "@type xydy");
            if (wt->Iexp)
            {
                /* Also write the target curve */
                fprintf(fp,  "@    s1 legend  \"Target intensity\"\n@type xydy\n");
                print_Ivsq(fp, wt, wt->Iexp,  wt->Iexp_sigma, FALSE, 1, "@type xydy");
                fprintf(stderr, "Wrote target intensity to %s\n", wo->fnFinal[t]);
                if (wr->ewaxs_Iexp_fit != ewaxsIexpFit_NO)
                {
                    /* Write target curve corrected by maximum likelihood estimates scale f and offset c */
                    fprintf(fp, "@    s2 legend  \"ML Target\"\n@type xydy\n");
                    fprintf(fp, "@type xydy\n");
                    for (i = 0; i<wt->nq; i++)
                    {
                        fprintf(fp, "%12g  %12g  %12g\n", wt->minq + i*dq, wt->f_ml * wt->Iexp[i] + wt->c_ml, wt->f_ml * wt->Iexp_sigma[i]);
                    }
                    fprintf(fp, "&\n");
                    fprintf(stderr, "Wrote maximum-likelihood-scaled target intensity to %s\n", wo->fnFinal[t]);
                }
            }
            ffclose(fp);
            break;
        case ewaxsanisoYES:
            fp = ffopen(wo->fnFinal[t], "w");
            write_intensity(fp, wr, t);
            ffclose(fp);
            break;
        default:
            gmx_fatal(FARGS, "This anisotropy (%d) is not supported (in done_waxs_output)\n", wr->ewaxsaniso);
        }
        fp = NULL;

        /* Write D(vec{q}), see eq. 10 of Chen/Hub, Biophys J, 2014 */
        fp = ffopen(wo->fnD[t], "w");
        fprintf(fp, "# %8s %10s %10s  %10s\n", "qx", "qy", "qz", "Intensity(vec{q})");
        for (i = 0; i < wt->qvecs->nabs; i++)
        {
            fprintf(fp, "# iq %2d  --  |q| = %g\n", i, wt->qvecs->abs[i]);
            for (j = wt->qvecs->ind[i]; j < wt->qvecs->ind[i+1]; j++)
            {
                fprintf(fp, "%12g %12g %12g  %10g\n", wt->qvecs->q[j][XX], wt->qvecs->q[j][YY],
                        wt->qvecs->q[j][ZZ], wt->wd->Dglobal[j]);
            }
            fprintf(fp, "\n\n");
        }
        printf("Wrote all scattering intensities D(q) to %s\n", wo->fnD[t]);
        ffclose(fp);

        /* Write averages of A and B into files */
        write_averages(&wr->wt[t], wo->fnAvgA[t], wr->bVacuum ? NULL : wo->fnAvgB[t]);

        /* Write file with contributions to I(q) */
        switch(wr->ewaxsaniso)
        {
        case ewaxsanisoNO:
            fprintf(stderr,"Writing contributions to I(q) to %s (group %s %d)\n", wo->fnContrib[t], wt->saxssansStr, t);
            sprintf(title, "Contributions to %s pattern", wt->saxssansStr);
            fp = xvgropen(wo->fnContrib[t], title, "q [nm\\S-1\\N]", "Intensity [e\\S2\\N]", oenv);
            if (wt->type == escatterNEUTRON)
            {
                fprintf(fp, "@    subtitle \"Deuterium concentration %g %%\"\n", wt->deuter_conc*100);
            }
            if (! wr->bVacuum)
            {
                fprintf(fp,
                        "@    s0 legend  \"<|A(q)|\\S2\\N>\"\n"
                        "@    s1 legend  \"<|B(q)|\\S2\\N>\"\n"
                        "@    s2 legend  \"2Re[ -<B\\S*\\N(q)> <A(q) - B(q)> ]\"\n"
                        "@    s3 legend  \"Re <A(|q|)>\"\n"
                        "@    s4 legend  \"Im <A(|q|)>\"\n"
                        "@    s5 legend  \"Re <B(|q|)>\"\n"
                        "@    s6 legend  \"Im <B(|q|)>\"\n"
                        "@    s7 legend  \"|<A-B>|\\S2\"\n"
                        "@    s8 legend  \"var(A)\"\n"
                        "@    s9 legend  \"var(B)\"\n"
                );
                s = 9;
                if (wr->bScaleI0)
                {
                    fprintf(fp,
                            "@    s%d legend  \"2Re[ \\xd\\f{}A\\S*\\N<A-B>] + |\\xd\\f{}A|\\S2\\N\"\n", ++s);
                }
                if (wr->bCorrectBuffer)
                {
                    fprintf(fp, "@    s%d legend  \"I\\sbuffcorr\\N\"\n", ++s);
                }
                if (wd->I_errSolvDens)
                {
                    fprintf(fp, "@    s%d legend  \"\\xD\\f{}I\\ssolvdens\\N\"\n", ++s);
                }
                for (i=0; i<=s; i++)
                {
                    fprintf(fp, "@    s%d errorbar size 0.250000\n", i);
                }
                /* For these, we have devided by the # of frames already in calculate_I_dkI()
                   Therefore, use factor of 1.0 instead of f. */
                print_Ivsq(fp, wt, wd->IA,    wd->varIA,          TRUE, 1., "@type xydy");
                print_Ivsq(fp, wt, wd->IB,    wd->varIB,          TRUE, 1., "@type xydy");
                print_Ivsq(fp, wt, wd->Ibulk, wd->varIbulk,       TRUE, 1., "@type xydy");
                print_Ivsq(fp, wt, wd->avAqabs_re, NULL,          TRUE, 1., "@type xy");
                print_Ivsq(fp, wt, wd->avAqabs_im, NULL,          TRUE, 1., "@type xy");
                print_Ivsq(fp, wt, wd->avBqabs_re, NULL,          TRUE, 1., "@type xy");
                print_Ivsq(fp, wt, wd->avBqabs_im, NULL,          TRUE, 1., "@type xy");
                print_Ivsq(fp, wt, wd->I_avAmB2, wd->varI_avAmB2, TRUE, 1., "@type xydy");
                print_Ivsq(fp, wt, wd->I_varA,   wd->varI_varA,   TRUE, 1., "@type xydy");
                print_Ivsq(fp, wt, wd->I_varB,   wd->varI_varB,   TRUE, 1., "@type xydy");
                if (wr->bScaleI0)
                {
                    print_Ivsq(fp, wt, wd->I_scaleI0, NULL,   TRUE, 1., "@type xy");
                }
                if (wr->bCorrectBuffer && t == 0)
                {
                    /* Ipuresolv presently only available with one scattering group */
                    print_Ivsq(fp, wt, wt->Ipuresolv, NULL,   TRUE, wr->soluteVolAv, "@type xy");
                }
                if (wd->I_errSolvDens)
                {
                    print_Ivsq(fp, wt, wd->I_errSolvDens, NULL,   TRUE, 1., "@type xy");
                }
            }
            else
            {
                fprintf(fp,
                        "@    s0 legend  \"<|A(q)|\\S2\\N>\"\n"
                        "@    s1 legend  \"Re <A(|q|)>\"\n"
                        "@    s2 legend  \"Im <A(|q|)>\"\n");
                print_Ivsq(fp, wt, wd->IA,    wd->varIA, TRUE, 1, "@type xydy");
                print_Ivsq(fp, wt, wd->avAqabs_re, NULL, TRUE, 1, "@type xy");
                print_Ivsq(fp, wt, wd->avAqabs_im, NULL, TRUE, 1, "@type xy");
            }
            ffclose(fp);
            fp = NULL;

            if (wd->Nindep)
            {
                fp = xvgropen(wo->fnNindep[t], "# of independent I(q)",
                        "q [nm\\S-1\\N]", "# independent", oenv);
                print_Ivsq(fp, wt, wd->Nindep, NULL, TRUE, 1., "@type xy");
                ffclose(fp);
            }

            break;
        case ewaxsanisoYES:
        case ewaxsanisoCOS2:
            printf("\nDo not (yet) write contributions to I(q) with anisotropic scattering\n");
            break;
        }

        if (wo->fpPot[t])
        {
            ffclose(wo->fpPot[t]);
        }

        if (wr->bPrintForces)
        {
            close_trn(wo->xfout[t]);
        }
    }


    /*
     * Closing output that is genetic to all scattering types
     */
    if (wo->fpExpRMSD)
    {
        fprintf(wo->fpExpRMSD, "&\n");
        ffclose(wo->fpExpRMSD);
    }

    /* Write computig time statistics to log */
    waxsTimingWrite(wr->compTime, wo->fpLog);

    fprintf(wo->fpLog,"\n\n======== WAXS MD STATISTICS ======\n");
    print2log(wo->fpLog, "Nr of waxs steps", "%d", wr->waxsStep);
    print2log(wo->fpLog, "Average nr of solvation shell atoms", "%g", wr->nAtomsLayerAver);
    print2log(wo->fpLog, "Average nr of excluded volume atoms", "%g", wr->nAtomsExwaterAver);

    print2log(wo->fpLog, "Average nr of excluded volume atoms", "%g", wr->nAtomsExwaterAver);
    print2log(wo->fpLog, "\nAverage nr of electrons in A",      "%g", wr->nElecAvA);
    print2log(wo->fpLog, "Stddev  nr of electrons in A",        "%g", sqrt(wr->nElecAv2A-sqr(wr->nElecAvA)));
    print2log(wo->fpLog, "Stddev/sqrt(N) nr of electrons in A", "%g",
            sqrt(wr->nElecAv2A-sqr(wr->nElecAvA))/sqrt(wr->waxsStep));
    print2log(wo->fpLog, "Average nr^2  of electrons in A", "%g", wr->nElecAv2A);
    print2log(wo->fpLog, "Stddev (nr^2) of electrons in A", "%g", sqrt(wr->nElecAv4A-sqr(wr->nElecAv2A)));
    print2log(wo->fpLog, "Stddev/sqrt(N) (nr^2) of electrons in A", "%g",
            sqrt(wr->nElecAv4A-sqr(wr->nElecAv2A))/sqrt(wr->waxsStep));

    if (! wr->bVacuum)
    {
        print2log(wo->fpLog,"Approximate av. volume of solute (nm3)", "%g", wr->soluteVolAv);
        print2log(wo->fpLog,"Approximate density of solute (e/nm3)",  "%g", wr->nElecProtA/wr->soluteVolAv);
        print2log(wo->fpLog,"Volume of envelope (nm3)",               "%g", droplet_volume(wr));
        print2log(wo->fpLog,"\nAverage nr of electrons in B",         "%g", wr->nElecAvB);
        print2log(wo->fpLog,"Stddev  nr of electrons in B",           "%g", sqrt(wr->nElecAv2B-sqr(wr->nElecAvB)));
        print2log(wo->fpLog,"Stddev/sqrt(N) nr of electrons in B",    "%g",
                sqrt(wr->nElecAv2B-sqr(wr->nElecAvB))/sqrt(wr->waxsStep));
        print2log(wo->fpLog,"Average  nr^2  of electrons in B",        "%g", wr->nElecAv2B);
        print2log(wo->fpLog,"Stddev  (nr^2) of electrons in B",        "%g", sqrt(wr->nElecAv4B-sqr(wr->nElecAv2B)));
        print2log(wo->fpLog,"Stddev/sqrt(N) (nr^2) of electrons in B", "%g",
                sqrt(wr->nElecAv4B-sqr(wr->nElecAv2B))/sqrt(wr->waxsStep));
    }

    print2log(wo->fpLog, "\nAverage electron density in A NOT protein+solvation layer (e/nm^3)", "%g", wr->solElecDensAv);
    print2log(wo->fpLog, "Stddev  electron density in A NOT protein+solvation layer", "%g",
            sqrt(wr->solElecDensAv2-sqr(wr->solElecDensAv)));
    print2log(wo->fpLog, "Stddev/sqrt(N) electron density in A NOT protein+solvation layer", "%g",
            sqrt(wr->solElecDensAv2-sqr(wr->solElecDensAv))/sqrt(wr->waxsStep));

    compwater = 18.01528/6.002214129;
    print2log(wo->fpLog,"This corresponds to a pure H2O density of (kg/m^3)", "%g", compwater*wr->solElecDensAv);
    print2log(wo->fpLog,"Stddev     pure H2O density", "%g",
            sqrt(wr->solElecDensAv2*compwater*compwater-sqr(wr->solElecDensAv*compwater)));
    print2log(wo->fpLog,"Stddev/sqrt(N)  pure H2O density", "%g",
            sqrt(wr->solElecDensAv2*compwater*compwater-sqr(wr->solElecDensAv*compwater))/sqrt(wr->waxsStep));

    print2log(wo->fpLog, "\nAverage (electron density)^2 in A NOT protein+solvation layer", "%g", wr->solElecDensAv2);
    print2log(wo->fpLog, "Stddev  (electron density)^2 in A NOT protein+solvation layer", "%g",
            sqrt(wr->solElecDensAv4-sqr(wr->solElecDensAv2)));
    print2log(wo->fpLog, "Stddev/sqrt(N) (electron density)^2 in A NOT protein+solvation layer", "%g",
            sqrt(wr->solElecDensAv4-sqr(wr->solElecDensAv2))/sqrt(wr->waxsStep));

    print2log(wo->fpLog, "\nElectron density in the bulk of A [e/nm3]", "%g", wr->solElecDensAv);
    if (! wr->bVacuum)
    {
        print2log(wo->fpLog, "Electron density in the bulk of B [e/nm3]", "%g", wr->solElecDensAv_SysB);
    }
    if (wr->bFixSolventDensity)
    {
        print2log(wo->fpLog, "\nSolvent density was fixed to [e/nm3]:", "%g", wr->givenElecSolventDensity);
        fprintf(wo->fpLog, "Average # of electrons added to solvent in A: ");
        fprintf(wo->fpLog, "%10g (added density [e/nm3] = %g)\n", wr->nElecAddedA, wr->givenElecSolventDensity-wr->solElecDensAv);
        fprintf(wo->fpLog, "Average # of electrons added to solvent in B: %10g (added density [e/nm3] = %g)\n\n",
                wr->nElecAddedB, wr->givenElecSolventDensity-wr->solElecDensAv_SysB);
    }
    else
    {
        fprintf(wo->fpLog, "\nSovlent density was simply from the the xtc file and not fixed.\n");
    }

    for (t = 0; t < wr->nTypes; t++)
    {
        fprintf(wo->fpLog, "\nScattering group %d (%s)", t, wr->wt[t].saxssansStr);
        if (wr->wt[t].type == escatterNEUTRON)
        {
            fprintf(wo->fpLog, ", %g %% D2O\n", 100*wr->wt[t].deuter_conc);
        }
        else
        {
            fprintf(wo->fpLog, "\n");
        }
        print2log(wo->fpLog, "Average number of electrons or NSL in A", "%g", wr->wt[t].wd->avAsum);
        print2log(wo->fpLog, "Average number of electrons or NSL in B", "%g", wr->wt[t].wd->avBsum);
        print2log(wo->fpLog, "Solvent density (electrons or NSL per nm3)", "%g",
                  wr->waxs_solv->avScattLenDens[t]);
        print2log(wo->fpLog, "Approx. contrast between solute and buffer (electrons or NSL per nm3)", "%g", wr->wt[t].soluteContrast);
        print2log(wo->fpLog, "Contrast", "%s", (wr->wt[t].wd->avAsum > wr->wt[t].wd->avBsum) ? "positive" : "negative");
    }

    for (t = 0; t < wr->nTypes; t++)
    {
        if (wr->wt[t].type == escatterNEUTRON)
        {
            fprintf(wo->fpLog, "\nScattering group %d (Neutron), statistics on deuteratable hydrogens:\n", t);
            print2log(wo->fpLog, "Average number of hydrogen atoms in A", "%g", wr->wt[t].nHydAv_A);
            print2log(wo->fpLog, "Average number of     deuterated in A", "%g", wr->wt[t].n2HAv_A);
            print2log(wo->fpLog, "Average number of not deuterated in A", "%g", wr->wt[t].n1HAv_A);
            print2log(wo->fpLog, "Average fraction deuterated      in A", "%g", wr->wt[t].n2HAv_A/wr->wt[t].nHydAv_A);
            print2log(wo->fpLog, "Average number of hydrogen atoms in B", "%g", wr->wt[t].nHydAv_B);
            print2log(wo->fpLog, "Average number of     deuterated in B", "%g", wr->wt[t].n2HAv_B);
            print2log(wo->fpLog, "Average number of not deuterated in B", "%g", wr->wt[t].n1HAv_B);
            print2log(wo->fpLog, "Average fraction deuterated      in B", "%g", wr->wt[t].n2HAv_B/wr->wt[t].nHydAv_B);
        }
    }

    gmx_envelope_bounding_sphere(wr->wt[0].envelope, cent, &r2);
    print2log(wo->fpLog, "\nDiameter of the bounding envelope [nm]", "%g", 2*sqrt(r2));
    print2log(wo->fpLog, "Average radius of gyration (solute only, electron-weighted) [nm]", "%g", wr->RgAv);
    if (wr->ewaxsaniso == ewaxsanisoNO)
    {
        /* Guinier fit of final intensity */
        for (t = 0; t<wr->nTypes; t++)
        {
            RgGuinier = guinierFit(&wr->wt[t], wr->wt[t].wd->I, wr->wt[t].wd->varI, wr->RgAv);
            if (wr->nTypes == 1)
            {
                print2log(wo->fpLog, "Final radius of gyration (Guiner fit) [nm]", "%g", RgGuinier);
            }
            else
            {
                sprintf(title, "Final radius of gyration (Guiner fit) [nm] (%s group %d)", wr->wt[t].saxssansStr, t);
                print2log(wo->fpLog, title, "%g", RgGuinier);
            }
        }
    }
    for (t = 0; t<wr->nTypes; t++)
    {
        sprintf(title, "Number of Shannon channels (%s group %d)", wr->wt[t].saxssansStr, t);
        print2log(wo->fpLog, title, "%g", wr->wt[t].nShannon);
    }

    fprintf(wo->fpLog,"\nNr of frames with a solvation layer thinner than %f: %d of %d\n",
            wr->solv_warn_lay, wr->nSolvWarn, wr->waxsStep);
    fprintf(wo->fpLog,"Nr of frames with a distance to the box boundary smaller than %g: %d of %d\n",
            WAXS_WARN_BOX_DIST, wr->nWarnCloseBox, wr->waxsStep);
    fprintf(wo->fpLog,"\n");

    if (wr->bGridDensity)
    {
        gmx_envelope_griddensity_write(wr->wt[0].envelope, wo->fnDensity);
    }
}

void
done_waxs_solvent(t_waxs_solvent ws)
{
    int i;
    /* Should do something about that topology. */
    done_mtop(ws->mtop, TRUE );
    sfree(ws->box);
    for(i = 0; i < ws->nframes; i++)
    {
        sfree(ws->x[i]);
    }
    sfree(ws->x);
    sfree(ws);
}

/* NB: Only MASTER reads the waxs solvent. */
static void
read_waxs_solvent(output_env_t oenv, t_waxs_solvent ws, const char *fntps, const char *fnxtc,
                  gmx_bool bVerbose, t_waxsrec *wr)
{
    rvec *xtop;
    matrix topbox;
    char title[STRLEN];
    t_trxstatus *status;
    int natoms, j, i, ngrps_solvent;
    real mb;
    t_inputrec ir;
    t_state dummystate;
    t_atoms *atoms;
    gmx_bool bHaveEnough;
    t_trxframe fr;

    ws->nframes = 0;
    ws->x       = NULL;
    ws->box     = NULL;
    /* scattTypes is written later, to allow to overwrite the solvent scattering types
       by the solvent types from the main simulation */
    ws->cmtypeList             = NULL;
    ws->nsltypeList            = NULL;
    ws->avInvVol               = 0;
    ws->avDensity              = 0;
    ws->nelec                  = 0;
    ws->avScattLenDens         = NULL;

    /* Note that it will be more useful for us to prepare this exactly like an MD with -rerun */
    /* read solvent topology */
    snew(ws->mtop,   1);
    fprintf(stderr, "\n");
    read_tpx_state(fntps, &ir, &dummystate, NULL, ws->mtop);

    ngrps_solvent = get_actual_ngroups( &(ws->mtop->groups), egcWAXSSolvent );

    fprintf(stderr, "Read water topology %s file successfully:\n\t"
            "Containtes %d Cromer-Mann types, %d NSL types, and %d solvent groups)\n",
            fntps, ws->mtop->scattTypes.ncm, ws->mtop->scattTypes.nnsl, ngrps_solvent);

    /* Check if NSLs are defined in the water tpr, ws->mtop->scattTypes.nnsl has the number of neutron scattering lengths. 0=false */
    ws->bHaveNeutron = ws->mtop->scattTypes.nnsl;

    if (ngrps_solvent  != 1)
    {
        gmx_fatal(FARGS, "Found %d solvent groups in %s. Please define one group as waxs-solvent and regenerate a\n"
                "tpr file for the pure-buffer system with grompp.", ngrps_solvent, fntps);
    }
    if (ws->mtop->scattTypes.ncm == 0)
    {
        gmx_fatal(FARGS,"No Cromer-Mann definitions found in %s\n"
                "Regenerate the tpr file with mdp option waxs-coupl = yes and waxs-solvent defined.\n",
                fntps);
    }

    if (bVerbose)
    {
        fprintf(stderr, "Found these Cromer-Mann / NSL types in %s:\n", fntps);
        fprintf(stderr, "\t    %-36s%-36s%-8s\n","a1-a4", "b1-b4", "c");
        for (i = 0; i < ws->mtop->scattTypes.ncm; i++)
        {
            fprintf(stderr, "%3d) ",i);
            print_cmsf(stderr, &ws->mtop->scattTypes.cm[i], 1);
        }
        if (ws->mtop->scattTypes.nnsl)
        {
            fprintf(stderr, "\t    %-36s\n","cohb");
            for (i = 0; i < ws->mtop->scattTypes.nnsl; i++)
            {
                fprintf(stderr, "%3d) %g", i, ws->mtop->scattTypes.nsl[i].cohb);
            }
        }
    }

    /* Get coninuous array with number of electrons per atom */
    ws->nElectrons = make_nElecList(ws->mtop);

    /* read solvent coordinates into ws->x. Type definitions are left until later. */
    read_first_frame(oenv, &status, fnxtc, &fr, TRX_NEED_X);
    natoms = fr.natoms;
    if (natoms != ws->mtop->natoms)
    {
        gmx_fatal(FARGS, "Error while reading solvent for WAXS analysis\n"
                "%d atoms in %s, but %d atoms in %s\n", ws->mtop->natoms, fntps,
                natoms, fnxtc);
    }
    if (! fr.bBox)
    {
        gmx_fatal(FARGS, "No box (unit cell) information found in %s\n", fnxtc);
    }
    ws->ePBC = fr.ePBC;

    /* Read trajectory up until waxs-nfrsolvent frames were found */
    j = 0;
    do{
        srenew(ws->x,   j+1);
        srenew(ws->box, j+1);
        snew(ws->x[j],  natoms);
        for (i = 0; i < natoms; i++)
        {
            copy_rvec(fr.x[i], ws->x[j][i]);
        }
        copy_mat(fr.box, ws->box[j]);
        j++;
        bHaveEnough = ((wr->nfrsolvent > 0) && (j >= wr->nfrsolvent));
        if (bHaveEnough)
        {
            printf("\n\nStopped reading from %s after %d frames,\n\tas requested by mdp option waxs_nfrsolvent = %d\n",
                    fnxtc, j, wr->nfrsolvent);
        }
    } while(!bHaveEnough && read_next_frame(oenv, status, &fr));
    ws->nframes = j;
    if (wr->nfrsolvent < 0)
    {
        wr->nfrsolvent = ws->nframes;
    }
    mb = 1.0*natoms*j*sizeof(rvec)/1024/1024;
    fprintf(stderr, "Read %d frames from %s (%d atoms) (%4g MiByte)\n\n", j, fnxtc, natoms, mb);

    /* Allocate memory for current x */
    snew(ws->xPrepared, natoms);
}


/* Get total number of electrons or NSLs inside the enelope */
static real
number_of_electrons_or_NSLs(t_waxsrecType *wt, int *scatTypeIndex, int isize, gmx_mtop_t *mtop)
{
    int            i;
    real           sum;

    sum = 0;
    for (i = 0; i < isize; i++)
    {
        if (wt->type == escatterXRAY)
        {
            sum += CROMERMANN_2_NELEC(mtop->scattTypes.cm[scatTypeIndex[i]]);
        }
        else
        {
            sum += wt->nsl_table[scatTypeIndex[i]];
        }
    }
    return sum;
}

/* Estimate the contrast of the solute. This is used to scale down the SAXS-derived forces.

   Rationale:

   SAXS: When we dispalce a solute groups by some SAXS-derived forces, any gaps will be
   filled by solvent, hence the change in the intensity is smaller than expected from the
   solute atoms alone. A good estimate for this reduced effect of conformational transitions
   on the SAXS/SANS curves is given by the contrast
     (rho_solute-rho_solvent) / rho_solvent.
   In other words: when computing the intensity gradients on solute atoms along, we would
   overestimate the true gradients.

   SANS: This effect is most apparent when doing SANS, here, if we have negative contrast
   in case high D2O concentration, the correcting factor is even negative, flipping the direction
   of SANS-derived forces. In other words: In this case, when the solute moves left, the
   contrast moves to the right.

   KNOWN ISSUES: If different domains have a different perdeuteration level, this
   correction does no work. In this case, we would have to work with buffer-subtracted
   form factors ("reduced form factors"). This may be the best solution for computing
   intensity gradients anyway, but we would need atomic forces.
*/
static void
updateSoluteContrast(t_waxsrec *wr, t_commrec *cr)
{
    int            t;
    double         soluteDensity, solventDensity;
    t_waxsrecType *wt;

    if (MASTER(cr) && !wr->waxs_solv)
    {
        gmx_incons("waxs_solv not available in updateSoluteContrast()\n");
    }

    for (t = 0; t < wr->nTypes; t++)
    {
        wt = &wr->wt[t];

        if (MASTER(cr))
        {
            soluteDensity       = wt->soluteSumOfScattLengths / wr->soluteVolAv;
            solventDensity      = wr->waxs_solv->avScattLenDens[t];
            wt->soluteContrast  = soluteDensity - solventDensity;
            wt->contrastFactor  = wt->soluteContrast / soluteDensity;
            /* printf("Updating contrast: soluteDens %8g  solventDens %8g  constrast %8g  factor %8g\n",
               soluteDensity, solventDensity, wt->soluteContrast, wt->contrastFactor); */
        }
        if (PAR(cr))
        {
            gmx_bcast(sizeof(double), &wt->soluteContrast, cr);
            gmx_bcast(sizeof(double), &wt->contrastFactor, cr);
        }
    }
}


/* Compute scatterig of pure solvent from RDFs. Required if the buffer subtraction was reduced by the
   partial volume of the solute */
static void
pure_solvent_scattering(t_waxsrec *wr, t_commrec *cr, const char *fnxtcSolv, gmx_mtop_t *mtop)
{
    char           *fn, path[10000], *base;
    double         *intensitySum = NULL;
    t_waxs_solvent  ws = wr->waxs_solv;
    t_waxsrecType  *wt;
    gmx_bool        bHaveSolventIntesityFile = FALSE;
    int             t;

    const int allocBlockSize = 100;
    const double nAtomsTimesFrames = 3e9;
    const char fnSolventIntensity[] = "Isolvent.dat";
    int  nq          = 501;
    int  qmax        = 50;

    /* Pure solvent works only if we have exactly one scattering type, and that is XRAY */
    if (wr->nTypes != 1)
    {
        gmx_fatal(FARGS, "Pure solvent scattering works only with one scattering type XRAY (found %d types)\n", wr->nTypes);
    }
    if (wr->wt[0].type != escatterXRAY)
    {
        gmx_fatal(FARGS, "Pure solvent scattering for correct for over-subtracted buffer works only with scattering type XRAY\n");
    }

    t  = 0;
    wt = &wr->wt[t];
    snew(wt->Ipuresolv, wt->nq);

    if (MASTER(cr))
    {
        /* Check if there is a solvent intensity file in the solvent xtc directory, or passed by
           an environment variable */
        if ( (fn = getenv("GMX_WAXS_BUFFER_INTENSITY_FILE")) != NULL)
        {
            printf("Found envrionment variable GMX_WAXS_BUFFER_INTENSITY_FILE, reading buffer intensity from %s.\n", fn);
            bHaveSolventIntesityFile = TRUE;
            strcpy(path, fn);
        }
        else
        {
            base = strrchr(fnxtcSolv, '/');
            if (! base)
            {
                strcpy(path, "./");
            }
            else
            {
                strncpy(path, fnxtcSolv, base-fnxtcSolv+1);
                path[base-fnxtcSolv+1] = '\0';
            }
            strcat(path, fnSolventIntensity);
            printf ("\nLooking for pure solvent intensity file at: %s ... ", path);
            bHaveSolventIntesityFile = gmx_fexist(path);
            if (bHaveSolventIntesityFile)
            {
                printf("found\n");
            }
            else
            {
                printf("not found.\n\t -> Will compute the RDFs, which may take a while, but it needs to be done only once.\n");
            }
        }
        if (bHaveSolventIntesityFile)
        {
            /* Read pure-solvent intensity and store in wr->Ipuresolv[] */
            read_pure_solvent_intensity_file(path, wt);
        }
    }
    if (PAR(cr))
    {
        gmx_bcast(sizeof(gmx_bool), &bHaveSolventIntesityFile, cr);
    }
    if (bHaveSolventIntesityFile)
    {
        /* If we found a pure-solvent intensity file, nothing more to be done. */
        if (PAR(cr))
        {
            gmx_bcast(wt->nq*sizeof(double), wt->Ipuresolv, cr);
        }
        return;
    }

    /* Compute RDFs between all atom types, do the sine transforms, and put the intensity into intensitySum[] */
    snew(intensitySum, nq);
    do_pure_solvent_intensity(wr, cr, mtop,
                              MASTER(cr) ? ws->x   : NULL,
                              MASTER(cr) ? ws->mtop->natoms : 0,
                              MASTER(cr) ? ws->box : NULL,
                              MASTER(cr) ? ws->nframes : 0,
                              MASTER(cr) ? ws->cmtypeList : 0,
                              MASTER(cr) ? ws->mtop->scattTypes.ncm : 0,
                              qmax, nq, intensitySum, path, t);

    /* Interpolate intensity to the q values in the mdp file, and write the result to wr->Ipuresolv */
    if (MASTER(cr))
    {
        interpolate_solvent_intensity(intensitySum, qmax, nq, wt);
    }
    /* Send wr->Ipuresolv to the nodes */
    if (PAR(cr))
    {
        gmx_bcast(wt->nq*sizeof(double), wt->Ipuresolv, cr);
    }

    /* Clean up */
    sfree(intensitySum);
    if (PAR(cr))
    {
        gmx_barrier(cr);
    }
}


static void
shift_all_atoms(int n, rvec *x, rvec diff)
{
    waxs_debug("Begin of shift_all_atoms()\n");
    int i;
    for (i=0; i<n; i++)
    {
        rvec_inc(x[i], diff);
    }
    waxs_debug("End of shift_all_atoms()\n");
}

static void
get_solvent_density(t_waxsrec *wr)
{
    int            i, j, d, iWaterframe;
    real           inv_v = 0., inv_v2 = 0., sum, err, inv_vstddev, vDroplet, varDropletDensity;
    gmx_bool       bInside = FALSE;
    double         nat_frame, nelec_frame, nelec_sum = 0., nelec2_sum = 0., tmp, var_nelec, nat_sum, nel;
    t_waxs_solvent ws;

    ws = wr->waxs_solv;

    for (i = 0; i < ws->nframes; i++)
    {
        tmp     = 1./det(ws->box[i]);
        inv_v  += tmp;
        inv_v2 += sqr(tmp);
    }
    inv_v      /= ws->nframes;
    inv_v2     /= ws->nframes;
    inv_vstddev = inv_v2 - sqr(inv_v);

    for (i = 0; i < wr->nindB_sol; i++)
    {
        nel = ws->nElectrons[wr->indB_sol[i]];
        if (nel < 0)
        {
            gmx_fatal(FARGS, "Found illegal number of Electrons (%g) of solvent box atom %d.\n",
                    nel, wr->indB_sol[i]);
        }
        nelec_sum += nel;
    }

    ws->nelec           = nelec_sum;
    ws->avInvVol        = inv_v;
    ws->avDensity       = nelec_sum*inv_v;
    printf("\nAnalyzing complete solvent system (from %d frames):\n"
           "\t# of atoms          = %d\n"
           "\t# of electrons      = %g\n"
           "\tav. volume [nm3]    = %g +- %g\n"
           "\tav. density [e/nm3] = %g\n\n",
           ws->nframes,
           wr->nindB_sol, nelec_sum, 1./inv_v, inv_vstddev/sqr(inv_v), ws->avDensity);

    if (ws->avDensity > 334*1.1 || ws->avDensity < 334*0.9)
    {
        fprintf(stderr, "\n\nWARNING, you have a solvent density of %g e/nm3 (for water, this should"
                " be ~334 e/nm3)\n\n", ws->avDensity);
    }

    /* Now get density and variance of density inside the envelope */
    nat_sum    = 0.;
    nelec_sum  = 0.;
    nelec2_sum = 0.;

    for (iWaterframe = 0; iWaterframe < ws->nframes; iWaterframe++)
    {
        if ((iWaterframe%20) == 0 || iWaterframe == (ws->nframes-1))
        {
            printf("\rComputing density (average + variance) of excluded volume - %6.1f%% done",
                    100.*(iWaterframe+1)/ws->nframes);
            fflush(stdout);
        }

        /* Shift pure-solvent frame number 'iWaterframe' onto the enevelope. On exit, ws->xPrepared
           is on the envelope */
        preparePureSolventFrame(ws, iWaterframe, wr->wt[0].envelope, wr->debugLvl);

        /* Count electrons and atoms within droplet */
        nat_frame    = 0;
        nelec_frame  = 0;

        #pragma omp parallel private(bInside)
        {
            double nelec_loc = 0;
            double nat_loc   = 0;

            #pragma omp for
            for (j = 0; j < wr->nindB_sol; j++)
            {
                bInside = gmx_envelope_isInside(wr->wt[0].envelope, ws->xPrepared[wr->indB_sol[j]]);
                if (bInside)
                {
                    nelec_loc += ws->nElectrons[wr->indB_sol[j]];
                    nat_loc   += 1.;
                }
            }

            #pragma omp critical
            {
                /* Sum over threads */
                nat_frame   += nat_loc;
                nelec_frame += nelec_loc;
            }
        }

        /* Store for computing variance of nelec */
        nelec_sum  += nelec_frame;
        nelec2_sum += dsqr(nelec_frame);
        nat_sum    += nat_frame;
    }
    printf("\n\n");

    nelec_sum            /= ws->nframes;
    nelec2_sum           /= ws->nframes;
    nat_sum              /= ws->nframes;
    var_nelec             = nelec2_sum - dsqr(nelec_sum);
    vDroplet              = droplet_volume(wr);
    ws->avDropletDensity  = nelec_sum/vDroplet;
    varDropletDensity     = var_nelec/sqr(vDroplet);
    if (ws->avDropletDensity > 334*1.1 || ws->avDropletDensity < 334*0.9)
    {
        fprintf(stderr, "\n\nWARNING, you have a solvent density of %g e/nm3 (for water, this should"
                " be ~334 e/nm3)\n\n", ws->avDropletDensity);
    }

    if ( fabs(ws->avDropletDensity/ws->avDensity-1) > 0.01 )
    {
        gmx_fatal(FARGS, "The solvent density estimated from the whole box (%g e/nm3) differs by more than 1%%"
                  "from the density estimated from the solvent inside the envelope (%g e/nm3). This might indicate that"
                  "don't have enough frames, or your pure-solvent systems does not contain only solvent.",
                  ws->avDensity, ws->avDropletDensity);
    }

    printf("Solvent droplet inside the envelope (computed from %d frames):\n"
           "\t# of atoms           = %g\n"
           "\t# of electrons       = %g  +-  %g\n"
           "\tvolume [nm3]         = %g\n"
           "\tdensity [e/nm3]      = %g  +- %g \n",
           ws->nframes, nat_sum, nelec_sum, sqrt(var_nelec), vDroplet,
           ws->avDropletDensity, sqrt(varDropletDensity));

    if (wr->bFixSolventDensity)
    {
        printf("Adding a constant electron density of %g e/nm3 to the solvent (to reach requested density of %g e/nm3).\n\n",
                wr->givenElecSolventDensity - ws->avDropletDensity, wr->givenElecSolventDensity);
    }
    else
    {
        printf("Will not change the solvent density but simply use density in xtc file (because waxs-solvdens = 0)\n\n");
    }
}

/* Return average electron density inside the envelope (for A or B system) */
static real
droplet_density(t_waxsrec *wr, atom_id* index, int isize,  double *nElectrons, real *nelecRet)
{
    int           i;
    double        nelec = 0., nel;
    real          dens;
    t_cromer_mann cm;

    for (i = 0; i < isize; i++)
    {
        nel = nElectrons[index[i]];
        if (nel < 0)
        {
            gmx_fatal(FARGS, "While computing the density of the droplet in droplet_density(), stepped over"
                    " atom %d does not have Cromer-Mann parameters (nElectrons = %g)\n", index[i]+1, nel);
        }
        nelec += nel;
    }
    /* There is no volume with bVacuum */
    dens = wr->bVacuum ? -1. : nelec/droplet_volume(wr);
    *nelecRet = nelec;
    return dens;
}

void
read_intensity_and_interpolate(const char *fn, real minq, real maxq, int nq, gmx_bool bRequireErrors,
        gmx_bool bReturnVariance, double **I_ret, double **Ierr_ret)
{
    real left=0.,right=0., Iright, Ileft, Isigright, Isigleft, q, slopeI, slopeIsig, dq;
    int ncol, nlines, i, j;
    gmx_bool bFound;
    double **y, *I, *Ierr;

    nlines = read_xvg(fn, &y, &ncol);
    /* Now: q=y[0], I=y[1], sigma=y[2] */
    if (ncol != 3 && ncol != 2)
    {
        gmx_fatal(FARGS,"Expected 2 or 3 rows in file %s (q, Intensity, and (optionally) sigma(Intensity). Found %d rows.\n",
                fn, ncol);
    }
    if (bRequireErrors && ncol < 3)
    {
        gmx_fatal(FARGS, "Require uncertainties in intensity file %s, but found only %d columns\n", fn, ncol);
    }
    if (ncol == 2)
    {
        printf("WAXS-MD: No experimental errors found in %s\n", fn);
    }
    if (y[0][0] > minq)
    {
        gmx_fatal(FARGS,"Smallest q-value in %s is %g, but the smallest q requested is %g. Provide a different\n "
                "scattering intensity file or increase waxs-startq in the mdp file\n", fn, y[0][0], minq);
    }
    if (y[0][nlines-1] < maxq)
    {
        gmx_fatal(FARGS,"Largest q-value in %s is %g, but the largest q requested is %g. Provide a different\n "
                "scattering intensity file or decrease waxs-endq in the mdp file\n", fn, y[0][nlines-1], maxq);
    }

    snew(I, nq);
    if (ncol >= 3)
    {
        snew(Ierr, nq);
    }
    else
        Ierr = NULL;

    dq = nq > 1 ? (maxq - minq)/(nq - 1) : 0.0 ;
    for (i = 0; i < nq; i++)
    {
        q = minq + i*dq;
        bFound = FALSE;
        /* Simple linear interpolation. Maybe we can improve this later. */
        for (j = 0; j < nlines-1; j++)
        {
            if ((left = y[0][j]) <= q && q <= (right = y[0][j+1]))
            {
                bFound = TRUE;
                break;
            }
        }
        if (!bFound)
        {
            gmx_fatal(FARGS,"Error in read_intensity_and_interpolate(). This should not happen.\n");
        }
        Ileft    = y[1][j];
        Iright   = y[1][j+1];
        slopeI   = (Iright-Ileft) / (right-left);
        I[i]     = Ileft + (q-left) * slopeI;

        if (ncol >= 3)
        {
            Isigleft  = y[2][j];
            Isigright = y[2][j+1];
            slopeIsig = (Isigright-Isigleft) / (right-left);
            Ierr[i]   = Isigleft + (q-left) * slopeIsig;
            if (bReturnVariance)
            {
                Ierr[i] = dsqr(Ierr[i]);
            }
            if (Ierr[i] < 1e-20)
            {
                gmx_fatal(FARGS,"sigma of exprimental I(q) is zero at q = %g. Provide a non-zero sigma.\n",q);
            }
        }
    }
    *I_ret    = I;
    *Ierr_ret = Ierr;
}

static void
read_waxs_curves(const char *fnScatt, t_waxsrec *wr, t_commrec *cr)
{
    int i, t, len;
    char *fnUsed = NULL, *base = NULL, *fnInterp = NULL;
    t_waxsrecType *wt = NULL;

    if (MASTER(cr))
    {
        len = strlen(fnScatt);
        snew(base, len);
        strncpy(base, fnScatt, len-4);
        base[len-4] = '\0';
        fprintf(stderr, "Using base file name for WAXS curve reading: %s\n", base);
        snew(fnUsed,   len+10);
        snew(fnInterp, len+10);
    }

    for (t = 0; t<wr->nTypes; t++)
    {
        wt = &wr->wt[t];

        if (wt->nq <= 0 || wt->nq > 999999)
        {
            gmx_incons("Invalid waxs.nq in read_waxs_curves()\n");
        }

        if (MASTER(cr))
        {
            /* With one type:       Reading file fnScatt, such as saxs-curve.xvg
             * With multiple types: Reading files such as saxs-curve_1.xvg, saxs-curve_2.xvg,
             */
            if (wr->nTypes > 1)
            {
                sprintf(fnUsed,   "%s_%d.xvg", base, t+1);
                sprintf(fnInterp, "Iinterp_%d.dat",  t+1);
            }
            else
            {
                sprintf(fnUsed, "%s", fnScatt);
                sprintf(fnInterp, "Iinterp.dat");
            }

            printf("WAXS-MD: Reading intensities from %s\n", fnUsed);
            printf("WAXS-MD: Writing interpolated intensities to %s\n", fnInterp);
            read_intensity_and_interpolate(fnUsed, wt->minq, wt->maxq, wt->nq, FALSE, FALSE, &wt->Iexp, &wt->Iexp_sigma);

            if (wt->Iexp_sigma == NULL && (
                    wr->weightsType == ewaxsWeightsEXPERIMENT                  ||
                    wr->weightsType == ewaxsWeightsEXP_plus_CALC               ||
                    wr->weightsType == ewaxsWeightsEXP_plus_SOLVDENS           ||
                    wr->weightsType == ewaxsWeightsEXP_plus_CALC_plus_SOLVDENS ))
            {
                gmx_fatal(FARGS, "Requested to use exerimental SAXS errors, but no errors were read from "
                          "experimental intensity file %s\n", fnScatt);
            }

            FILE *fp = fopen(fnInterp, "w");
            fprintf(fp, "@type xydy\n");
            for (i = 0; i < wt->nq; i++)
            {
                fprintf(fp, "%g %g %g\n", wt->minq + i*(wt->maxq-wt->minq)/(wt->nq-1), wt->Iexp[i], wt->Iexp_sigma[i]);
            }
            fclose(fp);
        }

        /* bcast it */
        if (wt->Iexp_sigma == NULL)
        {
            snew(wt->Iexp,       wt->nq);
            snew(wt->Iexp_sigma, wt->nq);
        }
        if (PAR(cr))
        {
            gmx_bcast(wt->nq*sizeof(double), wt->Iexp,       cr);
            gmx_bcast(wt->nq*sizeof(double), wt->Iexp_sigma, cr);
        }

        /* Scale I(q=0) to input I(q)? Then store Iexp(q=0) to targetI0 */
        if (wr->bScaleI0)
        {
            if (wt->minq != 0.0)
            {
                gmx_fatal(FARGS, "When scaling I(q=0) to target, you must compute I(q=0). Use waxs-startq = 0 (found %g)\n",
                        wt->minq);
            }
            /*NEW, differend mode to get I0. Added by MH*/
            /* wt->targetI0 = wt->Iexp[0]; */
            char *buf;
            if ((buf = getenv("GMX_WAXS_I0")) != NULL)
            {
                wt->targetI0 = atof(buf);
                if (MULTISIM(cr) && (cr)->nodeid == 0 && MASTER(cr) && cr->ms->sim == 0)
                {
                    fprintf(stderr, "\nWAXS-MD read saxs curve: Found environment variable GMX_WAXS_I0 = %.1f\n", wt->targetI0);
                }
                else
                {
                    fprintf(stderr, "\nWAXS-MD read saxs curve:   Found environment variable GMX_WAXS_I0 = %.1f\n", wt->targetI0);
                }
            }
            else
            {
                wt->targetI0 = wt->Iexp[0];
                if (MASTER(cr))
                {
                    fprintf(stderr, "\nWAXS-MD read saxs curve: Did not  found GMX_WAXS_I0, taking the experimental value: %.1f\n", wt->targetI0);
                }
            }
            if (MASTER(cr))
            {
                printf("WAXS-MD read saxs curve: Will scale I(q=0) to predefined target from input I(q) curve: %g\n", wt->targetI0);
            }
            if (PAR(cr))
            {
                gmx_bcast(sizeof(real), &wt->targetI0, cr);
            }
        }
    }

    if (MASTER(cr))
    {
        sfree(base);
        sfree(fnUsed);
        sfree(fnInterp);
    }
}

static
void init_ensemble_stuff(t_waxsrec *wr,  t_commrec *cr)
{
    int m, t;
    char fn[1024], typeStr[STRLEN];
    t_waxsrecType *wt;

    if (wr->ensemble_nstates < 2 || wr->ensemble_nstates > 50)
    {
        gmx_fatal(FARGS, "Error in init_ensemble_stuff(), wr->ensemble_nstates = %d\n", wr->ensemble_nstates);
    }

    for (t = 0; t < wr->nTypes; t++)
    {
        wt = &wr->wt[t];

        /* Read and bcast intensities of fixed states from files intensity_stateXX.dat */
        snew(wt->ensemble_I,    wr->ensemble_nstates - 1);
        snew(wt->ensemble_Ivar, wr->ensemble_nstates - 1);


        if (wr->nTypes > 1)
        {
            sprintf(typeStr, "_scatt%d", t);
        }
        else
        {
            strcpy(typeStr, "");
        }

        /* Refining only one state with the MD, while reading the SAXS curves of all other states from a file. */
        if (wr->ewaxs_ensemble_type == ewaxsEnsemble_BayesianOneRefined)
        {
            for (m = 0; m < wr->ensemble_nstates - 1; m++)
            {
                if (MASTER(cr))
                {
                    sprintf(fn, "intensity%s_state%02d.dat", typeStr, m);
                    if (! gmx_fexist(fn) )
                    {
                        gmx_fatal(FARGS,
                                "For ensemble refinement with %d states, you need to place %d file(s) with name(s) intensity_stateXX.dat\n"
                                "into the current directory, where XX are two digits (00, 01, 02, ...)\n"
                                "If you have multiple scattering groups (xray, neutron, etc.), then the file names are\n"
                                "intensity_scattY_stateXX.dat, where Y = 0, 1, 2, etc. for each scatterig group defined in the mdp file (option scatt-coupl)\n\n"
                                "However, %s was not found.\n", wr->ensemble_nstates, wr->ensemble_nstates-1, fn);
                    }
                    read_intensity_and_interpolate(fn, wt->minq, wt->maxq, wt->nq, TRUE, TRUE,
                            &wt->ensemble_I[m], &wt->ensemble_Ivar[m]);
                    printf("WAXS-MD: Read intensity of fixed state %d from file %s (scattering group %d)\n", m, fn, t);
                }
                /* bcast it */
                if (! MASTER(cr))
                {
                    snew(wt->ensemble_I   [m], wt->nq);
                    snew(wt->ensemble_Ivar[m], wt->nq);
                }
                if (PAR(cr))
                {
                    gmx_bcast(wt->nq*sizeof(double), wt->ensemble_I   [m], cr);
                    gmx_bcast(wt->nq*sizeof(double), wt->ensemble_Ivar[m], cr);
                }
            }
        }
        else
        {
            gmx_fatal(FARGS, "Unsupported ensemble type in  init_ensemble_stuff()\n");
        }

        /* init intensity sum arrays for ensemble refinement */
        snew(wt->ensembleSum_I,    wt->nq);
        snew(wt->ensembleSum_Ivar, wt->nq);
    }
}

void
waxs_quick_debug(t_waxsrec *wr)
{
    t_waxs_datablock wd = wr->wt[0].wd;
    int nprot = wr->nindA_prot;
    int nloc  = wr->wt[0].qvecs->qhomenr;
    int n     = wr->wt[0].qvecs->n;
    fprintf(stderr,"Quick debug report:\n");
    fprintf(stderr,"First real in avA: %g\n",wd->avA[0].re);
    fprintf(stderr,"First real in avB: %g\n",wd->avB[0].re);
    fprintf(stderr,"Last real in avA: %g\n",wd->avA[nloc-1].re);
    fprintf(stderr,"Last real in avB: %g\n",wd->avB[nloc-1].re);
    fprintf(stderr,"NormA: %g\n",wd->normA);
    fprintf(stderr,"NormB: %g\n",wd->normB);
    //fprintf(stderr,"First D(local): %g\n",wd->D[0]);
    //fprintf(stderr,"First D(global): %g\n",wd->Dglobal[0]);
    //fprintf(stderr,"Last D(local): %g\n",wd->D[nloc-1]);
    //fprintf(stderr,"Last D(global): %g\n",wd->Dglobal[n-1]);
}



#define GMX_WAXS_SOLVENT_ROUGHNESS 0.4
void
check_selected_solvent(rvec *xA, rvec *xB,
        atom_id *indexA, int isizeA, atom_id *indexB, int isizeB,
        gmx_bool bFatal)
{
    int i,d;
    rvec *x;
    rvec maxA={-1e20, -1e20, -1e20};
    rvec minA={ 1e20,  1e20,  1e20};
    rvec maxB={-1e20, -1e20, -1e20};
    rvec minB={ 1e20,  1e20,  1e20};
    char buf[1024];
    const char dimletter[3][2]={"x","y","z"};

    /* find smallest and largest coordinate of solvation layer and excluded solvent */
    for (i=0; i<isizeA; i++)
    {
        x=&xA[indexA[i]];
        for (d=0; d<DIM; d++)
        {
            if ((*x)[d] < minA[d])
                minA[d]=(*x)[d];
            if ((*x)[d] > maxA[d])
                maxA[d]=(*x)[d];
        }
    }
    for (i=0; i<isizeB; i++)
    {
        x=&xB[indexB[i]];
        for (d=0; d<DIM; d++)
        {
            if ((*x)[d] < minB[d])
                minB[d]=(*x)[d];
            if ((*x)[d] > maxB[d])
                maxB[d]=(*x)[d];
        }
    }

    for (d=0; d<DIM; d++)
    {
        if (fabs(maxA[d]-maxB[d]) > GMX_WAXS_SOLVENT_ROUGHNESS ||
                fabs(minA[d]-minB[d]) > GMX_WAXS_SOLVENT_ROUGHNESS)
        {
            sprintf(buf,"Solvation layer and excluded solvent are different in direction %s\n"
                    "Found minA / minB = %g / %g -- maxA / maxB = %g / %g\n",dimletter[d],
                    minA[d],minB[d],maxA[d],maxB[d]);
            if (bFatal)
                gmx_fatal(FARGS,buf);
            else
                fprintf(stderr,"\n\nWARNING - WARNING - WARNING \n\n%s\n\nWARNING - WARNING - WARNING\n\n",
                        buf);
        }
    }
}

void
write_x_f_this_frame(FILE *fp, rvec *x, atom_id *index_prot, int isize_prot, rvec * f )
{
    int i, ii, j;

    for (i = 0; i < isize_prot; i++)
    {
        ii = index_prot[i];
        for (j = 0; j < DIM; j++)
        {
            fprintf(fp, "%12g ", x[ii][j]);
        }
        for (j = 0; j < DIM; j++)
        {
            fprintf(fp, "%12g ", f[i][j]);
        }
        fprintf(fp, "\n");
    }
}

void
error_write_prot_with_this_water(const char* fn, t_topology *top, rvec *x, matrix box, atom_id* index_prot,
        int isize_prot, int jx, int ePBC)
{
    int *index;

    snew(index, isize_prot+1);
    memcpy(index, index_prot, isize_prot*sizeof(int));
    index[isize_prot] = jx;

    fprintf(stderr,"Write pdb file %s\n\n",fn);
    write_sto_conf_indexed(fn, "Water PBC error", &top->atoms, x, NULL,
            ePBC, box, isize_prot+1, index);
    sfree(index);
}

/* Largest distance of an atom from the origin */
static real
maxDistFromOrigin(rvec x[], atom_id *index, int nind)
{
    int i;
    real max2=-1., n2;
    //fprintf(stderr,"Debug: nfit=%d, fit=%p\n",nind_fit,ind_fit);
    for (i=0; i<nind; i++)
    {
        n2 = norm2(x[index[i]]);
        if (n2 > max2)
        {
            max2 = n2;
        }
    }
    return sqrt(max2);
}


/* Get solvent atoms inside the envelope of solute/solvent and pure-solvent system */
static void
get_solvation_shell(rvec x[], rvec xex[], t_waxsrec *wr,
                    gmx_mtop_t *mtop, gmx_mtop_t *mtopex, int ePBC, matrix box)
{
    rvec diff;
    real mindist, envMaxR2 = -1.;
    int j, iMindist, k, nOutside = 0;
    const int allocBlocksize=100, nWarnOutsideMax = 10;
    char fn[256], title[1024];
    t_atoms *atoms;
    gmx_bool bInside = FALSE, bMinDistToOuter;

    waxs_debug("Begin of get_solvation_shell()\n");

    /* Check if all atoms solute atoms are within envelope */
    for (j = 0; j<wr->nindA_prot; j++)
    {
        if (!gmx_envelope_isInside(wr->wt[0].envelope, x[wr->indA_prot[j]]))
        {
            nOutside ++;
            if (nOutside <= nWarnOutsideMax)
            {
                fprintf(stderr, "WARNING, atom %d of solute x = (%6.2f %6.2f %6.2f) is outside of the envelope\n",
                        wr->indA_prot[j]+1, x[wr->indA_prot[j]][XX], x[wr->indA_prot[j]][YY],x[wr->indA_prot[j]][ZZ]);
            }
            if (nOutside == nWarnOutsideMax)
            {
                fprintf(stderr, "       NOTE: Will not report more warnings of atoms outside of envelope\n");
            }
        }
    }
    if (nOutside)
    {
        fprintf(stderr, "WARNING, %d atoms outside of envelope\n", nOutside);
    }

    /* Check if solvation layer too thin */
    gmx_envelope_minimumDistance(wr->wt[0].envelope, (const rvec*) x, wr->indA_prot, wr->nindA_prot,
            &mindist, &iMindist, &bMinDistToOuter);
    if (mindist < wr->solv_warn_lay)
    {
        wr->nSolvWarn++;
        fprintf(stderr,"\n\n*** WARNING ***\n"
                "The current solvation layer is only %.3f nm thick.\n"
                "\tWarning at %g\n"
                "\tClosest atom #: %d, distance to %s envelope surface = %g\n\n",
                mindist, wr->solv_warn_lay, wr->indA_prot[iMindist],
                bMinDistToOuter ? "outer" : "inner", mindist);
    }
    envMaxR2 = sqr(gmx_envelope_maxR(wr->wt[0].envelope));

    /* Construct solvation layer around solute */
    wr->isizeA = wr->nindA_prot;
    for (j = 0; j<wr->nindA_sol; j++)
    {
        k       = wr->indA_sol[j];
        bInside = gmx_envelope_isInside(wr->wt[0].envelope, x[k]);
        if (norm2(x[k]) > envMaxR2 && bInside)
        {
            gmx_fatal(FARGS, "Envelope says inside, but diff = %g (maxR = %g)\n",
                    sqrt(norm2(x[k])), sqrt(envMaxR2));
        }

        if (bInside)
        {
            wr->isizeA++;
            if (wr->isizeA >= wr->indexA_nalloc)
            {
                wr->indexA_nalloc += allocBlocksize;
                srenew(wr->indexA, wr->indexA_nalloc);
            }
            wr->indexA[wr->isizeA-1] = k;
        }
    }

    /* Get atoms of pure-solvent system inside the envelope. */
    if (wr->bDoingSolvent)
    {
        wr->isizeB = 0;
        for (j = 0; j<wr->nindB_sol; j++)
        {
            k       = wr->indB_sol[j];
            bInside = gmx_envelope_isInside(wr->wt[0].envelope, xex[k]);
            if (norm2(xex[k]) > envMaxR2 && bInside)
            {
                gmx_fatal(FARGS, "Envelope says inside, but diff = %g (maxR = %g)\n",
                        sqrt(norm2(xex[k])), sqrt(envMaxR2));
            }

            if (bInside)
            {
                wr->isizeB++;
                if (wr->isizeB >= wr->indexB_nalloc)
                {
                    wr->indexB_nalloc += allocBlocksize;
                    srenew(wr->indexB, wr->indexB_nalloc);
                }
                wr->indexB[wr->isizeB-1] = k;
            }
        }
    }

    if (wr->debugLvl > 1 || wr->waxsStep == 0)
    {
        sprintf(fn, "prot+solvlayer_%d.pdb", wr->waxsStep);
        snew(atoms, 1);
        *atoms = gmx_mtop_global_atoms(mtop);
        write_sto_conf_indexed(fn, "Protein/Solute within the envelope", atoms, x, NULL, ePBC ,box, wr->isizeA, wr->indexA);
        sfree(atoms); /* Memory leak */
        printf("Wrote %s\n", fn);

        if (wr->bDoingSolvent)
        {
            sprintf(fn, "excludedvolume_%d.pdb", wr->waxsStep);
            snew(atoms, 1);
            *atoms = gmx_mtop_global_atoms(mtopex);
            write_sto_conf_indexed(fn, "Excluded solvent within the envelope", atoms, xex, NULL, ePBC, box, wr->isizeB, wr->indexB);
            sfree(atoms); /* Memory leak */
            printf("Wrote %s\n", fn);
        }
    }

    waxs_debug("End of get_solvation_shell()\n");
}


void
calculate_node_qload(int nodeid, int nnodes, int nq, int *qhomenr, int *qstart, int *ind, int nabs)
{
    /* Make sure that q-vectors of the same |q| are on the same node, needed when using multiple GPUs */
    int qAbsPerNode = nabs / nnodes;
    int rem         = nabs - qAbsPerNode * nnodes;
    int iabsHomenr, iabsStart;

    if (nodeid >= rem)
    {
        iabsStart  = nodeid * qAbsPerNode + rem ;
        iabsHomenr = qAbsPerNode;
    }
    else
    {
        iabsStart  = nodeid * (qAbsPerNode+1);
        iabsHomenr = qAbsPerNode+1;
    }
    *qstart  = ind[iabsStart];
    *qhomenr = ind[iabsStart+iabsHomenr] - ind[iabsStart];
    fflush(stdout);

    if (*qhomenr > 1)
    {
        fprintf(stderr, "Node %2d) |q| indices range: %2d - %2d    q-vec indices range: %4d - %4d (of %d total)\n",
                nodeid, iabsStart, iabsStart+iabsHomenr-1, *qstart, *qstart + *qhomenr-1, nq);
    }
    else
    {
        fprintf(stderr, "Node %2d) No q-vectors on this node.\n", nodeid);
    }

    if ( (*qstart + *qhomenr > nq) || (*qhomenr < 0) || (*qstart < 0))
    {
        gmx_fatal(FARGS, "Inconsistent q vector distribution on node %d: qstart = %d  qhomenr = %d  nq = %d\n",
                  nodeid, *qstart, *qhomenr, nq);
    }
}



void gen_qvecs_accounting(t_waxsrec *wr, t_commrec *cr, int nnodes , int t)
{
    t_waxs_datablock wd;
    int i, nq, qstart, qhomenr, nprot;

    wd    = wr->wt[t].wd;
    nq    = wr->wt[t].qvecs->n;
    nprot = wr->nindA_prot;

    /* For the master, generate a comprehensive list */
    if (MASTER(cr))
    {
        snew(wd->masterlist_qhomenr, nnodes);
        snew(wd->masterlist_qstart,  nnodes);
        for (i = 0; i < nnodes; i++)
        {
            calculate_node_qload(i, nnodes, nq, &qhomenr, &qstart, wr->wt[t].qvecs->ind, wr->wt[t].qvecs->nabs);
            wd->masterlist_qhomenr[i] = qhomenr;
            wd->masterlist_qstart [i] = qstart;
        }
        fprintf(stderr, "Size and offsets of q-vecs constructed on master node"
                "for communications.\n");
    }
    else
    {
        wd->masterlist_qhomenr = NULL;
        wd->masterlist_qstart  = NULL;
    }
}

void
free_qvecs_accounting(t_waxs_datablock wd)
{
    if (wd)
    {
        sfree(wd->masterlist_qhomenr); wd->masterlist_qhomenr = NULL;
    }
    if (wd)
    {
        sfree(wd->masterlist_qstart);  wd->masterlist_qstart  = NULL;
    }
}

int
qvecs_get_qglob(t_spherical_map *qvec)
{
    return qvec->n;
}

int
qvecs_get_qloc(t_spherical_map *qvec)
{
    return qvec->qhomenr;
}

/* Generate the list of q-vectors where we compute A(q), B(q), and D(q) */
t_spherical_map *gen_qvecs_map(real minq, real maxq, int nqabs, int J,
                               gmx_bool bDebug, t_commrec *cr, int ewaxsaniso, real qbeam,
                               gmx_envelope_t env, int Jmin, real Jalpha, gmx_bool bVerbose)
{
    int iabs, j, thisJ, jj, qPerNode, nWaxsNodes, nodeid, rem;
    real qabs,dq, qx=0., qy=0., qz=0., qt=0., D = 0, qGuinier;
    double phi, theta, tmp, fact;
    FILE *fp;
    t_spherical_map *qvecs;
    rvec cent;
    const int nQGuinierRequired = 8;

    nWaxsNodes = cr->nnodes - cr->npmenodes;
    nodeid     = cr->nodeid;
    if (nodeid >= nWaxsNodes)
    {
        gmx_fatal(FARGS, "Inconsistency in WAXS code: nodeid = %d, nWaxsNodes = %d\n", nodeid, nWaxsNodes);
    }

    snew(qvecs, 1);
    dq = (nqabs > 1) ? (maxq-minq)/(nqabs-1) : 0.0 ;
    snew(qvecs->abs, nqabs);
    snew(qvecs->ind,  nqabs+1);
    qvecs->ind[0]  = 0;
    qvecs->q       = NULL;
    qvecs->iTable  = NULL;
    qvecs->n       = 0;
    qvecs->nabs    = nqabs;

    if (J <= 0)
    {
        /* Automatic detemination of J = alpha * (D*q)^2, but not smaller than wr->Jmin */
        if (! gmx_envelope_bHaveSurf(env))
        {
            gmx_fatal(FARGS, "mdp option waxs-nsphere is zero, meaning that the number of q-vectors per |q|\n"
                    "is determined automatically. For that purpose, however, the envelope must be generated\n"
                    "before starting mdrun. Use g_genenv and provide the envelope file with the environment\n"
                    "variable GMX_ENVELOPE_FILE.\n");
        }
        /* Get maximum diameter of envelope */
        D = gmx_envelope_diameter(env);
        if (bVerbose)
        {
            printf("\nAutomatic selection of number of q-vectors per |q|: Using Jmin = %d, alpha = %g, D = %g nm:\n",
                    Jmin, Jalpha, D);
        }
    }

    for (iabs=0; iabs<nqabs; iabs++)
    {
        qabs = minq + iabs*dq;

        if (qabs < 1e-5)
        {
            /* At q==0. we always use only J = 1 */
            thisJ = 1;
        }
        else
        {
            if (J > 0)
            {
                /* constant J over the entire q-range, given by waxs-nsphere */
                thisJ = J;
            }
            else
            {
                /* automatic selection of J = MAX(Jmin , alpha*(D*J)^2 */
                thisJ = round(Jalpha*sqr(qabs*D));
                if (thisJ < Jmin)
                {
                    thisJ = Jmin;
                }
            }
        }
        if (bVerbose && J <= 0)
        {
            printf("\tq = %8g   J = %d\n", qabs, thisJ);
        }

        /* Store absolute q, extend ind and q */
        qvecs->abs[iabs]   = qabs;
        qvecs->ind[iabs+1] = qvecs->ind[iabs] + thisJ;
        qvecs->n          += thisJ;
        srenew(qvecs->q,      qvecs->n);
        srenew(qvecs->iTable, qvecs->n);

        fact=sqrt(M_PI*thisJ);
        for (j = 0; j < thisJ; j++)
        {
            if (ewaxsaniso == ewaxsanisoNO || ewaxsaniso == ewaxsanisoCOS2)
            {
                /* spiral method */
                tmp   = (2.0*(j+1)-1.-thisJ)/thisJ;
                theta = acos(tmp);
                phi   = fact*asin(tmp);
                qx    = qabs*cos(phi)*sin(theta);
                qy    = qabs*sin(phi)*sin(theta);
                qz    = qabs*cos(theta);
            }
            else if (ewaxsaniso == ewaxsanisoYES)
            {
                /* circular grid in the y-z plane
                   We assume the x-ray beam comes along the x-axis, so there is not point
                   to compute the scattering amplitude along x. */
                phi = 2.0*M_PI*j/thisJ;
                if (qabs*2.0 > sqr(qbeam))
                {
                    gmx_fatal(FARGS,"qmax larger then two times qbeam. Geometrically not possible. Reduce qmax.\n");
                }
                qx  = qabs*qabs*0.5/qbeam;
                qt  = sqrt(qabs*qabs-qx*qx);
                qy  = qt*cos(phi);
                qz  = qt*sin(phi);
            }
            else
            {
                gmx_fatal(FARGS, "This anisotropy (nr %d) is not yet supported\n", ewaxsaniso);
            }

            /* fill qvecs->q vector by thisJ vectors */
            jj               = qvecs->ind[iabs]+j;
            qvecs->q[jj][XX] = qx;
            qvecs->q[jj][YY] = qy;
            qvecs->q[jj][ZZ] = qz;
            // printf("qvec %5d : %8f %8f %8f\n",jj, qx, qy, qz);

            /* Store the index of the absolute value. Used for table of
               atomic scattering factors as a function of qabs: aff_table[sftype][iabs] */
            qvecs->iTable[jj] = iabs;
        }
    }


    if (cr == NULL)
    {
        qvecs->qstart  = 0;
        qvecs->qhomenr = qvecs->n;
    }
    else
    {
        /* Store for which q the scattering will be computed on this node */
        calculate_node_qload(nodeid, nWaxsNodes, qvecs->n, &(qvecs->qhomenr), &(qvecs->qstart) , qvecs->ind, qvecs->nabs);

        if (PAR(cr)) gmx_barrier(cr);
        for (j = 0; j < nWaxsNodes; j++)
        {
            if (nodeid == j)
            {
                fprintf(stderr,"Node %d uses %d q's between %d and %d\n", nodeid,
                        qvecs->qhomenr, qvecs->qstart, qvecs->qstart+qvecs->qhomenr-1);
            }
            if (PAR(cr))
            {
                gmx_barrier(cr);
            }
        }
        if (PAR(cr)) gmx_barrier(cr);
    }
    if (cr == NULL || MASTER(cr)) fprintf(stderr,"\n");

    /* Output the spheremap vectors to assigned debug file. */
    if (bDebug && (cr == NULL || MASTER(cr))) {
        fp = fopen("debug_spiral.xyz", "w");
        fprintf(fp, "# The spiral method of the spheremap\n" );
        fprintf(fp, "# n = %i\n", qvecs->n );
        for(j=0;j<qvecs->n;j++)
        {
            fprintf(fp,"%g %g %g \n", qvecs->q[j][XX], qvecs->q[j][YY], qvecs->q[j][ZZ]);
        }
        fclose(fp);
        fp = fopen("debug_spiral.pdb","w");
        if (ewaxsaniso == ewaxsanisoNO || ewaxsaniso == ewaxsanisoCOS2)
        {
            for (j = qvecs->ind[nqabs-1]; j < qvecs->ind[nqabs]; j++)
            {
                /* Make sure the coordinates are not getting too large for a PDB file */
                fact = 50./maxq;
                fprintf(fp, "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n",
                        j%99999+1, "XX", "XXX", "", 1, fact*qvecs->q[j][XX],fact*qvecs->q[j][YY],fact*qvecs->q[j][ZZ], 1.0, 0.0);
            }
        }
        else if (ewaxsaniso == ewaxsanisoYES)
        {
            for (j = 0; j < qvecs->ind[nqabs]; j++)
            {
                /* Make sure the coordinates are not getting too large for a PDB file */
                fact = 50./maxq;
                fprintf(fp, "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n",
                        j%99999+1, "XX", "XXX", "", 1, fact*qvecs->q[j][XX],fact*qvecs->q[j][YY],fact*qvecs->q[j][ZZ], 1.0, 0.0);
            }
        }
        fclose(fp);
        fprintf(stderr, "Wrote debug_spiral.xyz and debug_spiral.pdb for debugging spiral method.\n");
    }
    return qvecs;
}

/* Updating the electron density inside the envelope. This we do only with scattering type 0 */
static void
update_envelope_griddensity(t_waxsrec *wr, rvec x[])
{
    int i, i0 = 0, ifinal = 0;
    double nelec;

    gmx_envelope_griddensity_nextFrame(wr->wt[0].envelope, wr->scale);
    switch(wr->gridDensityMode)
    {
    case 0:
        /* Average density of solute and hydration layer */
        i0     = 0;
        ifinal = wr->isizeA;
        break;
    case 1:
        /* Average density of solute only */
        i0     = 0;
        ifinal = wr->nindA_prot;
        break;
    case 2:
        /* Average density of hydration layer only only */
        i0     = wr->nindA_prot;
        ifinal = wr->isizeA;
        break;
    default:
        gmx_fatal(FARGS, "Invalid mode for averaging grid density. Found %d, allowed: 0, 1, or 2\n", wr->gridDensityMode);
    }

    for (i = i0; i < ifinal; i++)
    {
        nelec = wr->nElectrons[wr->indexA[i]];
        gmx_envelope_griddensity_addAtom(wr->wt[0].envelope, x[wr->indexA[i]], nelec);
    }

    gmx_envelope_griddensity_closeFrame(wr->wt[0].envelope);
}

static void
update_envelope_solvent_density(t_waxsrec *wr, t_commrec *cr, rvec x[], int t)
{
    gmx_bool       bRecalcSolventFT;
    int            qstart, qhomenr, i, wstep = wr->waxsStep+1;
    rvec          *q;
    double         nelec;
    t_waxsrecType *wt;

    real *ft_re, *ft_im;     // For pushing parameters of FT to GPU

    wt = &wr->wt[t];

    qstart  = wt->qvecs->qstart;
    qhomenr = wt->qvecs->qhomenr;
    q       = wt->qvecs->q;

    /* Update solvent density around the protein inside envelope (cumulative average) */
    gmx_envelope_solvent_density_nextFrame(wt->envelope, wr->scale);

    /* Master adds new electrons to the sphere */
    if (MASTER(cr))
    {
        /* Loop over solvation shell atoms */
        for (i = wr->nindA_prot; i < wr->isizeA; i++)
        {
            /* No matter if wt->type is XRAY or Neutron, we here collect the electron density. */
            nelec = wr->nElectrons[wr->indexA[i]];
            gmx_envelope_solvent_density_addAtom(wt->envelope, x[wr->indexA[i]], nelec);
        }
    }
    gmx_envelope_solvent_density_bcast(wt->envelope, cr);

    /* Update FT of solvent density in the first steps a few times */
    bRecalcSolventFT = ( !gmx_envelope_bHaveSolventFT(wt->envelope)  ||
            wr->waxsStep <= 5                          ||
            wr->waxsStep == 10                         ||
            wr->waxsStep == 15                         ||
            wr->waxsStep == 20                         ||
            wr->waxsStep == wr->nfrsolvent - 1);
    if (wr->tau > 0.)
    {
        /* with exponential averaging, update FT once every tau/4 (first time after tau/8) */
        bRecalcSolventFT = bRecalcSolventFT ||
                ( (wr->waxsStep + ((int) wr->tausteps)/8) % ((int) wr->tausteps/4) ) == 0;
    }
    else if (wr->tau < -1e-6)
    {
        /* With non-weighted average (typically in a rerun) update the FT a couple of times */
        bRecalcSolventFT = bRecalcSolventFT || wr->waxsStep == 30 || wr->waxsStep == 50
                || wr->waxsStep == 100 || wr->waxsStep == 300 || wr->waxsStep == 1000;
    }
    else if  (fabs(wr->tau) < 1e-6)
    {
        bRecalcSolventFT = TRUE;
    }

    if (bRecalcSolventFT && MASTER(cr))
    {
        printf("Recalculating the FT of the solvent density around solute in waxs step %d (tau = %g, scattering type %d)\n",
                wr->waxsStep, wr->tau, t);
    }
    /* Update the FT of the solvent in A if bRecalcSolventFT */

    if (bRecalcSolventFT)
    {
        waxsTimingDo(wr, waxsTimeFourier, waxsTimingAction_start, 0, cr);
    }

    gmx_envelope_solventFourierTransform(wt->envelope, q + qstart, qhomenr, bRecalcSolventFT, &ft_re, &ft_im);

    if (bRecalcSolventFT)
    {
        waxsTimingDo(wr, waxsTimeFourier, waxsTimingAction_end, 0, cr);
    }

    wr->bRecalcSolventFT_GPU = bRecalcSolventFT ;
}


#define WAXS_SCALEI0_ADD_TO_SOLVENT_OF_A 0
static void
scaleI0_getAddedDensity(t_waxsrec *wr, t_commrec *cr)
{
    double deltaAq0, deltaI, Inow, c, NelecEnvelope;
    int qstart, qhomenr, i, t;
    t_waxs_datablock wd;
    t_waxsrecType *wt;
    real *ft_re, *ft_im;

    for (t = 0; t < wr->nTypes; t++)
    {
        wt = &wr->wt[t];
        wd = wt->wd;

        /* A few checks first */
        if (wt->targetI0 == -1.)
        {
            gmx_fatal(FARGS, "Trying to scale I(q=0), but no target I(q=0) found. Provide input intensity\n or "
                    "an environment varialbe GMX_WAXS_SCALE_I0, or set mdp option waxs-scale-i0 to no\n");
        }
        if (wt->qvecs->abs[0] != 0.0)
        {
            gmx_fatal(FARGS, "Trying to scale I(q=0) to target I(q=0), but q = 0 is not computed\n");
        }
        if (wr->bVacuum)
        {
            gmx_fatal(FARGS, "Cannot scale I(q=0) in vacuum since electron density is added to the solvent aound the protein\n");
        }
        /* Should this be taken from an input or from the experimental curve? It is here a 2nd time for rerun purposes. MH*/
        char *buf;
        if ((buf = getenv("GMX_WAXS_I0")) != NULL)
        {
            wt->targetI0 = atof(buf);
            fprintf(stderr, "\nWAXS-MD rerunpart: Found environment variable GMX_WAXS_I0 = %.1f\n", wt->targetI0);
        }
        else
        {
            wt->targetI0 = wt->Iexp[0];
            if (MASTER(cr))
            {
                fprintf(stderr, "\nWAXS-MD rerunpart: Did not  found GMX_WAXS_I0, taking the experimental value: %.1f\n", wt->targetI0);
            }
        }

        if (wt->targetI0 == 0)
        {
            fprintf(stderr, "\nTarget I0 is: %.1f. This should not happen.\n", wt->targetI0);
        }
        /* End of new part MH*/
        qstart  = wt->qvecs->qstart;
        qhomenr = wt->qvecs->qhomenr;

        if (qstart == 0)
        {
            /* This should be on the master with the smallest |q|, but double-check: */
            if (!MASTER(cr))
            {
                gmx_fatal(FARGS, "scaleI0_getAddedDensity(): expected to be here on the Master\n");
            }
            if (norm2(wt->qvecs->q[0]) != 0.)
            {
                gmx_fatal(FARGS, "On this node, have qstart = 0, but norm2(q[0]) = %g\n", norm2(wt->qvecs->q[0]));
            }

            Inow          = sqr(wd->avA[0].re - wd->avB[0].re) + (wd->avAsq[0] - dsqr(wd->avA[0].re))
                                                                                                        - (wd->avBsq[0] - dsqr(wd->avB[0].re));
            deltaI        = wt->targetI0 - Inow;
            deltaAq0      = -(wd->avA[0].re - wd->avB[0].re) + sqrt( dsqr(wd->avA[0].re - wd->avB[0].re) + deltaI);
            if (WAXS_SCALEI0_ADD_TO_SOLVENT_OF_A)
            {
                NelecEnvelope = gmx_envelope_solvent_density_getNelecTotal(wt->envelope);
                /* Add density of c * rho[solvent] to A system*/
                c             = deltaAq0 / NelecEnvelope;
                printf("Adding %g electrons to A (%.3f %% increase) in order to scale I(q=0) to %g\n",
                        deltaAq0, 100.*c, wt->targetI0);
                if (wt->type == escatterNEUTRON)
                    gmx_fatal(FARGS, "WAXS_SCALEI0_ADD_TO_SOLVENT_OF_A is not implemented with Neutron scattering.");
            }
            else
            {
                c = deltaAq0 / gmx_envelope_getVolume(wt->envelope);
                printf("Removing %g electrons from B (%.3f %% decrease) in order to scale I(q=0) to %g\n",
                        deltaAq0, 100.*c/wr->waxs_solv->avDensity, wt->targetI0);
            }
        }
        if (PAR(cr))
        {
            gmx_bcast(sizeof(double), &c, cr);
        }

        /* Pick up Fourier transform of solvent within envelope */
        if (WAXS_SCALEI0_ADD_TO_SOLVENT_OF_A)
        {
            gmx_envelope_solventFourierTransform(wt->envelope, wt->qvecs->q + qstart, qhomenr, FALSE, &ft_re, &ft_im);
        }
        else
        {
            gmx_envelope_unitFourierTransform(wt->envelope, wt->qvecs->q + qstart, qhomenr, &ft_re, &ft_im);
        }

        if (wd->avAcorr == NULL)
        {
            snew(wd->avAcorr, qhomenr);
        }

        for (i = 0; i < qhomenr; i++)
        {
            wd->avAcorr[i].re = c * ft_re[i];
            wd->avAcorr[i].im = c * ft_im[i];
        }
    }
}

/*
 * Fixing the bulk solvent density to the value specified with mdp option waxs-solvdens.
 * See: Chen and Hub, Biophys. J., 107, 435-447 (2014), page 438
 */
static void
fix_solvent_density(t_waxsrec *wr, t_commrec *cr, int t)
{
    gmx_bool       bRecalcSolventFT;
    real          *ftSolvent_re, *ftSolvent_im, *ftEnv_re, *ftEnv_im, c, AmB_re, AmB_im, FT_diff_re, FT_diff_im;
    real           delta_rho_over_rho_bulk, delta_rho_B;
    int            qstart, qhomenr, i, wstep = wr->waxsStep+1;
    rvec          *q;
    double         elec2NSL_factor, nslPerMol;
    t_waxsrecType *wt;

    wt = &wr->wt[t];

    qstart  = wt->qvecs->qstart;
    qhomenr = wt->qvecs->qhomenr;
    q       = wt->qvecs->q;

    /* Factor to translate electron density into NSL density at this deuterium concentration */
    if (wt->type == escatterXRAY)
    {
        elec2NSL_factor = 1;
    }
    else
    {
        /* Water has 10 electrons per molecule, translating into NSL per molecule
         * NOTE: Strictly speaking, we would have to take this from the pure-solvent trajectory
         *       However, since the solvent is mainly water, this would make only a tiny difference.
         */
        nslPerMol = NEUTRON_SCATT_LEN_O +
                2 * ( (1. - wt->deuter_conc) * NEUTRON_SCATT_LEN_1H + wt->deuter_conc * NEUTRON_SCATT_LEN_2H);
        elec2NSL_factor = nslPerMol/10;
    }

    /* Pick up Fourier transform of solvent electron density in A-system, that is of the solvent
     * inside the envelope around the protein.
     * Important: ft_re and ft_im is always in unites of electron density (not NSL density) */
    gmx_envelope_solventFourierTransform(wt->envelope, q + qstart, qhomenr, FALSE, &ftSolvent_re, &ftSolvent_im);

    /* Add a small density to solvent around protein, proportional to the density already present */
    if (PAR(cr))
    {
        gmx_bcast(sizeof(double), &wr->solElecDensAv, cr);
    }

    /* Factor required to get *electron* density to the given density */
    delta_rho_over_rho_bulk = (wr->givenElecSolventDensity - wr->solElecDensAv) / wr->solElecDensAv;
    if (fabs(delta_rho_over_rho_bulk) > 0.05)
    {
        fprintf(stderr,
                "\nWARNING, you are correcting the density of the PROTEIN SYSTEM by more than 5%%.\n"
                "   delta_rho / rho = %g %%\n"
                "This may indicate that your waxs-solvent group is lacking atoms, that your\n"
                "waxs-solvdens is wrong, or that you have some other problem. The solvent density\n"
                "correction is meant for small corrections, typically by 1%% or less.\n\n",
                delta_rho_over_rho_bulk*100);
    }
    if (wr->bUseGPU == FALSE)
    {
        for (i = 0; i < qhomenr; i++)
        {
            /* ftSolvent_re/ftSolvent_im  are in units "number of electrons". For neutron scattering,
               translate to units number of NSL  */
            wt->wd->A[i].re += delta_rho_over_rho_bulk * elec2NSL_factor * ftSolvent_re[i];
            wt->wd->A[i].im += delta_rho_over_rho_bulk * elec2NSL_factor * ftSolvent_im[i];
        }
    }
    //fprintf(stderr, "norm2 %f, qstart %d",norm2(wt->qvecs->q[0]), wt->qvecs->qstart );
    if (MASTER(cr) && norm2(wt->qvecs->q[0]) == 0. && wt->qvecs->qstart == 0)
    {

        //fprintf(stderr, "Correct # electrons: delta_rho_over_rho_bulk %f , nElecAddedA %f \n", delta_rho_over_rho_bulk,wr->nElecAddedA );
        wr->nElecAddedA = 1.0/wstep * ( (wstep-1)*wr->nElecAddedA + delta_rho_over_rho_bulk*ftSolvent_re[0]);
        if (wt->type == escatterXRAY)
        {
            printf("Fixing solvent density: Added %g electrons to solvation shell around solute"
                   " (added density of %g)\n",
                   delta_rho_over_rho_bulk*ftSolvent_re[0],
                   wr->givenElecSolventDensity - wr->solElecDensAv);
        }
        else
        {
            printf("Fixing solvent density: Added %g electrons (%g NSLs) to solvation shell around solute"
                   " (added electron density of %g / NSL density of %g)\n",
                   delta_rho_over_rho_bulk*ftSolvent_re[0],
                   delta_rho_over_rho_bulk*elec2NSL_factor*ftSolvent_re[0],
                   wr->givenElecSolventDensity - wr->solElecDensAv,
                   elec2NSL_factor*(wr->givenElecSolventDensity - wr->solElecDensAv));
        }
    }

    /* In either of these cases, we will use the unit transform */
    if ( (wr->bDoingSolvent || wt->wd->DerrSolvDens))
    {
        gmx_envelope_unitFourierTransform(wt->envelope, q + qstart, qhomenr, &ftEnv_re, &ftEnv_im);
    }

    if (wr->bDoingSolvent)
    {
        /* Add a small constant density to the excluded solvent droplet */
        delta_rho_B = (wr->givenElecSolventDensity - wr->solElecDensAv_SysB);
        if (MASTER(cr) && fabs(delta_rho_B/wr->solElecDensAv_SysB) > 0.05)
        {
            fprintf(stderr,
                    "\nWARNING, you are correcting the density of the SOLVENT SYSTEM by more than 5%%.\n"
                    "   delta_rho / rho = %g %%\n"
                    "This may indicate that your waxs-solvent group is lacking atoms, that your\n"
                    "waxs-solvdens is wrong, or that you have some other problem. The solvent density\n"
                    "correction is meant for small corrections, typically by 1%% or less.\n\n",
                    delta_rho_B/wr->solElecDensAv_SysB*100);
        }

        if (MASTER(cr) && t == 0)
        {
            printf("Fixing solvent density: Added %g electrons to excluded solvent              (added density of %g)\n",
                   delta_rho_B*gmx_envelope_getVolume(wt->envelope), delta_rho_B);
            wr->nElecAddedB =
                1.0/wstep * ( (wstep-1)*wr->nElecAddedB + delta_rho_B*gmx_envelope_getVolume(wt->envelope));
        }
        /* Corrections in B takes the form of a pure homogenous background. */
        if(wr->bUseGPU == FALSE)
        {
            for (i = 0; i < qhomenr; i++)
            {
                /* ft_re/ft_im  are in units number of electrons. For neutron scattering, translate to units number of NSL  */
                wt->wd->B[i].re += delta_rho_B * elec2NSL_factor * ftEnv_re[i];
                wt->wd->B[i].im += delta_rho_B * elec2NSL_factor * ftEnv_im[i];
            }
        }
    }

    if (wt->wd->DerrSolvDens)
    {
        /* Uncertainty in D(q) due to uncertainty in the relative uncertainty of the solvent density:
           D[err] = 2 * drho / rho[bulk] * Re{ <A*-B*> . [ F(rho[s]) - rho[bulk].F(envelope) ] },
           where  drho/rho[bulk] is the relative uncertainty of the solvent given by the mdp option waxs-solvdens-uncert.
         */
        for (i = 0; i < qhomenr; i++)
        {
            /* In unites electrons or NSL: */
            AmB_re                  = wt->wd->avA[i].re - wt->wd->avB[i].re;
            AmB_im                  = wt->wd->avA[i].im - wt->wd->avB[i].im;
            /* Always in units electrons, so translate to electrons OR NSL with factor elec2NSL_factor: */
            FT_diff_re              = (ftSolvent_re[i] - wr->givenElecSolventDensity * ftEnv_re[i]) * elec2NSL_factor;
            FT_diff_im              = (ftSolvent_im[i] - wr->givenElecSolventDensity * ftEnv_im[i]) * elec2NSL_factor;
            wt->wd->DerrSolvDens[i] = 2 * wr->solventDensRelErr * ( AmB_re * FT_diff_re + AmB_im * FT_diff_im );
        }
    }
}


#define STR(x)   #x
#define SHOW_DEFINE(x) fprintf(stderr, "%s=%s\n", #x, STR(x))
static void
compute_scattering_amplitude (t_complex_d *scatt_amplitude, rvec *atomEnvelope_coord,
                                     int *atomEnvelope_type, int isize, int nprot,
                                     rvec *q, int *iTable, int qhomenr, real **aff_table, real *nsl_table,
                                     t_complex *grad_amplitude_bar,
                                     double *timing_ms)
{
    int        p, i;
    real       qdotx, aff;
    cvec       cv;
    t_complex  c;
    struct timespec t_start, t_end;
    long int delta_us;

    clock_gettime(CLOCK_MONOTONIC_RAW, &t_start);

    waxs_debug("Begin of compute_scattering_amplitude()\n");

    if (aff_table && nsl_table)
    {
        gmx_fatal(FARGS, "aff_table and nsl_table are both != NULL !\n");
    }
    if (!aff_table && !nsl_table)
    {
        gmx_fatal(FARGS, "aff_table and nsl_table are both == NULL !\n");
    }

    for (i = 0; i < qhomenr; i++)
    {
        scatt_amplitude[i] = cnul_d;
    }

    real       *p_aff, *p_qdotx,*re,*im,tmp;
    int        k,nq2,nReal,d,k0;
    REGISTER   sinm, cosm, *m_aff, *m_qdotx, *mRe, *mIm;

    nReal = sizeof(REGISTER)/sizeof(real);

    /* nq2 >= qhomenr and multiple of nReal (the # of reals per register) */
    nq2 = ((qhomenr-1)/nReal+1)*nReal;

    /* IMPORTANT: We must make sure that the float pointers are aligned in memory in same way as the
                  REGISTER pointers (e.g., REGISTER is __m256 for AVX_256 or __mm128 for FMA).
                  Otherwise, after casting a float* onto a REGISTER*, we may have in incorrect
                  aligntment of the REGISTER* variable, which may cause errors such as Segfaults.

                  For instance, floats are 4-byte-aligned, where as __m256 are 16-byte-aligned.

                  To make sure that the float arrays correctly aligned, they must be alloccated with
                  snew_aligned and freed with sfree_aligned.

                  For an explanation, see e.g.:
                  https://stackoverflow.com/questions/25596379/simd-intrinsics-segmentation-fault
    */

    /* allocate 32-byte alinged float real arrays */
    snew_aligned(p_aff,   nq2, 32);
    snew_aligned(p_qdotx, nq2, 32);
    snew_aligned(re,      nq2, 32);
    snew_aligned(im,      nq2, 32);

    for (p = 0; p < isize; p++)
    {
        if (atomEnvelope_type[p] == 10 && p == 0 && FALSE)
        {
            fprintf(stderr, "scattering Amplitude AT = 10 , atom %d !\n", i);
        }

        /* put atomic form factors or NSL and phase qdotx in a linear (real*) array */
        if (aff_table)
        {
            /* Xray scattering, atomic form factor depends on q */
            for (i = 0; i < qhomenr; i++)
            {
                p_aff  [i] = aff_table[atomEnvelope_type[p] ][iTable[i]];
                p_qdotx[i] = iprod(q[i], atomEnvelope_coord[p]);
            }
        }
        else
        {
            /* Neutron scattering NSL does NOT depend on q */
            for (i = 0; i < qhomenr; i++)
            {
                /* To keep this function generic for Xray and Neutron (for now), write the identical NSL into an array.
                   Because the SINCOS function below is by far the most expensive anyway, this will hardly make any difference.
                 */
                p_aff  [i] = nsl_table[atomEnvelope_type[p] ];
                p_qdotx[i] = iprod(q[i], atomEnvelope_coord[p]);
            }
        }

        /* Cast the real* to REGISTER* arrays */
        m_aff    = (REGISTER*) p_aff;
        m_qdotx  = (REGISTER*) p_qdotx;
        mRe      = (REGISTER*) re;
        mIm      = (REGISTER*) im;

        /* the SSE or AVX loop
           SSE or AVX128: float: 4 sincos per call. double: 2 sincos per call
           AVX256:        float: 8 sincos per call. double: 4 sincos per call
         */
        for (k = 0; k < qhomenr; k += nReal)
        {
            REGISTER_SINCOS(*m_qdotx, &sinm, &cosm);
            *mRe = REGISTER_MUL(*m_aff, cosm);
            *mIm = REGISTER_MUL(*m_aff, sinm);

            m_qdotx++;
            m_aff++;
            mRe++;
            mIm++;
        }

        for (i = 0; i < qhomenr; i++)
        {
            /* Requires a real to double conversion */
            scatt_amplitude[i].re += re[i];
            scatt_amplitude[i].im += im[i];
        }

        if (grad_amplitude_bar && p < nprot)
        {
            k0 = p*qhomenr;
            /* Store: [ i . f_k(q) . exp(iq*x) ]* = (i.SF)*   */
            for (i = 0; i < qhomenr; i++)
            {
                /* Use [(a+i.b).i]* = -b-i*a */
                k                        = k0 + i ;
                grad_amplitude_bar[k].re = -im[i] ;
                grad_amplitude_bar[k].im = -re[i] ;
            }
        }
    }

    sfree_aligned(p_aff);
    sfree_aligned(p_qdotx);
    sfree_aligned(re);
    sfree_aligned(im);


#ifdef PRINT_A_dkI
    FILE *fp;
    fp = fopen("./uncorrected_scattering_amplitude_CPU.txt", "w");
    fprintf(fp, "Atom-number HOST start is: %d\n", isize);
    fprintf(fp, "qhomenr HOST after loop is: %d\n", qhomenr);

    for (i = 0;i < qhomenr;i++)
    {
        fprintf(fp, "Real-part of scattering amplitude [ %d ] :  %f \n", i,  scatt_amplitude[i].re);
        fprintf(fp, "Im  -part of scattering amplitude [ %d ] :  %f \n", i,  scatt_amplitude[i].im);
        fprintf(fp, " \n");
    }
    fclose(fp);
#endif

    /* Return elapsed time in milliseconds */
    clock_gettime(CLOCK_MONOTONIC_RAW, &t_end);
    delta_us   = (t_end.tv_sec - t_start.tv_sec) * 1000000 + (t_end.tv_nsec - t_start.tv_nsec) / 1000;
    *timing_ms = delta_us/1000.;

    waxs_debug("End of compute_scattering_amplitude()\n");
}

extern double md_CMSF (t_cromer_mann cmsf, real lambda, real sin_theta,
        real alpha, real delta )
/*
 * return Cromer-Mann fit for the atomic scattering factor:
 * sin_theta is the sine of half the angle between incoming and scattered
 * vectors. See g_sq.h for a short description of CM fit.
 */
{
    int i,success;
    real tmp = 0.0, k2, Q2;
    real a[4],b[4];
    real c;

    /*
     *
     * f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]
     *             i=1,4
     */

    /* k2 is in units A^-2 (according to parameters b[i]), whereas lambda in nm */
    k2 = (sqr (sin_theta) / sqr (10.0 * lambda));
    tmp = cmsf.c;
    for (i = 0; (i < 4); i++)
    {
        tmp += cmsf.a[i] * exp (-cmsf.b[i] * k2);
    }

    /* scale AFF by [ 1 + alpha*exp(-Q^2/2delta^2) ]  --- only used for water.
       Now this is done on the force field level, however. Therefore, alpha is always == 0.
     * Note: we use alpha instead of the (alpha-1) used in eq. 5, Sorenson et al, JCP 113:9194, 2000
     * Note: Q2 is in units nm^-2, delta in nm^-1 */

    if ( alpha != 0 )
    {
        Q2 = sqr(4*M_PI*sin_theta/lambda);
        tmp *= ( 1 +alpha*exp(-Q2/(2*delta*delta)) );
    }
    return tmp;
}


/* Make a table of NSLs found in the topology, and add NSLs of 1H and 2H */
static void
fill_neutron_scatt_len_table(t_scatt_types *scattTypes, t_waxsrecType *wt, real backboneDeuterProb)
{
    int i;
    real probD, nsl, nslH, nslD;

    wt->nnsl_table = scattTypes->nnsl + 4;
    nslH           = NEUTRON_SCATT_LEN_1H;
    nslD           = NEUTRON_SCATT_LEN_2H;

    snew (wt->nsl_table, wt->nnsl_table);
    for (i = 0; i < scattTypes->nnsl; i++)
    {
        wt->nsl_table[i] = scattTypes->nsl[i].cohb;
    }
    wt->nsl_table[i++] = nslH;
    wt->nsl_table[i++] = nslD;

    /* Optinally, we may assign a mixed NSL of 1H and 2H, corresponding to the deuterium concentration.
       These are used if wr->bStochasticDeuteration == FALSE.
       First: deuteratable groups: */
    probD                  = wt->deuter_conc;
    nsl                    = probD*nslD + (1.0 - probD)*nslH;
    wt->nsl_table[i++]     = nsl;
    /* Next: deuteratable backbone hydrogens: */
    probD                  = wt->deuter_conc * backboneDeuterProb;
    nsl                    = probD*nslD + (1.0 - probD)*nslH;
    wt->nsl_table[i++]     = nsl;
}

/* Return the NSL type index, taking the deuterium concentration into account.

   wr->bStochasticDeuteration == TRUE:
   If the NSl is  NSL_H_DEUTERATABLE and NSL_H_DEUTERATABLE_BACKBONE, they
   are randomly assigned to 1H or 2H.
   wr->bStochasticDeuteration == FALSE:
   If the NSl is  NSL_H_DEUTERATABLE and NSL_H_DEUTERATABLE_BACKBONE, a linear
   interpolation between 1H and 2H is assigned.
 */
static int
get_nsl_type(t_waxsrec *wr, int t, int nslTypeIn)
{
    int nslTypeOut, nslType_1H, nslType_2H, nslType_deuteratable_mixed, nslType_deutBackbone_mixed;
    double probDeuter;
    t_waxsrecType *wt;

    wt = &wr->wt[t];

    /* types of 1H, 2H, deuteratable, and deuteratalbe backbone are always at the end of the nsl_table,
       see fill_neutron_scatt_len_table() */
    nslType_1H                 = wt->nnsl_table - 4;
    nslType_2H                 = wt->nnsl_table - 3;
    nslType_deuteratable_mixed = wt->nnsl_table - 2;
    nslType_deutBackbone_mixed = wt->nnsl_table - 1;

    if (fabs(wt->nsl_table[nslTypeIn] - NSL_H_DEUTERATABLE_BACKBONE) < 1e-5)
    {
        /* Backbone hydrogen */
        if (wr->bStochasticDeuteration)
        {
            probDeuter = wt->deuter_conc * wr->backboneDeuterProb;
            return (gmx_rng_uniform_real(wr->rng) < probDeuter) ? nslType_2H : nslType_1H;
        }
        else
        {
            return nslType_deutBackbone_mixed;
        }
    }
    else if (fabs(wt->nsl_table[nslTypeIn] - NSL_H_DEUTERATABLE) < 1e-5)
    {
        /* other deuteratable hydrogen (typically polar hydrogen) */
        if (wr->bStochasticDeuteration)
        {
            return (gmx_rng_uniform_real(wr->rng) < wt->deuter_conc) ? nslType_2H : nslType_1H;
        }
        else
        {
            return nslType_deuteratable_mixed;
        }
    }
    else
    {
        return nslTypeIn;
    }
}

static void
compute_atomic_form_factor_table(t_scatt_types *scattTypes, t_waxsrecType *wt, t_commrec *cr)
{
    int i, j;

    wt->naff_table = scattTypes->ncm;
    snew (wt->aff_table, wt->naff_table);

    if ( !wt->qvecs )
    {
        gmx_fatal(FARGS,"Atomic form factors cannot be calculated without a defined set of q-vectors."
                " Please initialise them first!\n");
    }

    if (MASTER(cr))
    {
        printf("Building atomic form factor table for %d SF-tytpes and %d |q|\n", wt->naff_table, wt->nq);
    }
    for (i = 0; (i < wt->naff_table); i++)
    {
        snew(wt->aff_table[i], wt->nq);
        for (j = 0; j < wt->nq; j++)
        {
            wt->aff_table[i][j] = CMSF_q(scattTypes->cm[i], wt->qvecs->abs[j]);
        }
    }
}


static void
copy_ir2waxsrec( t_waxsrec *wr, t_inputrec *ir, t_commrec *cr )
{
    waxs_debug("copy_ir2waxsrec");
    real tau,dt;
    int i;
    wr->nstlog     = ir->waxs_nstlog;
    wr->nfrsolvent = ir->waxs_nfrsolvent;
    wr->bVacuum    = FALSE;
    wr->stepCalcNindep = 0;
    waxs_debug("wr->npbcatom = ir->waxs_npbcatom");

    wr->npbcatom = ir->waxs_npbcatom;
    snew(wr->pbcatoms, wr->npbcatom);

    for (i = 0; i < wr->npbcatom; i++)
    {
        waxs_debug("in for loop, before assigning\n");
        wr->pbcatoms[i] = ir->waxs_pbcatoms[i];   // Here is some ugly bug when not using openMPI or compiling on mac. Works great on gwdg.
        waxs_debug("in for loop, after assigning\n");

    }

    wr->xray_energy             = ir->waxs_xray_energy;
    wr->bScaleI0                = ir->ewaxs_bScaleI0;
    wr->givenElecSolventDensity = ir->waxs_denssolvent;
    wr->solventDensRelErr       = ir->waxs_denssolventRelErr;
    if (wr->givenElecSolventDensity == 0 && wr->solventDensRelErr > 0)
    {
        gmx_fatal(FARGS, "To account for the uncertainty of the solvent density, you must specify the expected solvent density.\n"
                "Specify the solvent density in the mdp file, e.g. to 334 e/nm^3\n");
    }
    wr->bCorrectBuffer          = (ir->ewaxs_correctbuff == waxscorrectbuffYES);
    wr->bFixSolventDensity      = (ir->waxs_denssolvent > GMX_FLOAT_EPS);
    wr->bBayesianSolvDensUncert = (ir->ewaxs_solvdensUnsertBayesian == ewaxsSolvdensUncertBayesian_YES);
    wr->ewaxsaniso              = ir->ewaxsaniso;
    wr->solv_warn_lay           = ir->waxs_warn_lay;

    tau = wr->tau = ir->waxs_tau ;
    wr->nstcalc   = ir->waxs_nstcalc ;
    dt            = 1.0*wr->nstcalc*ir->delta_t;
    wr->tausteps  = wr->tau/dt;
    if (dt<=0.)
        gmx_fatal(FARGS,"dt for WAXS intensity calculation %g is <= 0\n",dt);
    if ( tau > 0.0 )
    {
        wr->scale = exp( -1.0*dt/tau );
        wr->bSwitchOnForce = TRUE;
        wr->stepCalcNindep = (int)((tau/dt))/2;
    }
    else if (tau == -1.0)
    {
        wr->scale          = 1.0;       /* non-weighted averaging */
        wr->bSwitchOnForce = FALSE;
    }
    else if (tau == 0.0)
    {
        wr->scale          = 0.0;       /* no averaging, i.e., instantaneous coupling. */
        wr->bSwitchOnForce = FALSE;
    }
    else
    {
        gmx_fatal(FARGS,"Illegal value for waxs_tau found (%g). Allowed: -1, 0, >0)\n", tau);
    }

    wr->ewaxs_Iexp_fit          = ir->ewaxs_Iexp_fit;
    wr->potentialType           = ir->ewaxs_potential;

    wr->weightsType             = ir->ewaxs_weights;
    wr->ewaxs_ensemble_type     = ir->ewaxs_ensemble_type;
    wr->t_target                = ir->waxs_t_target;

    wr->kT   = ir->opts.ref_t[0] * BOLTZ;
    if (MASTER(cr))
    {
        fprintf(stderr, "WAXS-MD: kT = %g\n",wr->kT);
        fprintf(stderr, "WAXS-MD: Computing # independent points every %d steps.\n", wr->stepCalcNindep);
    }

    /* init stuff that is specific to the type of scattering (xray or neutron) */
    wr->nTypes = ir->waxs_nTypes;
    wr->bDoingNeutron = FALSE;
    snew(wr->wt,  wr->nTypes);
    //snew(wr->GPU_data, );
    for (i = 0; i<wr->nTypes; i++)
    {
        wr->wt[i]             = init_t_waxsrecType();
        wr->wt[i].fc          = ir->waxs_fc         [i];
        wr->wt[i].type        = ir->escatter        [i];
        wr->bDoingNeutron     = (wr->bDoingNeutron || ir->escatter[i] == escatterNEUTRON);
        wr->wt[i].nq          = ir->waxs_nq         [i];
        wr->wt[i].minq        = ir->waxs_start_q    [i];
        wr->wt[i].maxq        = ir->waxs_end_q      [i];
        wr->wt[i].deuter_conc = ir->waxs_deuter_conc[i];
        sprintf(wr->wt[i].saxssansStr, "%s", ir->escatter[i] == escatterXRAY ? "SAXS" : "SANS");
        sprintf(wr->wt[i].scattLenUnit, "%s", ir->escatter[i] == escatterXRAY ? "electrons" : "NSLs");
    }
    wr->J = ir->waxs_J ;
    if (tau>0.0)
    {
        wr->nAverI = 2*tau/dt;
    }
    else
    {
        wr->nAverI = 100;
    }

    /* Hardcoded - uh, ugly. Or should we depend this on tau? */
    if (MASTER(cr))
    {
        fprintf(stderr, "WAXS-MD: Running sigma of I(q) computed from the last %d intensities (tau/dt = %g/%g) - NOT used at the moment.\n",
                wr->nAverI, tau, dt);
    }

    /* In case of SAXS ensemble refinement: set initial weights of states */
    wr->ensemble_nstates = ir->waxs_ensemble_nstates;
    snew(wr->ensemble_weights,      wr->ensemble_nstates);
    snew(wr->ensemble_weights_init, wr->ensemble_nstates);

     wr->ensemble_weights_fc = ir->waxs_ensemble_fc;
    for (i = 0; i < wr->ensemble_nstates; i++)
    {
        wr->ensemble_weights     [i] = ir->waxs_ensemble_init_w[i];
        wr->ensemble_weights_init[i] = ir->waxs_ensemble_init_w[i];
    }
    if (MASTER(cr) && wr->ensemble_nstates)
    {
        fprintf(stderr, "WAXS-MD: Found initial weights for %d states for ensemble refinement =", wr->ensemble_nstates);
        for (i = 0; i < wr->ensemble_nstates; i++)
        {
            fprintf(stderr, " %g", wr->ensemble_weights_init[i]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "WAXS-MD: Force constant for ensemble weights = %g", wr->ensemble_weights_fc);
    }

    /* other structures to be initialised later. */
}


/* Returns a continuous array (size mtop->natoms) of Cromer-Mann or NSL (bNeutron = TRUE) types from mtop */
int*
mtop2scattTypeList(gmx_mtop_t *mtop , gmx_bool bNeutron)
{
    int mb, isum=0, nmols, imol=0, natoms_mol, iatom=0, itot=0, *scattTypeId, i, stype;
    t_atoms *atoms;
    /* Bookkeeping: natoms = Sum _i ^ nmolblock ( molblock[i]->nmol * molblock[i]->natoms_mol ) */
    /* moltype.atoms->nr = molblock[i]->natoms_mol */
    /* Loop over mblocks instead and use its information to write */

    snew(scattTypeId, mtop->natoms);
    for (i = 0; i < mtop->natoms; i++)
    {
        scattTypeId[i] = -1000;
    }
    /* Loop over molecule types */
    for(mb = 0; mb < mtop->nmolblock; mb++)
    {
        atoms      = &mtop->moltype[mtop->molblock[mb].type].atoms;
        nmols      = mtop->molblock[mb].nmol;
        natoms_mol = mtop->molblock[mb].natoms_mol;

        /* Loop over atoms in this molecule type */
        for(iatom = 0; iatom < natoms_mol; iatom++)
        {
            /* Pick NSL or Cromer-Mann type of this atom */
            stype = bNeutron ? atoms->atom[iatom].nsltype : atoms->atom[iatom].cmtype;

            /* Loop over moleculs of this molecule type */
            for (imol = 0; imol < nmols; imol++)
            {
                itot = isum + imol*natoms_mol + iatom;
                scattTypeId[itot] = stype;
            }
        }
        isum += nmols*natoms_mol;
    }
    /* For debugging, make sure that Cromer-Mann or NSL type of *every* atom was set. */
    for (i = 0; i < mtop->natoms; i++)
    {
        if (scattTypeId[i] == -1000)
        {
            gmx_fatal(FARGS, "scattTypeId[%d] was not set in mtop2sftype_list\n",i);
        }
    }
    return scattTypeId;
}

/* Assume both solvent and solute .tpr files are set up identically, allowing matching of the atomic form factors.
   Do not allow types that are only found in solvent, as this would indicate some buffer mismatch between solute
   and solvent systems. */
void redirect_md_solventtypes(gmx_mtop_t *solvent_top, t_scatt_types *scattTypesSolute)
{
    /* gmx_mtop_t *soltop = ws->mtop; */
    t_scatt_types  *solvent_scattTypes = &solvent_top->scattTypes;
    t_atoms *atoms;
    int mb, isum = 0, itot, iatom, i, j, k, ngrps_solvent ;
    gmx_bool bSolvent;

    ngrps_solvent = get_actual_ngroups( &(solvent_top->groups), egcWAXSSolvent );
    fprintf(stderr, "In water-background: ngrps_solvent = %d. \n", ngrps_solvent );

    /* This section is somewhat shared with grompp's remap to unique atom-types. Do we combine later? */
    /* Set up redirect map. Although there normally should be only one moltype and molblock,
     * we should make allowances for complex solvents in the future. */

    for(mb = 0 ; mb < solvent_top->nmoltype; mb++)
    {
        atoms = &solvent_top->moltype[solvent_top->molblock[mb].type].atoms;
        for(iatom = 0; iatom < atoms->nr; iatom++)
        {
            itot = iatom + isum;

            bSolvent = ( ggrpnr(&(solvent_top->groups),egcWAXSSolvent, itot) < ngrps_solvent );
            if  (bSolvent)
            {
                /* Look for idenitcal Cromer-Mann parameters in solute topology */
                j = atoms->atom[iatom].cmtype;
                k = search_scattTypes_by_cm(scattTypesSolute, &(solvent_scattTypes->cm[j]));
                if (k == NOTSET)
                {
                    gmx_fatal(FARGS,"Cromer-Mann type %d of solvent not found in main system\n\n"
                            "We do not allow using waxs_solvents that has scattering types"
                            " not found in the solute system! Please use waxs_solvents molecules that are exactly like"
                            " or are subsets of the solvent in the main simulation.\n",j);
                }
                else if (k >= scattTypesSolute->ncm || k < 0)
                {
                    gmx_fatal(FARGS,"search_scattTypes_by_cm returned illegal sftype index (%d). (Allowed 0 to %d)",
                            k, scattTypesSolute->ncm);
                }
                else
                {
                    atoms->atom[iatom].cmtype = k;
                    fprintf(stderr, "Redirecting solvent Cromer-Mann type %d (atom %d) to main system Cromer-Mann type %d\n", j, iatom, k);
                }

                /* Optionally: Look or neutron scattering length in solute topology. Only needed if we have
                   NSL types in the solute system, otherwise we don't do neutron scattering anyway. */
                j = atoms->atom[iatom].nsltype;
                if (j >= 0 && scattTypesSolute->nnsl > 0)
                {
                    k = search_scattTypes_by_nsl(scattTypesSolute, &(solvent_scattTypes->nsl[j]));
                    if (k == NOTSET)
                    {
                        gmx_fatal(FARGS, "Neutron scattering length type %d of solvent not found in main system (cohb = %g)\n\n"
                                "We do not allow using waxs_solvents that has NEUTRON scattering types\n"
                                "not found in the main system! Please use waxs_solvents molecules that are exactly like\n"
                                "or are subsets of the solvent in the main simulation.\n", j, solvent_scattTypes->nsl[j].cohb);
                    }
                    else if (k >= scattTypesSolute->nnsl || k < 0)
                    {
                        gmx_fatal(FARGS,"search_scattTypes_by_nsl returned illegal NSL type index (%d). (Allowed 0 to %d)",
                                k, scattTypesSolute->nnsl);
                    }
                    else
                    {
                        atoms->atom[iatom].nsltype = k;
                        fprintf(stderr, "Redirecting solvent NSL type %d (atom %d) to main system NSL type %d\n", j, iatom, k);
                    }
                }
            }
        }
    }
}

/* Return the number of unconnected molecules in the solute */
static int
nSoluteMolecules(gmx_mtop_t *mtop)
{
    int      mb_last = -1, imol_last = -1, nSoluteMols = 0;
    int      nmols, imol, iatom, itot, isum = 0, mb, natoms_mol, ngrps_solute;
    t_atoms *atoms;
    gmx_bool bSolute;

    ngrps_solute = get_actual_ngroups( &(mtop->groups), egcWAXSSolute );
    for(mb = 0; mb < mtop->nmolblock; mb++)
    {
        atoms      = &mtop->moltype[mtop->molblock[mb].type].atoms;
        nmols      = mtop->molblock[mb].nmol;
        natoms_mol = mtop->molblock[mb].natoms_mol;

        /* molecules */
        for (imol = 0; imol < nmols; imol++)
        {
            /* atoms in molecule */
            for(iatom = 0; iatom < natoms_mol; iatom++)
            {
                itot = isum + imol*natoms_mol + iatom;
                bSolute = (ggrpnr( &(mtop->groups), egcWAXSSolute, itot ) < ngrps_solute );
                if (bSolute && (mb != mb_last || imol != imol_last))
                {
                    nSoluteMols++;
                    mb_last   = mb;
                    imol_last = imol;
                }
            }
        }
        isum += nmols*natoms_mol;
    }
    printf("Found %d unconnected molecule%s in the solute group.\n", nSoluteMols, nSoluteMols > 1 ? "s" : "");
    return nSoluteMols;
}


/* Setup indices of protein and solvent */
static void
prep_md_indices(atom_id **indA_prot, int *nindA_prot, atom_id **indA_solv, int *nindA_solv, gmx_mtop_t *mtop,
        gmx_bool bDoProt, gmx_bool bDoSolv)
{
    int i, natoms = mtop->natoms;
    int ngrps_solute  = get_actual_ngroups( &(mtop->groups), egcWAXSSolute );
    int ngrps_solvent = get_actual_ngroups( &(mtop->groups), egcWAXSSolvent );

    atom_id *prot=NULL, *solv=NULL;
    int nprot=0, nsolv=0;

    for (i = 0; i < natoms; i++)
    {
        if ( bDoSolv && ggrpnr( &(mtop->groups), egcWAXSSolvent, i ) < ngrps_solvent )
        {
            nsolv++;
            srenew(solv, nsolv);
            solv[nsolv-1] = i;
        }
        else if ( bDoProt && ggrpnr( &(mtop->groups), egcWAXSSolute, i ) < ngrps_solute )
        {
            nprot++;
            srenew(prot, nprot);
            prot[nprot-1] = i;
        }
    }
    if (bDoProt)
    {
        *nindA_prot = nprot;
        *indA_prot  = prot;
        fprintf(stderr, "Created array of %d solute atoms.\n",*nindA_prot);
    }
    if (bDoSolv)
    {
        *nindA_solv = nsolv;
        *indA_solv  = solv;
        fprintf(stderr, "Created array of %d solvent atoms.\n",*nindA_solv);
    }

}

void prep_fitting_weights(t_waxsrec *wr, atom_id **ind_RotFit, int *nind_RotFit, real *w_fit, gmx_mtop_t *mtop)
{
    int        natoms=mtop->natoms, nmols, imol, iatom, itot, isum = 0, mb, natoms_mol;
    gmx_bool   bFit;
    int        ngrps_fit  = get_actual_ngroups( &(mtop->groups), egcROTFIT );
    int        nfit = 0;
    atom_id   *fitgrp=NULL;
    real       m;
    t_atoms   *atoms;

    printf("WAXS-MD: Preparing weights for %s fitting...", wr->bMassWeightedFit ? "mass-weighted" : "non-weighted");
    fflush(stdout);

    for (itot = 0; itot < natoms; itot++)
    {
        w_fit[itot] = 0.;
    }
    /* This is a loop over all atoms. Global atom id is itot */
    /* Molecule types */
    for (mb = 0; mb < mtop->nmolblock; mb++)
    {
        atoms      = &mtop->moltype[mtop->molblock[mb].type].atoms;
        nmols      = mtop->molblock[mb].nmol;
        natoms_mol = mtop->molblock[mb].natoms_mol;

        /* molecules */
        for (imol = 0; imol < nmols; imol++)
        {
            /* atoms in molecule */
            for (iatom = 0; iatom < natoms_mol; iatom++)
            {
                itot = isum + imol*natoms_mol + iatom;
                bFit = ( ggrpnr( &(mtop->groups), egcROTFIT, itot ) < ngrps_fit );
                if (bFit)
                {
                    nfit++;
                    srenew(fitgrp, nfit);
                    fitgrp[nfit-1] = itot;
                    m              = atoms->atom[iatom].m;
                    w_fit[itot]    = wr->bMassWeightedFit ? m : 1.;
                    /* printf("Using atom %d for fitting (weight = %g)\n", itot, w_fit[itot]); */
                }
            }
        }
        isum += nmols*natoms_mol;
    }

    *nind_RotFit = nfit;
    *ind_RotFit  = fitgrp;
    printf("done. Using %d fitting atoms (ptr = %p).\n", *nind_RotFit, ind_RotFit);
}


/* allocate memory for the local storage of the averages A(q), B(q)
   A(q)-B(q), grad[k](A*(q)), and A(q)*grad[k](A*(q)

   Allocating for waxsType t
 */
static void
alloc_waxs_datablock(t_commrec *cr, t_waxsrec *wr, int nindA_prot, int t)
{
    int qhomenr,i, nIvalues = -1, nqvalues = -1, nabs;
    gmx_large_int_t nByte = 0;
    gmx_large_int_t nByteForce = 0;
    gmx_large_int_t nByteInten = 0;
    t_waxsrecType *wt;
    t_waxs_datablock wd;

    wt      = &wr->wt[t];
    qhomenr = wt->qvecs->qhomenr;

    snew(wd, 1);

    switch(wr->ewaxsaniso)
    {
    case ewaxsanisoNO:
        /* Store intensity for each |q| */
        nIvalues = wt->qvecs->nabs;
        break;
    case ewaxsanisoYES:
    case ewaxsanisoCOS2:
        /* Store intensity for all q-vectors */
        nIvalues = wt->qvecs->ind[wt->qvecs->nabs];
        break;
    default:
        gmx_fatal(FARGS, "This aniotropy (%d) is not supported\n", wr->ewaxsaniso);
    }
    wd->nIvalues = nIvalues;
    nabs = wt->qvecs->nabs;

    /* init norm of cumulative average to zero */
    wd->normA = 0.;
    wd->normB = 0.;

    /* init average overall densities */
    wd->avAsum = 0;
    wd->avBsum = 0;

    if (wr->bCalcPot)
    {
        snew(wd->vAver,  nIvalues);
        snew(wd->vAver2, nIvalues);
        /* nByte += 2*nIvalues*sizeof(real); */
        nByte += nIvalues*(sizeof(wd->vAver[0]) + sizeof(wd->vAver2[0]));
    }

    wd->B                           = NULL;
    wd->IA = wd->IB = wd->Ibulk     = NULL;
    wd->I_scaleI0                   = NULL;
    wd->I_varA = wd->I_varB         = NULL;
    wd->varI = wd->varIA            = NULL;
    wd->varIB = wd->varIbulk        = NULL;
    wd->I_errSolvDens               = NULL;
    wd->varI_avAmB2                 = NULL;
    wd->varI_varA = wd->varI_varB   = NULL;
    wd->avA = wd->avB = wd->avAcorr = NULL;
    wd->re_avBbar_AmB               = NULL;
    wd->avBsq                       = NULL;
    wd->Orientational_Av            = NULL;
    wd->avAqabs_re = wd->avBqabs_re = NULL;
    wd->avAqabs_im = wd->avBqabs_im = NULL;
    wd->Nindep                      = NULL;
    wd->Dglobal                     = NULL;
    wd->DerrSolvDens                = NULL;
    wd->avAglobal                   = NULL;
    wd->avAsqglobal                 = NULL;
    wd->avA4global                  = NULL;
    wd->av_ReA_2global              = NULL;
    wd->av_ImA_2global              = NULL;
    wd->avBglobal                   = NULL;
    wd->avBsqglobal                 = NULL;
    wd->avB4global                  = NULL;
    wd->av_ReB_2global              = NULL;
    wd->av_ImB_2global              = NULL;

    snew(wd->A,          qhomenr);
    snew(wd->avA,        qhomenr);
    snew(wd->av_ReA_2,   qhomenr);
    snew(wd->av_ImA_2,   qhomenr);
    snew(wd->avAsq,      qhomenr);
    snew(wd->avA4,       qhomenr);
    snew(wd->D,          qhomenr);
    snew(wd->I,          nIvalues);
    snew(wd->varI,       nIvalues);
    snew(wd->IA,         nIvalues);
    snew(wd->varIA,      nIvalues);
    snew(wd->avAqabs_re, nIvalues);
    snew(wd->avAqabs_im, nIvalues);
    if (wr->solventDensRelErr > 0)
    {
        snew(wd->DerrSolvDens,  qhomenr);
        snew(wd->I_errSolvDens, nIvalues);
    }
    /* nByte += nIvalues*(6*sizeof(double)) + qhomenr*(6*sizeof(double) + 1*sizeof(real)); */
    nByte += qhomenr*(sizeof(wd->A[0]) + sizeof(wd->avA[0]) + sizeof(wd->av_ReA_2[0]) + sizeof(wd->av_ImA_2[0]) +
            sizeof(wd->avAsq[0]) + sizeof(wd->avA4[0]) + sizeof(wd->D[0]) );
    nByte += nIvalues*(sizeof(wd->I[0]) + sizeof(wd->varI[0]) + sizeof(wd->IA[0]) +
            sizeof(wd->varIA[0]) + sizeof(wd->avAqabs_re[0]) + sizeof(wd->avAqabs_im[0]));

    for (i = 0; i < qhomenr; i++)
    {
        wd->avA  [i]    = cnul_d;
        wd->avAsq[i]    = 0.;
        wd->avA4 [i]    = 0.;
        wd->av_ReA_2[i] = 0.;
        wd->av_ImA_2[i] = 0.;
    }

    if (! wr->bVacuum)
    {
        snew(wd->IB,          nIvalues);
        snew(wd->Ibulk,       nIvalues);
        snew(wd->varIB,       nIvalues);
        snew(wd->varIbulk,    nIvalues);
        snew(wd->avBqabs_re,  nIvalues);
        snew(wd->avBqabs_im,  nIvalues);
        snew(wd->I_avAmB2,    nIvalues);
        snew(wd->I_varA,      nIvalues);
        snew(wd->I_varB,      nIvalues);
        if (wr->bScaleI0)
        {
            snew(wd->I_scaleI0, nIvalues);
        }
        snew(wd->varI_avAmB2,   nIvalues);
        snew(wd->varI_varA,     nIvalues);
        snew(wd->varI_varB,     nIvalues);
        snew(wd->B,             qhomenr);
        snew(wd->avB,           qhomenr);
        snew(wd->av_ReB_2,      qhomenr);
        snew(wd->av_ImB_2,      qhomenr);
        snew(wd->avBsq,         qhomenr);
        snew(wd->avB4,          qhomenr);
        snew(wd->re_avBbar_AmB, qhomenr);
        /* nByte += nIvalues*(12*sizeof(double)) + qhomenr*(6*sizeof(double) + 1*sizeof(real)); */
        nByte += qhomenr*(sizeof(wd->B[0]) + sizeof(wd->avB[0]) + sizeof(wd->av_ReB_2[0]) + sizeof(wd->av_ImB_2[0]) +
                sizeof(wd->avBsq[0]) + sizeof(wd->avB4[0]) + sizeof(wd->re_avBbar_AmB[0]));
        nByte += nIvalues*(sizeof(wd->IB[0]) + sizeof(wd->Ibulk[0]) + sizeof(wd->varIB[0]) + sizeof(wd->varIbulk[0]) +
                sizeof(wd->avBqabs_re[0]) + sizeof(wd->avBqabs_im[0]) + sizeof(wd->I_avAmB2[0]) +
                sizeof(wd->I_varA[0]) + sizeof(wd->I_varB[0]) + sizeof(wd->varI_avAmB2[0]) + sizeof(wd->varI_varA[0])
                + sizeof(wd->varI_varB[0])
        );

        for (i = 0; i < qhomenr; i++)
        {
            wd->avB     [i]      = cnul_d;
            wd->avBsq   [i]      = 0;
            wd->avB4    [i]      = 0.;
            wd->av_ReB_2[i]      = 0.;
            wd->av_ImB_2[i]      = 0.;
            wd->re_avBbar_AmB[i] = 0.;
        }
    }

    nByteInten = nByte;
    if (wr->bCalcForces)
    {
        snew(wd->dkI, nIvalues*nindA_prot);
        nByteForce += nIvalues*nindA_prot*sizeof(wd->dkI[0]);

        /* Now of size qhomenr*nindA_prot for gradients */
        snew(wd->dkAbar, qhomenr*nindA_prot);
        nByteForce += qhomenr*nindA_prot*sizeof(wd->dkAbar[0]);

        snew(wd->Orientational_Av, nindA_prot * nabs);
        nByteForce += nindA_prot * DIM * nIvalues;

        nByte += nByteForce;
    }

    wd->nByteAlloc = nByte;
    wt->wd         = wd;

    for (i = 0; i < (cr->nnodes - cr->npmenodes); i++)
    {
        if (cr->nodeid == i)
        {
            if (wr->bCalcForces)
            {
                fprintf(stderr,"Node %2d) Allocated %8g MB to store averages "
                        "(%4g MiB for intensities, %8g MB for forces) (scattering type %d).\n",
                        cr->nodeid, 1.0*wd->nByteAlloc/1024/1024,
                        1.0*nByteInten/1024/1024, 1.0*nByteForce/1024/1024, t);
            }
            else
            {
                fprintf(stderr,"Node %2d) Allocated %8g MiB to store averages (scattering type %d)\n",
                        cr->nodeid, 1.0*wd->nByteAlloc/1024/1024, t);
            }
        }
        if (PAR(cr))
        {
            gmx_barrier(cr);
        }
    }
}

void
done_waxs_datablock(t_waxsrec *wr)
{
    int i, t;
    t_waxs_datablock wd;

    for (t = 0; t < wr->nTypes; t++)
    {
        wd = wr->wt[t].wd;

        sfree(wd->A);
        sfree(wd->avA);
        sfree(wd->av_ReA_2);
        sfree(wd->av_ImA_2);
        sfree(wd->avAsq);
        sfree(wd->avA4);
        sfree(wd->D);
        sfree(wd->I);
        sfree(wd->varI);
        sfree(wd->IA);
        sfree(wd->varIA);
        sfree(wd->avAqabs_re);
        sfree(wd->avAqabs_im);
        if (! wr->bVacuum)
        {
            sfree(wd->IB);
            sfree(wd->Ibulk);
            sfree(wd->varIB);
            sfree(wd->varIbulk);
            sfree(wd->avBqabs_re);
            sfree(wd->avBqabs_im);
            sfree(wd->I_avAmB2);
            sfree(wd->I_varA);
            sfree(wd->I_varB);
            sfree(wd->I_scaleI0);
            sfree(wd->varI_avAmB2);
            sfree(wd->varI_varA);
            sfree(wd->varI_varB);
            sfree(wd->B);
            sfree(wd->avB);
            sfree(wd->av_ReB_2);
            sfree(wd->av_ImB_2);
            sfree(wd->avBsq);
            sfree(wd->avB4);
            sfree(wd->re_avBbar_AmB);
        }

        if (wr->bCalcForces)
        {
            sfree(wd->dkI);
            sfree(wd->dkAbar);
            sfree(wd->Orientational_Av);
        }

        sfree(wd);
        wr->wt[t].wd = NULL;
    }
}

static t_waxs_eavrmsd
init_waxs_eavrmsd()
{
    t_waxs_eavrmsd we;
    snew(we,1);
    we->x_avg=NULL;
    we->norm=0;
    we->scale=0;

    we->rmsd_now=0;
    we->rmsd_av=0;
    we->rmsd_avsq=0;
    we->sd_now=0;
    return we;
}

static void
done_waxs_eavrmsd(t_waxs_eavrmsd we)
{
    sfree(we->x_avg);
    sfree(we);
    we = NULL;
}

/* Changed to raw malloc becuase GROMACS doesn't like things being sfree'd
 * to give space then reallocated.

   Jochen: Strange, I don't think there is a problem with snew/sfree (Feb 2016)

 */
void waxs_alloc_globalavs(t_waxsrec *wr)
{
    int nq, t;
    t_waxs_datablock wd;

    for (t = 0; t < wr->nTypes; t++)
    {
        nq = wr->wt[t].qvecs->n;
        wd = wr->wt[t].wd;

        wd->Dglobal        = calloc(nq, sizeof(*(wd->Dglobal)));
        wd->avAglobal      = calloc(nq, sizeof(*(wd->avAglobal)));
        wd->avAsqglobal    = calloc(nq, sizeof(*(wd->avAsqglobal)));
        wd->avA4global     = calloc(nq, sizeof(*(wd->avA4global)));
        wd->av_ReA_2global = calloc(nq, sizeof(*(wd->av_ReA_2global)));
        wd->av_ImA_2global = calloc(nq, sizeof(*(wd->av_ImA_2global)));
        if (!wr->bVacuum)
        {
            wd->avBglobal      = calloc(nq, sizeof(*(wd->avBglobal)));
            wd->avBsqglobal    = calloc(nq, sizeof(*(wd->avBsqglobal)));
            wd->avB4global     = calloc(nq, sizeof(*(wd->avB4global)));
            wd->av_ReB_2global = calloc(nq, sizeof(*(wd->av_ReB_2global)));
            wd->av_ImB_2global = calloc(nq, sizeof(*(wd->av_ImB_2global)));
        }
    }
}

void waxs_free_globalavs(t_waxsrec *wr)
{
    int t;
    t_waxs_datablock wd;

    for (t = 0; t < wr->nTypes; t++)
    {
        wd = wr->wt[t].wd;

        free(wd->Dglobal);         wd->Dglobal        = NULL;
        free(wd->avAglobal);       wd->avAglobal      = NULL;
        free(wd->avAsqglobal);     wd->avAsqglobal    = NULL;
        free(wd->avA4global);      wd->avA4global     = NULL;
        free(wd->av_ReA_2global);  wd->av_ReA_2global = NULL;
        free(wd->av_ImA_2global);  wd->av_ImA_2global = NULL;
        if (!wr->bVacuum)
        {
            free(wd->avBglobal);      wd->avBglobal      = NULL;
            free(wd->avBsqglobal);    wd->avBsqglobal    = NULL;
            free(wd->avB4global);     wd->avB4global     = NULL;
            free(wd->av_ReB_2global); wd->av_ReB_2global = NULL;
            free(wd->av_ImB_2global); wd->av_ImB_2global = NULL;
        }
    }
}


static gmx_bool
readBooleanEnvVariable(const char* s)
{
    char *buf;

    if ((buf = getenv(s)) == NULL)
    {
        return FALSE;
    }
    if (!strcasecmp("FALSE", buf) || !strcasecmp("NO", buf) || !strcasecmp("0", buf))
    {
        printf("WAXS-MD: Found environment varialbe %s set to FALSE\n", s);
        return FALSE;
    }
    else if (!strcasecmp("TRUE", buf) || !strcasecmp("YES", buf) || !strcasecmp("1", buf))
    {
        printf("WAXS-MD: Found environment varialbe %s set to TRUE\n", s);
        return TRUE;
    }
    else
    {
        gmx_fatal(FARGS, "Found boolean environment varialbe %s, value should be false, \n"
                "true, yes, no, 1, or 0 (case-insensitive), but found value: \"%s\"\n", s, buf);
        return FALSE;
    }
}

static gmx_bool
readBooleanEnvVariableDefault(const char* s, gmx_bool theDefault, const char* comment)
{
    char *buf;
    gmx_bool value;

    if (getenv(s) == NULL)
    {
        return theDefault;
    }
    else
    {
        value = readBooleanEnvVariable(s);
        printf("WAXS-MD: Found boolean environment varialbe %s = %s", s, value ? "true" : "false");
        if (comment)
        {
            printf(" - %s%s\n", value ? "" : "NOT ", comment);
        }
        else
        {
            printf("\n");
        }
        return value;
    }
}

void read_env_variables(t_waxsrec *wr, t_commrec *cr)
{
    char *buf;

    /* Read environment variables */
    if (MASTER(cr))
    {
        wr->bDoNotCorrectForceByContrast =
            readBooleanEnvVariableDefault("GMX_WAXS_DONT_CORRECT_FORCE_BY_CONTRAST", FALSE,
                                          "Do not correct SAXS/SANS-derived forces by the contrast.");
        wr->bDoNotCorrectForceByNindep =
            readBooleanEnvVariableDefault("GMX_WAXS_DONT_CORRECT_FORCE_BY_NINDEP", FALSE,
                                          "Do not multiply SAXS/SANS-derived forces/potential by the number of independent data points.");

        if ((buf = getenv("GMX_WAXS_JMIN")) != NULL)
        {
            wr->Jmin = atoi(buf);
            fprintf(stderr, "\nWAXS-MD: Found environment variable GMX_WAXS_JMIN = %d\n", wr->Jmin);
        }
        if ((buf = getenv("GMX_WAXS_J_ALPHA")) != NULL)
        {
            wr->Jalpha = atof(buf);
            fprintf(stderr, "\nWAXS-MD: Found environment variable GMX_WAXS_J_ALAPHA = %g\n", wr->Jalpha);
        }
        wr->bHaveFittedTraj = readBooleanEnvVariable("GMX_WAXS_HAVE_FITTED_TRAJ");
        if  (wr->bHaveFittedTraj)
        {
            fprintf(stderr,"\nWAXS-MD:\nI am not fitting solute, and not shifting water into the box.\n"
                    " Probably, you are doing a rerun with a rotation-fitted trajectory\n\n");
        }

        wr->bRotFit = (readBooleanEnvVariable("GMX_WAXS_NOROTFIT") == FALSE);
        if  (wr->bRotFit)
        {
            fprintf(stderr,"\nWAXS-MD:\nWill rotate the box at each waxs calculation."
                    "This is intended to test the action of fitted forces...\n\n");
        }

        wr->bMassWeightedFit = (readBooleanEnvVariableDefault("GMX_WAXS_NONWEIGHTED_FIT", FALSE, "doing non-mass-weighted fit") == FALSE);

        wr->bDampenForces = readBooleanEnvVariable("GMX_WAXS_FORCEDAMPING");
        if (wr->bDampenForces)
        {
            fprintf(stderr,"WAXS-MD: Force damping is switched on.\nWill calculate and report exponential average data.\n");
        }
        if (getenv("GMX_WAXS_DAMPMIN") != NULL)
        {
            wr->damp_min = atof(getenv("GMX_WAXS_DAMPMIN"));
            fprintf(stderr, "\nWAXS-MD: Found environment variable GMX_WAXS_DAMPMIN = %f\n", wr->calcWAXS_begin);
        }
        if (getenv("GMX_WAXS_DAMPMAX") != NULL)
        {
            wr->damp_max = atof(getenv("GMX_WAXS_DAMPMAX"));
            fprintf(stderr, "\nWAXS-MD: Found environment variable GMX_WAXS_DAMPMAX = %f\n", wr->calcWAXS_end);
        }

        wr->x_ref_file = getenv("GMX_WAXS_FIT_REFFILE");
        wr->bHaveRefFile = ( wr->x_ref_file != NULL );
        if ( wr->bHaveRefFile )
        {
            fprintf(stderr,"Using coordinates from %s as WAXS-MD fit coordinates.\n", wr->x_ref_file);
        }

        wr->bPrintForces = readBooleanEnvVariable("GMX_WAXS_PRINTFORCES") && wr->bCalcForces;
        if  (wr->bPrintForces)
        {
            fprintf(stderr,"\nWAXS-MD:\n Will print coordinates and forces on WAXS-solute atoms."
                    "This may take up a lot of space in your storage!\n\n");
        }

        if ((buf = getenv("GMX_WAXS_VERBOSE")) != NULL)
        {
            wr->debugLvl = atoi(buf);
        }
        fprintf(stderr,"\nWAXS-MD: WAXS verbosity level = %d\n\n", wr->debugLvl);

        if ((buf = getenv("GMX_WAXS_BEGIN")) != NULL)
        {
            wr->calcWAXS_begin = atof(buf);
            fprintf(stderr, "\nWAXS-MD: Found environment variable GMX_WAXS_BEGIN = %f\n", wr->calcWAXS_begin);
        }
        if ((buf = getenv("GMX_WAXS_END")) != NULL)
        {
            wr->calcWAXS_end = atof(buf);
            fprintf(stderr, "\nWAXS-MD: Found environment variable GMX_WAXS_END = %f\n", wr->calcWAXS_end);
        }
        if ((buf = getenv("GMX_WAXS_WEIGHT_TOLERANCE")) != NULL)
        {
            wr->stateWeightTolerance = atof(buf);
            fprintf(stderr, "\nWAXS-MD: Found environment variable GMX_WAXS_WEIGHT_TOLERANCE = %g\n", wr->stateWeightTolerance);
        }
        wr->bGridDensity = readBooleanEnvVariableDefault("GMX_WAXS_GRID_DENSITY", FALSE, "computing electron density on a grid.");

        if ((buf = getenv("GMX_WAXS_GRID_DENSITY_MODE")) != NULL)
        {
            wr->gridDensityMode = atoi(buf);
            fprintf(stderr, "\nWAXS-MD: Found environment variable GMX_WAXS_GRID_DENSITY_MODE = %d\n", wr->gridDensityMode);
        }

        if ((buf = getenv("BACKBONE_DEUTERATED_PROB")) != NULL)
        {
            wr->backboneDeuterProb = atof(buf);
            fprintf(stderr, "\nWAXS-MD: Found environment variable BACKBONE_DEUTERATED_PROB, backbone deuterated with probability %g\n",
                    wr->backboneDeuterProb);
            if (wr->backboneDeuterProb < 0 || wr->backboneDeuterProb > 1)
            {
                gmx_fatal(FARGS, "Environment variable BACKBONE_DEUTERATED_PROB must be between 0 and 1 (found %g)\n", wr->backboneDeuterProb);
            }
        }
        wr->bHaveWholeSolute       = readBooleanEnvVariableDefault("GMX_WAXS_SOLUTE_IS_WHOLE",        FALSE, "Will not make solute whole.");
        wr->bStochasticDeuteration = readBooleanEnvVariableDefault("GMX_WAXS_STOCHASTIC_DEUTERATION", FALSE, "");
        wr->bRemovePosresWithTau   = readBooleanEnvVariableDefault("GMX_WAXS_TURN_DOWN_POSRES",       TRUE,  "");
        fprintf(stderr, "\n");
    }
    if (PAR(cr))
    {
        gmx_bcast(sizeof(int),      &wr->Jmin,                         cr);
        gmx_bcast(sizeof(real),     &wr->Jalpha,                       cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bHaveFittedTraj,              cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bDoNotCorrectForceByContrast, cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bDoNotCorrectForceByNindep,   cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bRotFit,                      cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bDampenForces,                cr);
        gmx_bcast(sizeof(real),     &wr->damp_min,                     cr);
        gmx_bcast(sizeof(real),     &wr->damp_max,                     cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bPrintForces,                 cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bScaleI0,                     cr);
        gmx_bcast(sizeof(int),      &wr->debugLvl,                     cr);
        gmx_bcast(sizeof(double),   &wr->calcWAXS_begin,               cr);
        gmx_bcast(sizeof(double),   &wr->calcWAXS_end,                 cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bGridDensity,                 cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bHaveWholeSolute,             cr);
        gmx_bcast(sizeof(real),     &wr->backboneDeuterProb,           cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bStochasticDeuteration,       cr);
        gmx_bcast(sizeof(gmx_bool), &wr->bRemovePosresWithTau,         cr);
        gmx_bcast(sizeof(double),   &wr->stateWeightTolerance,         cr);
    }
}


void
init_waxs_md( t_waxsrec *wr,
              t_commrec *cr, t_inputrec *ir,
              gmx_mtop_t *top_global,
              const output_env_t oenv, double t0,
              const char *fntpsSolv, const char *fnxtcSolv,const char *fnOut,
              const char *fnScatt,
              t_state *state_local, gmx_bool bRerunMD, gmx_bool bWaterOptSet,
              gmx_bool bReadI)
{
    real            momentum, lambda;
    int             i, j, t, nsltype;
    int             qhomenr, qstart;
    rvec           *q;
    real           *ft_reEnv, *ft_imEnv;
    t_waxsrecType  *wt;
    t_waxs_solvent  ws = NULL;
    gmx_bool        bStochDeuterBackup;

    wr->bDoingMD    = (ir != NULL);

    /* As soon as we have a targt SAXS curve, we compute the SAXS potential */
    wr->bCalcPot    = bReadI;

    /* Compute forces only if waxs_tau is positive, we don't do a rerun, and we have
       a target SAXS curve */
    wr->bCalcForces = (!bRerunMD && (ir->waxs_tau > 0) && bReadI);

#ifdef RERUN_CalcForces
    // JUST FOR TEST - use only in case you know what you are doing!
    bRerunMD = FALSE ;
    wr->bCalcForces = TRUE; //TEST!!!
#endif

    if (ir == NULL)
    {
        gmx_fatal(FARGS, "Error in init_waxs_md(), input record not initialized\n");
    }
    if (MASTER(cr) && (!bRerunMD && ir->waxs_tau <= 0))
    {
        fprintf(stderr, "WAXS-MD: Found waxs-tau <= 0 : will NOT calculate forces.\n");
    }
    if (MASTER(cr))
    {
        fprintf(stderr, "WAXS-MD: Using SIMD level: %s\n", waxs_simd_string());
    }
    if (bReadI && !bRerunMD && !(ir->waxs_tau > 0))
    {
        gmx_fatal(FARGS, "You are running an MD simulation with a target SAXS curve, but waxs-tau is <=0 (%g)\n"
                  "For a SAXS-driven MD simulation, you should use a positive waxs-tau (e.g. 200ps)\n", ir->waxs_tau);
    }

    if (MASTER(cr))
    {
        fprintf(stderr, "WAXS-MD: We do %scalculate the potential\n",wr->bCalcPot?"":"NOT ");
        fprintf(stderr, "WAXS-MD: We do %scalculate forces.\n", wr->bCalcForces ? "" : "NOT ");
        printf("WAXS-MD: Will compute WAXS patterns on %d nodes\n", cr->nnodes - cr->npmenodes);
    }
    if (cr->nodeid >= (cr->nnodes - cr->npmenodes))
    {
        gmx_fatal(FARGS, "Inconsnsitency in init_waxs_md, nodeid = %d, but expected only PP %d nodes (nnodes = %d, npmenodes = %d)\n",
                cr->nodeid, cr->nnodes - cr->npmenodes, cr->nnodes, cr->npmenodes);
    }

    /* For the moment has to be implemented later */
    if (gmx_omp_nthreads_get(emntDefault) > 1)
    {
        printf("WAXS-MD: NOTE: OpenMP paralellization is supported but not fully optimized yet!\n");
    }

    /* We compute the WAXS forces after calculating the virial in do_force(). Therefore,
       it is anyway not added to the virial. No need to set fr->bF_NoVirSum=TRUE
       Note: If we decide to compute WAXS forces before computing the virial, we must
       add the WAXS force to f_novirsum and set fr->bF_NoVirSum=TRUE NOT here but in
       forcerec.c. Otherwise a segfault would occur in do_force().
     */
    /* fr->bF_NoVirSum = fr->bF_NoVirSum || wr->bCalcForces; */

    /* Store a pointer on the local state - required for dd_collect_state() within do_force */
    if (state_local)
    {
        wr->local_state = state_local;
    }

    /* Copy values to waxsrec to make the rest source-independent. */
    if (wr->bDoingMD)
    {
        copy_ir2waxsrec( wr, ir, cr );
    }
    if (MASTER(cr))
    {
        /* Read water.  */
        if (! wr->bVacuum)
        {
            snew(ws, 1);
            wr->waxs_solv = ws;
            read_waxs_solvent(oenv, ws, fntpsSolv, fnxtcSolv, TRUE, wr);
            if (wr->bDoingNeutron && !ws->bHaveNeutron)
            {
                gmx_fatal(FARGS, "Doing neutron scattering, but found no neutron scattering length in solvent tpr file.\n"
                          "Use scatt-coupl = Neutron in your solvent mdp file.");
            }
        }
        else if (wr->bDoingMD && bWaterOptSet)
        {
            gmx_fatal(FARGS,"No waxs_solvent was specified in main simulation, but options\n"
                    " to provide exclued solvent was passed to mdrun\n");
        }
    }

    /* Broadcast values that were changed on MASTER after reading waxs solvent.
     * Required for all nodes to keep in step when determining bDoingSolvent */
    if (PAR(cr))
    {
        gmx_bcast(sizeof(int),&wr->nfrsolvent, cr);
    }

    /* Keep initial simulation time */
    wr->simtime0 = t0;

    /* read scattering curve to which we couple and broadcast it */
    if (wr->bCalcPot)
    {
        read_waxs_curves(fnScatt, wr, cr);
    }

    /* Redirect the atomtypes in waxs-solv to existing types and allocate waxsrec to receive these types. */
    if (MASTER(cr) && ws)
    {
        redirect_md_solventtypes(ws->mtop, &top_global->scattTypes);
    }

    /* \hbar \omega \lambda = hc = 1239.842 eV * nm */
    momentum  = (2. * 1000.0 * M_PI * wr->xray_energy) / CONSTANT_HC ;
    lambda    = CONSTANT_HC / (1000.0 * wr->xray_energy);
    wr->qbeam = momentum;
    if (MASTER(cr))
    {
        printf("\nWAXS-MD: q[beam] = %g 1/nm\n",  wr->qbeam);
    }

    /* init envelope */
    if (getenv("GMX_ENVELOPE_FILE") == NULL)
    {
        gmx_fatal(FARGS, "Environment varialbe GMX_ENVELOPE_FILE not found. Please define it to the file name of\n"
                  "the evelope (such as envelope.py), written by g_genenv. In addition, define GMX_WAXS_FIT_REFFILE\n"
                  "to the reference file written by g_genenv.\n");
    }
    for (t = 0; t < wr->nTypes; t++)
    {
        wr->wt[t].envelope = gmx_envelope_init_md(-1, cr, MASTER(cr));
    }

    /* Read environment variables into waxsrec */
    read_env_variables(wr, cr);

    /* setup q */
    /* boolean == TRUE means write debug output */
    for (t = 0; t < wr->nTypes; t++)
    {
        if (wr->wt[t].nq <= 0)
        {
            gmx_fatal(FARGS, "Number of q vales must be >= 1. Fixs waxs-nq mdp option.\n");
        }
        wr->wt[t].qvecs = gen_qvecs_map(wr->wt[t].minq, wr->wt[t].maxq, wr->wt[t].nq, wr->J,
                                        t == 0 && wr->debugLvl > 1,
                                        cr,
                                        wr->ewaxsaniso, wr->qbeam,
                                        wr->wt[t].envelope, wr->Jmin, wr->Jalpha, MASTER(cr) && t == 0);
    }

    if (wr->nfrsolvent > 0)
    {
#ifdef GMX_GPU
        /* Unit FT is only calculated once. It is therefore also only copied once to GPU */
        if (wr->bUseGPU == TRUE)
        {
            push_unitFT_to_GPU(wr);
        }
#endif
    }

    if (MASTER(cr))
    {
        /* Init output stuff */
        if (fnOut)
        {
            init_waxs_output(wr, fnOut, oenv);
        }
    }

    if (MASTER(cr))
    {
        /* Memory for atomc coordinates x, so we can overwrite the coordinates when makeing the solute whole, etc. */
        snew(wr->x, top_global->natoms);

        /* Short cut array with number of electrons per atom (size top_global->natoms) */
        wr->nElectrons = make_nElecList(top_global);

        prep_md_indices( &wr->indA_prot, &wr->nindA_prot, &wr->indA_sol, &wr->nindA_sol, top_global,
                TRUE, !wr->bVacuum);
        if (ws)
        {
            prep_md_indices( NULL, NULL, &wr->indB_sol, &wr->nindB_sol, ws->mtop, FALSE, TRUE );
        }

        /* Pick cmtypes and nsltypes from mtop (main simulation and solvent) into continuous array in waxsrec */
        wr->cmtypeList = mtop2scattTypeList(top_global, FALSE);
        if (wr->bDoingNeutron)
        {
            wr->nsltypeList = mtop2scattTypeList(top_global, TRUE);
        }
        if (ws)
        {
            ws->cmtypeList = mtop2scattTypeList(ws->mtop, FALSE);
            if (wr->bDoingNeutron)
            {
                ws->nsltypeList = mtop2scattTypeList(ws->mtop, TRUE);
            }
        }

        /* Write protein atoms into beginning of waxsrec->indexA. These will not change
           throughout the calculations, but solvation shell atoms will be added to indexA. */
        snew(wr->indexA,  wr->nindA_prot);
        wr->isizeA        = wr->nindA_prot;
        wr->indexA_nalloc = wr->nindA_prot;
        for (i = 0; i < wr->nindA_prot; i++)
        {
            wr->indexA[i] = wr->indA_prot[i];
        }
        /* Array for atomic coordinates inside the envelope, update every each step */
        snew(wr->atomEnvelope_coord_A,    wr->nindA_prot);
        for (t = 0; t < wr->nTypes; t++)
        {
            snew(wr->wt[t].atomEnvelope_scatType_A, wr->nindA_prot);
        }

        /* Init PBC atom. Number-wise center if not defined. If defined, -1 for mdrun-interal numbering */
        if ( wr->npbcatom == 0 )
        {
            gmx_fatal(FARGS,"Something has gone wrong - pbcatom information is missing!\n");
        }
        /* Get # of molecules in Solute - if ==1, we can make the solute whole using gmx internal routines */
        wr->nSoluteMols = nSoluteMolecules(top_global);

        /* Check for the type of pbcatom fitting. */
        if ( wr->pbcatoms[0] == -1 )
        {
            wr->pbcatoms[0] = (wr->nindA_prot+1)/2-1;
        }
        else if ( wr->pbcatoms[0] < -1 )
        {
            if (wr->nSoluteMols > 1)
            {
                gmx_fatal(FARGS, "Found waxs-pbc atom = %d, meaning that the protein is made whole using\n"
                          "the distances between number-wise atomic neighbors. However, this is only possible\n"
                          "when there is only 1 molecule in the solute group (found %d)\n",
                          wr->pbcatoms[0]+1, wr->nSoluteMols);
            }
        }
        fprintf(stderr, "Using %d PBC atoms for WAXS Solute. First atom index (global): %d\n",wr->npbcatom,wr->pbcatoms[0]);
        /* fprintf(stderr,"Using PBC atom id (mdrun-internal) for WAXS solute: %d\n",wr->pbcatom); */

        /* Rotational-fitting requires the solute indices to be defined. */
        if (wr->bRotFit)
        {
            snew(wr->x_ref, top_global->natoms);
            snew(wr->w_fit, top_global->natoms);
            for (i=0; i<top_global->natoms;i++)
            {
                clear_rvec(wr->x_ref[i]);
            }
            prep_fitting_weights(wr, &wr->ind_RotFit, &wr->nind_RotFit, wr->w_fit, top_global);

            if (wr->bHaveRefFile)
            {
                read_fit_reference(wr->x_ref_file, wr->x_ref, top_global->natoms, wr->indA_prot,
                        wr->nindA_prot, wr->ind_RotFit, wr->nind_RotFit);

                reset_x(wr->nind_RotFit, wr->ind_RotFit, top_global->natoms, NULL, wr->x_ref, wr->w_fit);
                fprintf(stderr,"Loaded reference file into x_ref.\n");
            }
            else
            {
                gmx_fatal(FARGS, "Environtment variable GMX_WAXS_FIT_REFFILE not defined. Define "
                        "it with the fit reference coordinate file written by g_genenv\n");
            }
        }

        wr->rng = gmx_rng_init(gmx_rng_make_seed());

        if (!wr->bVacuum)
        {
            /* Get and distribute density of the pure solvent box.
               We use the density compute inside the envelope to specify the density correction. This way,
               the density entering for B(q) is (hopefully) *exactly* the value given by waxs-solvdens. */
            get_solvent_density(wr);
            wr->solElecDensAv_SysB = wr->waxs_solv->avDropletDensity;
        }
    }

    if (PAR(cr))
    {
        gmx_bcast(sizeof(real), &wr->solElecDensAv_SysB, cr);
    }

    for (t = 0; t<wr->nTypes; t++)
    {
        /* allocate memory for cumulative averages and init wd->norm (for averaging) to zero. */
        if (PAR(cr))
        {
            gmx_bcast(sizeof(int),&wr->nindA_prot,cr);
        }
        alloc_waxs_datablock(cr, wr, wr->nindA_prot, t);

        wt = &wr->wt[t];
        if (wt->type == escatterXRAY)
        {
            /* build table of atomic form factors: 1st index: cmtype. 2nd index: absolute value of q. */
            compute_atomic_form_factor_table( &top_global->scattTypes, wt, cr);
        }
        else if (wt->type == escatterNEUTRON)
        {
            /* build table of NSLs, taking deuterium concentration into account */
            fill_neutron_scatt_len_table( &top_global->scattTypes, wt, wr->backboneDeuterProb);
        }

        /* Store qstart and qhomenr on Master. Requires both alloc_datablock and gen_qvecs*/
        gen_qvecs_accounting(wr, cr, cr->nnodes-cr->npmenodes, t);
    }

    if (wr->bCalcForces)
    {
        if (PAR(cr))
        {
            if (wr->indA_prot == NULL)
            {
                snew(wr->indA_prot, wr->nindA_prot);
            }
            gmx_bcast(wr->nindA_prot*sizeof(int), wr->indA_prot,cr);
        }
        snew(wr->fLast, wr->nindA_prot);
        for (t = 0; t<wr->nTypes; t++)
        {
            snew(wr->wt[t].fLast, wr->nindA_prot);
        }
    }

    /* Count total number of electrons in A system */
    if (MASTER(cr))
    {
        for (i = 0; i < wr->nindA_prot; i++)
        {
            wr->nElecTotA += wr->nElectrons[wr->indA_prot[i]];
        }
        wr->nElecProtA = wr->nElecTotA;
        for (i = 0; i < wr->nindA_sol; i++)
        {
            wr->nElecTotA += wr->nElectrons[wr->indA_sol[i]];
        }
        printf("There are %g electrons in total in solute + solvent (%g per atom on average)\n",
                wr->nElecTotA, wr->nElecTotA/(wr->nindA_prot+wr->nindA_sol));
    }

    /* Save sum of scattering lengths of solute and solvent (# of electrons or # of NSLs).
       This will be used to estimate the contrast of the solute */
    bStochDeuterBackup         = wr->bStochasticDeuteration;
    wr->bStochasticDeuteration = FALSE;
    if (MASTER(cr) && ws)
    {
        snew(ws->avScattLenDens, wr->nTypes);
    }
    for (t = 0; t < wr->nTypes; t++)
    {
        if (MASTER(cr))
        {
            wt = &wr->wt[t];

            if (wt->type == escatterXRAY)
            {
                wt->soluteSumOfScattLengths = wr->nElecProtA;
                if (ws)
                {
                    ws->avScattLenDens[t] = ws->nelec * ws->avInvVol;
                }
            }
            else
            {
                /* Sum NSLs of solute without stochastic deuteration */
                wt->soluteSumOfScattLengths = 0;
                for (i = 0; i < wr->nindA_prot; i++)
                {
                    j = wr->indA_prot[i];
                    nsltype                      = get_nsl_type(wr, t, wr->nsltypeList[j]);
                    wt->soluteSumOfScattLengths += wt->nsl_table[nsltype];
                }
                if (ws)
                {
                    double sum = 0;
                    for (i = 0; i < wr->nindB_sol; i++)
                    {
                        j = wr->indB_sol[i];
                        nsltype  = get_nsl_type(wr, t, ws->nsltypeList[j]);
                        sum     += wt->nsl_table[nsltype];
                    }
                    ws->avScattLenDens[t] = sum * ws->avInvVol;
                }
            }
        }
    }
    wr->bStochasticDeuteration = bStochDeuterBackup;
    if (MASTER(cr))
    {
        printf("\nSum of scattering lengths of solute:\n");
        for (t = 0; t < wr->nTypes; t++)
        {
            wt = &wr->wt[t];
            printf("\tType %d (%s) sum = %10g %s\n", t, wt->saxssansStr, wt->soluteSumOfScattLengths,
                   (wt->type == escatterXRAY) ? "electrons" : "NSLs");
        }
        printf("\n");
        if (ws)
        {
            printf("\nDensities of pure-solvent system:\n");
            for (t = 0; t < wr->nTypes; t++)
            {
                wt = &wr->wt[t];
                printf("\tType %d (%s)  density = %10g %s/nm3\n", t, wt->saxssansStr,
                       ws->avScattLenDens[t],
                       (wt->type == escatterXRAY) ? "electrons" : "NSLs");
            }
            printf("\n");
        }
    }

    /* Get scattering intensity of the pure buffer. Used to add back oversubtracted buffer */
    if (MASTER(cr))
    {
        printf("WAXS-MD: %s back oversubtracted buffer intensity\n",
                (wr->bCorrectBuffer && !wr->bVacuum) ? "Adding" : "Not adding");
    }
    if (wr->bCorrectBuffer && !wr->bVacuum)
    {
        if (wr->bDoingNeutron)
        {
            gmx_fatal(FARGS, "For neutron scattering, only the simple buffer subtraction scheme "
                      "I_sam - I_buf is supported so far.\nChange mdp option waxs-correct-buffer to no.");
        }
        pure_solvent_scattering(wr, cr, fnxtcSolv, top_global);
    }

    if (wr->bDampenForces)
    {
        wr->wrmsd = init_waxs_eavrmsd();
        if (MASTER(cr))
        {
            /* Duplicate for easy access */
            wr->wrmsd->scale = wr->scale;
            /* Additionally store the exponentially averaged structure for RMSD */
            snew(wr->wrmsd->x_avg, top_global->natoms);
            for (i = 0; i < top_global->natoms; i++)
            {
                clear_rvec(wr->wrmsd->x_avg[i]);
            }
        }
    }

    /* Consitency checks for ensembles */
    if ( !(wr->ewaxs_ensemble_type == ewaxsEnsembleNone || wr->ewaxs_ensemble_type == ewaxsEnsemble_BayesianOneRefined))
    {
        gmx_fatal(FARGS, "Only ensemble types 'none' or 'bayesian-one-refined' supported\n");
    }
    /* if ( wr->ewaxs_ensemble_type == ewaxsEnsemble_BayesianOneRefined && !wr->bBayesianMD) */
    /* { */
    /*     gmx_fatal(FARGS, "Ensemble refinement so far only with Bayesian MD. wr->bBayesianMD == %d, wr->ewaxs_ensemble_type = %d\n", */
    /*               wr->bBayesianMD, wr->ewaxs_ensemble_type ); */
    /* } */
    if (WAXS_ENSEMBLE(wr) && wr->ewaxs_ensemble_type == ewaxsEnsembleNone)
    {
        gmx_fatal(FARGS, "Found >1 number of states in enbemble (%d), but ewaxs_ensemble_type = none\n", wr->ensemble_nstates);
    }
    if (WAXS_ENSEMBLE(wr))
    {
        /* For ensemble refinement, read the intensities of the other fixed states */
        init_ensemble_stuff(wr, cr);
    }

    if (ir->cutoff_scheme == ecutsVERLET)
    {
#ifdef GMX_GPU
        /* Here all data is copied to GPU that will remain constantly there. It initializes the GPU for all scattering types. */
        if (wr->bUseGPU)
        {
            init_gpudata_type(wr);
        }
        fprintf(stderr, "Initiated data for SAXS/SANS calculations on the GPU.\n");
#endif
    }

    /* Init computing time measurements */
    if (MASTER(cr))
    {
        wr->compTime = waxsTimeInit();
    }
}

void done_waxs_md(t_waxsrec *wr)
{
    int t;

    for (t = 0; t<wr->nTypes; t++)
    {
        free_qvecs_accounting(wr->wt[t].wd);
        if (wr->wt[t].wd)
        {
            done_waxs_datablock(wr);
            if (wr->waxs_solv)
                done_waxs_solvent(wr->waxs_solv);
        }
    }
    if (wr->wrmsd)
        done_waxs_eavrmsd(wr->wrmsd);
}

/* Update cumulative averages: new = fac1*sum + fac2*new */
real r_accum_avg( real sum, real new, real fac1, real fac2 )
{
    real tmp = fac1*sum + fac2*new;
    return tmp;
}
double d_accum_avg( double sum, double new, double fac1, double fac2 )
{
    double tmp = fac1*sum + fac2*new;
    return tmp;
}
t_complex c_accum_avg( t_complex sum, t_complex new, real fac1, real fac2 )
{
    t_complex tmp = cadd( rcmul(fac1,sum), rcmul(fac2,new) );
    return tmp;
}
t_complex_d cd_accum_avg( t_complex_d sum, t_complex_d new, double fac1, double fac2 )
{
    t_complex_d tmp = cadd_d( rcmul_d(fac1,sum), rcmul_d(fac2,new) );
    return tmp;
}
void cvec_accum_avg( cvec *sum, cvec new, real fac1, real fac2 )
{
    cvec tmp;
    int d;
    for (d=0 ; d<DIM; d++)
    {
        tmp [d] = c_accum_avg(*sum[d],new[d],fac1,fac2);
        *sum[d] = tmp[d];
    }
}

/* Update the ensemble average position of x_avg, and its associated RMSD measures.
 * This is to predict conformational change. */
static void
update_waxs_eavrmsds(t_waxsrec *wr )
{
    t_waxs_eavrmsd we = wr->wrmsd;
    int ntot = wr->nind_RotFit;
    atom_id *ind = wr->ind_RotFit;
    double fac1, fac2;
    real rmsd, q;
    rvec xtmp;
    int i, j;

    //Get RMSD to past ensemble.
    rmsd = calc_similar_ind(FALSE,wr->nind_RotFit,wr->ind_RotFit,
                            wr->w_fit, wr->x, we->x_avg);

    //normA is shared between the ensemlbe and the RMSD averages,
    //and should also be identical to normA in wd.
    we->norm = 1.0 + we->scale*we->norm;
    fac1     = 1.0*(we->norm - 1)/we->norm;
    fac2     = 1.0/we->norm;

    //Update RMSD averages
    we->rmsd_now  = rmsd;
    we->rmsd_av   = r_accum_avg(we->rmsd_av, rmsd, fac1, fac2);
    we->rmsd_avsq = r_accum_avg(we->rmsd_avsq, rmsd*rmsd, fac1, fac2);

    q          = we->rmsd_avsq-we->rmsd_av*we->rmsd_av;
    we->sd_now = (q < GMX_FLOAT_EPS) ? 0 : sqrt(q);
    //Remove float rounding errors, and initial step problems.

    //update ensemble avg conformation.
    for (i=0; i<ntot; i++)
    {
        copy_rvec(we->x_avg[ind[i]],xtmp);
        for (j=0; j<DIM; j++)
        {
            xtmp[j] = r_accum_avg(we->x_avg[ind[i]][j], xtmp[j], fac1, fac2);
        }
        copy_rvec(xtmp,we->x_avg[ind[i]]);
    }

    fprintf(stderr,"WAXS RMSD ensembles and averages updated.\n");
}

/* Update the average values, or calculate the average values.
   Initialized to zero in alloc_waxs_datablock() */
static void
update_AB_averages( t_waxsrec* wr )
{
    t_waxs_datablock   wd;
    t_waxsrecType     *wt;
    double             fac1A, fac2A, dtmp, fac1B= 0., fac2B = 0.;
    t_complex_d        tmp_cd,  A_avB_tmp;
    double             tmpc1, tmpc2;
    t_complex          tmp_c;
    real               tmp;
    int                i, j, jj, k, d, l , n , p , t, qhomenr, nprot = wr->nindA_prot, nabs, qstart ;
    dvec               tmp_dkAbar_A_avB;

#ifdef RERUN_CalcForces
    gmx_bool bCalcForces = TRUE;
#endif
    waxs_debug("Entering update_AB_averages()");

    for (t = 0; t < wr->nTypes; t++)
    {
        wt      = &wr->wt[t];
        wd      = wt->wd;
        qhomenr = wt->qvecs->qhomenr;
        qstart  = wt->qvecs->qstart;
        nabs    = wt->qvecs->nabs;

        /* Get fac1 and fac2 for running weighted average
           <X[i+1]> = fac1*<X[i]> + fac2*x[i+1] =
                    =  (N[i+1]-1)/N[i+1]) * <X[i]> + 1/N[i+1]*x[i+1]
           This is valid for either exponential or equally-weighted averaging.
         */
        wd->normA = 1.0 + wr->scale*wd->normA;
        fac1A     = 1.0*(wd->normA - 1)/wd->normA;
        fac2A     = 1.0/wd->normA;

        /* For the B system (pure solvent), always use a non-weighted average (scale = 1) */
        if (wr->bDoingSolvent)
        {
            wd->normB = 1.0 + wd->normB;
            fac1B     = 1.0*(wd->normB - 1)/wd->normB;
            fac2B     = 1.0/wd->normB;
        }

        /* Note on precision: A[i], B[i] and dkAbar[k] are real !
         *                    All averages are double !
         */

        /* For each local q vector */
        for (i = 0; i < qhomenr; i++)
        {
            tmp_cd.re       = wd->A[i].re;
            tmp_cd.im       = wd->A[i].im;
            wd->avA  [i]    = cd_accum_avg( wd->avA    [i], tmp_cd,            fac1A, fac2A );

            dtmp            = cabs2_d(wd->A[i]);
            wd->avAsq[i]    = d_accum_avg( wd->avAsq   [i], dtmp,              fac1A, fac2A );
            wd->avA4[i]     = d_accum_avg( wd->avA4    [i], dtmp*dtmp,         fac1A, fac2A );
            wd->av_ReA_2[i] = d_accum_avg( wd->av_ReA_2[i], dsqr(wd->A[i].re), fac1A, fac2A );
            wd->av_ImA_2[i] = d_accum_avg( wd->av_ImA_2[i], dsqr(wd->A[i].im), fac1A, fac2A );

            if (wr->bDoingSolvent)
            {

                tmp_cd.re       = wd->B[i].re;
                tmp_cd.im       = wd->B[i].im;

                /* We could pick the avB from the GPU if available.
                 * But due to the low calculation but expensive data transfer cost, avB is recalculated here. */
                wd->avB  [i]    = cd_accum_avg( wd->avB  [i], tmp_cd,              fac1B, fac2B );
                dtmp            = cabs2_d(wd->B[i]);
                wd->avBsq[i]    = d_accum_avg( wd->avBsq[i], dtmp,                 fac1B, fac2B );
                wd->avB4 [i]    = d_accum_avg( wd->avB4 [i], dtmp*dtmp,            fac1B, fac2B );
                wd->av_ReB_2[i] = d_accum_avg( wd->av_ReB_2[i], dsqr(wd->B[i].re), fac1B, fac2B );
                wd->av_ImB_2[i] = d_accum_avg( wd->av_ImB_2[i], dsqr(wd->B[i].im), fac1B, fac2B );
            }
        }

        /* If we use the GPU, the orientiational average for grad D(q) is already done on the GPU */
        if (wr->bCalcForces && wr->bUseGPU == FALSE)
        {
            for (p = 0; p < nprot; p++)
            {
                for (n = 0 ; n < nabs ; n++)
                {
                    /* 1st summand */
                    clear_dvec(tmp_dkAbar_A_avB);

                    /* loop over q-vec which belong to this absolute value |q_i| */
                    for (j = wt->qvecs->ind[n] ; j < wt->qvecs->ind[n+1] ; j++)
                    {
                        if (j >= qstart && j < (qstart+qhomenr))
                        {
                            jj = j - qstart;              /* index of A and avB         */

                            /* index of derivatives atoms */
                            k = p * qhomenr + jj ;

                            /* Do product of A.grad[k]*.(A(q) - B_av)  (before time average)
                             *
                             * Calculate the difference A[k] - B_av
                             * Note that after a certain number of steps <B> is not changing anymore, therefore, it can be taken as a constant
                             */
                            A_avB_tmp.re = wd->A[jj].re - wd->avB[jj].re;
                            A_avB_tmp.im = wd->A[jj].im - wd->avB[jj].im;

                            if (wr->bScaleI0)
                            {
                                A_avB_tmp.re += wd->avAcorr[jj].re - wd->avB[jj].re;
                                A_avB_tmp.im += wd->avAcorr[jj].im - wd->avB[jj].im;
                            }

                            /* grad[k](A*(q)) . (A(q) - B_av)  (before time average). Only the real part remains: */
                            tmpc1 = +wd->dkAbar[k].re * A_avB_tmp.re
                                    -wd->dkAbar[k].im * A_avB_tmp.im;

                            for (d = 0 ; d < DIM ; d++)
                            {
                                /* Multiply with \vec{q} */
                                tmp_dkAbar_A_avB[d] += tmpc1 * wt->qvecs->q[j][d] ;
                            }
                        }
                    }

                    l = p * nabs + n;
                    for (d = 0 ; d < DIM ; d++)
                    {
                        /* Temporal running average */
                        wd->Orientational_Av[l][d] = d_accum_avg(wd->Orientational_Av[l][d], tmp_dkAbar_A_avB[d], fac1A, fac2A);
                    }
                }
            }
        }
    }

    waxs_debug("Leaving update_AB_averages()");
}


/* Computes I(q) and grad[k] I(q) (in parallel) */
void calculate_I_dkI (t_waxsrec *wr, t_commrec *cr)
{
    t_spherical_map *qvecs;
    int               nabs, qstart, qhomenr,qend, left, right,nprot,p, nIvalues, stepA, stepB;
    t_waxs_datablock  wd;
    t_waxsrecType    *wt;
    int               i, j, jj, *ind, l, k, d, m, N, nFramesSolvent, t;
    t_complex_d       tmpc, this_avA;
    double            sum, sum2, variance, dkItmp, tmp, Bprefact;

    waxs_debug("Entering calculate_I_dkI()");

    for (t = 0; t < wr->nTypes; t++)
    {
        wt      = &wr->wt[t];
        qvecs   = wt->qvecs;
        wd      = wt->wd;

        ind     = qvecs->ind;
        qstart  = qvecs->qstart;
        qhomenr = qvecs->qhomenr;
        nabs    = qvecs->nabs;
        qend    = qstart+qhomenr;
        nprot   = wr->nindA_prot;
        nIvalues= wd->nIvalues;

        if (wr->tau < 0.)
        {
            /* Prefactor in front of <B^2>-<B>^2 to get an unbiased estimator. Park et al., eq. 31 */
            nFramesSolvent = ( wr->waxsStep < wr->nfrsolvent ) ? (wr->waxsStep+1) :  wr->nfrsolvent;
            Bprefact       = (nFramesSolvent > 1) ? (1.*nFramesSolvent+1.)/(1.*nFramesSolvent-1.) : 1;
        }
        else
        {
            Bprefact = 1;
        }

        /* clear I, grad[k] I */
        for (i = 0; i < nIvalues; i++)
        {
            wd->I          [i] = 0.;
            wd->IA         [i] = 0.;
            wd->varIA      [i] = 0.;
            wd->avAqabs_re [i] = 0.;
            wd->avAqabs_im [i] = 0.;
            if (!wr->bVacuum)
            {
                wd->IB         [i] = 0.;
                wd->Ibulk      [i] = 0.;
                wd->varIB      [i] = 0.;
                wd->varIbulk   [i] = 0.;
                wd->avBqabs_re [i] = 0.;
                wd->avBqabs_im [i] = 0.;
                wd->I_avAmB2   [i] = 0.;
                wd->I_varA     [i] = 0.;
                wd->I_varB     [i] = 0.;
                wd->varI_avAmB2[i] = 0.;
                wd->varI_varA  [i] = 0.;
                wd->varI_varB  [i] = 0.;
                if (wr->bScaleI0)
                {
                    wd->I_scaleI0[i] = 0.;
                }
            }
            if (wd->I_errSolvDens)
            {
                wd->I_errSolvDens[i] = 0;
            }
        }
        if (wr->bCalcForces && wr->bUseGPU == FALSE)   /* do not clear, if dkI is calculated on GPU! */
        {
            for (i = 0; i < nIvalues*nprot; i++)
            {
                clear_dvec(wd->dkI[i]);
            }
        }

        /* Compute D(q) for home-qvalues */
        waxs_debug(" Compute D(q) for home-qvalues ");
        for (i = 0; i < qhomenr; i++)
        {
            if (!wr->bVacuum)
            {
                /* Equilvalent to equation below */
                if (wr->bScaleI0)
                {
                    this_avA = cadd_d( wd->avA[i], wd->avAcorr[i]);
                }
                else
                {
                    this_avA = wd->avA[i];
                }
                wd->D[i] = cabs2_d( csub_d( this_avA, wd->avB[i] ) )  +
                        ( wd->avAsq[i] - cabs2_d( wd->avA[i] ) ) -
                        Bprefact*( wd->avBsq[i] - cabs2_d( wd->avB[i] ) );
            }
            else
            {
                wd->D[i] = wd->avAsq[i];
            }
        }

        if ( wr->stepCalcNindep > 0  &&
             (
                 (wr->waxsStep+1)%wr->stepCalcNindep == 0 ||
                 wr->waxsStep == 0 || wr->waxsStep == 5 || wr->waxsStep == 10 || wr->waxsStep == 20
             ) &&
             wr->ewaxsaniso == ewaxsanisoNO)
        {
            waxsEstimateNumberIndepPoints(wr, cr, FALSE, FALSE);
        }

        switch (wr->ewaxsaniso)
        {
        case ewaxsanisoNO:
            /* - spherical averging of D(\vec{q})
               - Here we have one intensity per absolute q-value: I[0..nabs)
             */

            /* Loop over absolute values of q, |q| */
            for (i = 0; i < nabs; i++)
            {
                /* Loop over q-vectors with this absolute value of q */
                for (j = ind[i]; j < ind[i+1]; j++)
                {
                    /* is this q-vector on this node? */
                    if (j >= qstart && j < (qstart+qhomenr))
                    {
                        jj        = j-qstart;
                        wd->I[i] += wd->D[jj];

                        /* Also get intensity from only atoms A or B and the third term, that is
                           <|A|^2>, <|B|^2>, and -2Re[ <B>.<A-B>*]    */
                        stepA              = wr->waxsStep + 1;
                        wd->IA[i]         += wd->avAsq[jj];
                        wd->varIA[i]      += (wd->avA4[jj] - dsqr(wd->avAsq[jj]))/stepA;
                        /* Also get < A(|q|)> := int_dOmega[q] < A(q) > */
                        wd->avAqabs_re[i] += wd->avA[jj].re;
                        wd->avAqabs_im[i] += wd->avA[jj].im;

                        if (wd->I_errSolvDens)
                        {
                            /* Uncertainty in I(q) due to uncertainty in solvent density */
                            wd->I_errSolvDens[i] += wd->DerrSolvDens[jj];
                        }

                        if (!wr->bVacuum)
                        {
                            stepB = (wr->waxsStep+1 < wr->nfrsolvent) ? wr->waxsStep+1 : wr->nfrsolvent;

                            wd->IB   [i] += wd->avBsq[jj];
                            wd->varIB[i] += (wd->avB4[jj] - dsqr(wd->avBsq[jj]))/stepB;

                            tmpc          = csub_d( wd->avA[jj], wd->avB[jj]);
                            tmpc          = conjugate_d(tmpc);
                            tmpc          = cmul_d(wd->avB[jj], tmpc);
                            wd->Ibulk[i] -= 2*tmpc.re;
                            /* Also get < B(|q|)> := int_dOmega[q] < B(q) > */
                            wd->avBqabs_re[i] += wd->avB[jj].re;
                            wd->avBqabs_im[i] += wd->avB[jj].im;

                            wd->varIbulk[i] +=
                                      dsqr( 2*wd->avB[jj].re                   ) * (wd->av_ReA_2[jj] - dsqr(wd->avA[jj].re))/stepA
                                    + dsqr( 2*wd->avB[jj].im                   ) * (wd->av_ImA_2[jj] - dsqr(wd->avA[jj].im))/stepA
                                    + dsqr(-2*wd->avA[jj].re + 4*wd->avB[jj].re) * (wd->av_ReB_2[jj] - dsqr(wd->avB[jj].re))/stepB
                                    + dsqr(-2*wd->avA[jj].im + 4*wd->avB[jj].im) * (wd->av_ImB_2[jj] - dsqr(wd->avB[jj].im))/stepB;

                            /* Finally, we comute the contributions using the alternative fomula by Park et al.
                               that is D(q) = |<A-B>|^2 + var(A) - var(B)
                               These there terms are stored in variables AmB2, varA, varB
                             */
                            wd->I_avAmB2[i] += cabs2_d(csub_d(wd->avA[jj], wd->avB[jj]));
                            wd->I_varA[i]   += wd->avAsq[jj] - cabs2_d(wd->avA[jj]);
                            wd->I_varB[i]   += wd->avBsq[jj] - cabs2_d(wd->avB[jj]);
                            /* And the respective variances:
                               Note: |<A-B>|^2 = <Re A>^2 + <Im A>^2 + <Re B>^2 - 2*<Re B>.<Re A> (using <Im B> = 0)
                             */
                            wd->varI_avAmB2[i] +=
                                      dsqr( 2*wd->avA[jj].re - 2*wd->avB[jj].re ) * (wd->av_ReA_2[jj] - dsqr(wd->avA[jj].re))/stepA
                                    + dsqr( 2*wd->avA[jj].im                    ) * (wd->av_ImA_2[jj] - dsqr(wd->avA[jj].im))/stepA
                                    + dsqr( 2*wd->avB[jj].re - 2*wd->avA[jj].re ) * (wd->av_ReB_2[jj] - dsqr(wd->avB[jj].re))/stepB;
                            /* The standard error of the variance estimate is sigma^2*sqrt(2/k), where k is the
                               number of data points used to compute the variace
                               Will devide by # of waxs steps (k) when writing the contrib file.
                             */
                            wd->varI_varA[i] +=
                                    2 * dsqr(wd->av_ReA_2[jj] - dsqr(wd->avA[jj].re))/stepA
                                    + 2 * dsqr(wd->av_ImA_2[jj] - dsqr(wd->avA[jj].im))/stepA;
                            wd->varI_varB[i] +=
                                    2 * dsqr(wd->av_ReB_2[jj] - dsqr(wd->avB[jj].re))/stepB;

                            if (wr->bScaleI0)
                            {
                                /* Additional intensity due to scaling of I(q=0)
                                   Iscale_I0 := |<A+dA-B>|^2 - |<A-B>|^2 */
                                tmpc             = csub_d( wd->avA[jj], wd->avB[jj]);
                                tmpc             = cmul_d(conjugate_d(tmpc), wd->avAcorr[jj]);
                                wd->I_scaleI0[i] = 2*tmpc.re + cabs2_d(wd->avAcorr[jj]);
                            }
                        }
                    }
                }
            }
            break;
        case ewaxsanisoYES:
        case ewaxsanisoCOS2:
            /* - No averaging. Simply fill the local D, avAsq, avBsq, etc into global arrays I, IA, IB, etc
               - We have J intensities per absolute q-value: I[0...nabs*J)
               - Recall: with ewaxsanisoYES: nIvalues = nabs*J
             */
            for (j = 0; j < (nIvalues); j++)
            {
                if (j >= qstart && j < (qstart+qhomenr))
                {
                    jj        = j-qstart;
                    wd->I [j] = wd->D    [jj];
                    wd->IA[i] = wd->avAsq[jj];
                    if (!wr->bVacuum)
                    {
                        wd->IB[j]     = wd->avBsq[jj];
                        tmpc          = csub_d( wd->avA[jj], wd->avB[jj]);
                        tmpc          = conjugate_d(tmpc);
                        tmpc          = cmul_d(wd->avB[jj], tmpc);
                        wd->Ibulk[j] -= 2*tmpc.re;
                    }
                    if (wd->I_errSolvDens)
                    {
                        wd->I_errSolvDens[j] = wd->DerrSolvDens[jj];
                    }
                }
            }
            break;
        default:
            gmx_fatal(FARGS, "This aniotropy (%d) is not supported\n", wr->ewaxsaniso);
        }


        /* Summ over the nodes and normalize */
        if (PAR(cr))
        {
            gmx_sumd(nIvalues, wd->I,          cr);
            gmx_sumd(nIvalues, wd->IA,         cr);
            gmx_sumd(nIvalues, wd->varIA,      cr);
            gmx_sumd(nIvalues, wd->avAqabs_re, cr);
            gmx_sumd(nIvalues, wd->avAqabs_im, cr);

            if (!wr->bVacuum)
            {
                gmx_sumd(nIvalues, wd->IB,          cr);
                gmx_sumd(nIvalues, wd->Ibulk,       cr);
                gmx_sumd(nIvalues, wd->varIB,       cr);
                gmx_sumd(nIvalues, wd->varIbulk,    cr);
                gmx_sumd(nIvalues, wd->avBqabs_re,  cr);
                gmx_sumd(nIvalues, wd->avBqabs_im,  cr);
                gmx_sumd(nIvalues, wd->I_avAmB2,    cr);
                gmx_sumd(nIvalues, wd->I_varA,      cr);
                gmx_sumd(nIvalues, wd->I_varB,      cr);
                gmx_sumd(nIvalues, wd->varI_avAmB2, cr);
                gmx_sumd(nIvalues, wd->varI_varA,   cr);
                gmx_sumd(nIvalues, wd->varI_varB,   cr);
            }
            if (wd->I_errSolvDens)
            {
                gmx_sumd(nIvalues, wd->I_errSolvDens, cr);
            }
        }
        if (wr->ewaxsaniso == ewaxsanisoNO)
        {
            /* In case with averaged over all q orientiations: divide by nr of q vectors per |q| */
            for(i = 0; i < nabs; i++)
            {
                N                  = ind[i+1]-ind[i];
                wd->I         [i] /= N;
                wd->IA        [i] /= N;
                wd->varIA     [i] /= N;
                wd->avAqabs_re[i] /= N;
                wd->avAqabs_im[i] /= N;
                /* Get final variance of I based on error propagation */
                if (wr->bVacuum)
                {
                    wd->varI[i] = fabs(wd->varIA[i]);
                }
                if (!wr->bVacuum)
                {
                    wd->IB         [i] /= N;
                    wd->Ibulk      [i] /= N;
                    wd->varIB      [i] /= N;
                    wd->varIbulk   [i] /= N;
                    // wd->varI[i] += wd->varIB[i] + wd->varIbulk[i];
                    wd->avBqabs_re [i] /= N;
                    wd->avBqabs_im [i] /= N;
                    wd->I_avAmB2   [i] /= N;
                    wd->I_varA     [i] /= N;
                    wd->I_varB     [i] /= N;
                    wd->varI_avAmB2[i] /= N;
                    wd->varI_varA  [i] /= N;
                    wd->varI_varB  [i] /= N;
                    wd->varI[i] = fabs(wd->varI_avAmB2[i] + wd->varI_varA[i] + wd->varI_varB[i]);
                }
                if (wr->waxsStep <= 1)
                {
                    /* In the first step, we have no reasonable estimate for varI - just use 50% of I(q) for stddev
                       Otherwise, we get a numercal error in the Bayesian WAXS-MD .
                     */
                    if (MASTER(cr) && i == 0)
                    {
                        printf("WAXS step %d: Using 50%% of I(q) as stddev of I(q)\n", wr->waxsStep);
                    }
                    wd->varI[i] = dsqr(0.5 * wd->I[i]);
                }
                if (wd->I_errSolvDens)
                {
                    wd->I_errSolvDens[i] /= N;
                }
                if (wr->bHaveNindep)
                {
                    waxs_debug("Divide by # of independent q-vectors if we have it already");
                    /* Divide by # of independent q-vectors if we have it already */
                    wd->varI[i] /= wd->Nindep[i];
                }
            }
        }

        if (wr->bCorrectBuffer)
        {
            if (wr->ewaxsaniso != ewaxsanisoNO)
            {
                gmx_fatal(FARGS, "Requesting to correct for oversubtracted buffer. This is only possible "
                        "with isotropic scattering\n");
            }
            if (t > 0)
            {
                gmx_fatal(FARGS, "In calculate_I_dkI(): correcting for oversubtracted buffer only "
                        "implemented for *one* scattering group, which must be X-ray");
            }
            /* Correcting for oversubtracted buffer,
               eq. 2 in Koefinger & Hummer, Phys Rev E 87 052712 (2013) */
            for(i = 0; i < nabs; i++)
            {
                wd->I[i] += wr->soluteVolAv * wt->Ipuresolv[i];
            }
        }

        if (WAXS_ENSEMBLE(wr))
        {
            if (wr->ewaxsaniso != ewaxsanisoNO)
            {
                gmx_fatal(FARGS, "Ensemble-refinement works only with isotropic WAXS\n");
            }

            if (MASTER(cr))
            {
                /* Compute ensemble-averaged Icalc and stddev of Icalc.
                   Note: weights are at present normalized - but let's be sure and normalize again.
                 */
                sum = 0;
                for (m = 0; m < wr->ensemble_nstates; m++)
                {
                    sum += wr->ensemble_weights[m];
                }
                for (i = 0; i < nabs; i++)
                {
                    wt->ensembleSum_I   [i] = 0;
                    wt->ensembleSum_Ivar[i] = 0;
                    /* Sum over fixed states */
                    for (m = 0; m < wr->ensemble_nstates - 1; m++)
                    {
                        wt->ensembleSum_I   [i] += wr->ensemble_weights[m] * wt->ensemble_I   [m][i];
                        wt->ensembleSum_Ivar[i] += wr->ensemble_weights[m] * wt->ensemble_Ivar[m][i];
                    }
                    /* Add the state of the MD simulation */
                    m = wr->ensemble_nstates - 1;
                    wt->ensembleSum_I   [i] += wr->ensemble_weights[m] * wd->I   [i];
                    wt->ensembleSum_Ivar[i] += wr->ensemble_weights[m] * wd->varI[i];
                    /* Normalize */
                    wt->ensembleSum_I   [i] /= sum;
                    wt->ensembleSum_Ivar[i] /= sum;
                }

            }
            if (PAR(cr))
            {
                gmx_bcast(nabs*sizeof(double), wt->ensembleSum_I,    cr);
                gmx_bcast(nabs*sizeof(double), wt->ensembleSum_Ivar, cr);
            }

        }

        /* Compute I(q) and gradients of I(q)  atomic coordinates if not already done on GPU */
        waxs_debug("Compute I(q) and gradients of I(q)  atomic coordinates if not already done on GPU ");

        if (wr->bCalcForces)
        {

            waxs_debug("wr->bCalcForces is true");

            if (wr->bUseGPU)
            {
                waxs_debug("wr->bCalcForces with gpu");
#ifdef GMX_GPU
                double GPU_time ;

                calculate_dkI_GPU (t, wd->dkI , wr->atomEnvelope_coord_A, wt->atomEnvelope_scatType_A, qstart, qhomenr, nabs, nprot, &GPU_time, cr);

                waxsTimingDo(wr, waxsTimedkI, waxsTimingAction_add, GPU_time/1000, cr);
#endif
            }
            else
            {
                waxs_debug("wr->bCalcForces without gpu");
                waxsTimingDo(wr, waxsTimedkI, waxsTimingAction_start, 0, cr);

                if (wr->ewaxsaniso != ewaxsanisoNO)
                {
                    gmx_fatal(FARGS, "Forces from anisotropic WAXS patterns are not implemented.\n");
                }

                for (i = 0; i < nabs; i++)
                {
                    int J  = (ind[i+1] - ind[i]);

                    /* Calculate grad[k] D(vec{q}) and sum immediately into grad[k] I(q) */
                    for (p = 0; p < nprot; p++)
                    {
                        l = p * nabs + i;              /* index of dkI */

                        /* Now calculate dkI, the J denotes the number of q-vectors for this absolute value of q */
                        for (d = 0 ; d < DIM; d++)
                        {
                            wd->dkI[l][d] = 2. * wd->Orientational_Av[l][d] / J ;
                        }
                    }
                }
                waxs_debug("dKi done");

                /* globally sum rvec array and divide by nr of qvecs on q-sphere */
                if (PAR(cr))
                {
                    gmx_sumd(nabs * nprot * DIM, wd->dkI[0], cr);
                    waxs_debug("after gmx_sumd");
                }

                waxsTimingDo(wr, waxsTimedkI, waxsTimingAction_end, 0, cr);
            } /* end dkI calculation on the CPU */
            waxs_debug("end if bCalcForces");
        } /* end if bCalcForces */

#ifdef PRINT_A_dkI
        if (wr->bCalcForces && wr->debugLvl > 1)
        {
            FILE *fdkI;
            char fn_dkI[STRLEN];

            sprintf(fn, "fn_dkI_%s_%s.txt", wr->bUseGPU ? "GPU" : "CPU",
                    wt->type == escatterXRAY ? "XRAY" : "Neutron");
            fdkI = fopen(fn, "w");
            fprintf(fdkI, "nprot is: %d\n", nprot);
            fprintf(fdkI, "nabs  is: %d\n", nabs);
            for(i = 0 ; i < nabs ; i++)
            {
                fprintf(fdkI, "q_abs = %f \n", wt->qvecs->abs[i]);
                for(p = 0; p < nprot ;p++)
                {
                    l = i*nprot + p;              /* index of dkI                   */
                    for (d = 0 ; d < DIM ; d++)
                    {
                        fprintf(fdkI, "dkI [ %d ][%d] =  %f \n", l, d,  wd->dkI[l][d]);
                    }
                    fprintf(fdkI, "\n");
                }
            }
            fclose(fdkI);
            printf("Debugging level %d: Wrote dkI to %s\n", wr->debugLvl, fn);
        }

        /* Test code for comparing CPU- vs. GPU-computed scattering amplitudes */
        if (qstart > 0 && && wr->debugLvl > 1)
        {
            FILE *fp;
            FILE *fb;
            char fn_p[STRLEN],  fn_b[STRLEN];

            sprintf(fn_p, "scattering_amplitude_%s_%s.txt", wr->bUseGPU ? "GPU" : "CPU",
                    wt->type == escatterXRAY ? "XRAY" : "Neutron");
            sprintf(fn_b, "water_scattering_amplitude_%s_%s.txt", wr->bUseGPU ? "GPU" : "CPU",
                    wt->type == escatterXRAY ? "XRAY" : "Neutron");
            fp = fopen(fn_p, "w");
            fb = fopen(fn_b, "w");

            fprintf(fp, "Atom-number HOST start is: %d\n", wr->isizeA);
            fprintf(fp, "qhomenr HOST after loop is: %d\n", qhomenr);
            fprintf(fb, "Atom-number HOST start is: %d\n", wr->isizeB);
            fprintf(fb, "qhomenr HOST after loop is %d \n", qhomenr);

            for (i = 0 ; i < qhomenr; i++)
            {
                fprintf(fp, "Real-part of scattering amplitude [ %d ] :  %f \n", i, wd->A[i].re);
                fprintf(fp, "Im  -part of scattering amplitude [ %d ] :  %f \n", i, wd->A[i].im);
                fprintf(fp, " \n");

                fprintf(fb, "Real-part of scattering amplitude [ %d ] :  %f \n", i, wd->B[i].re);
                fprintf(fb, "Im  -part of scattering amplitude [ %d ] :  %f \n", i, wd->B[i].im);
                fprintf(fb, " \n");
            }

            fclose(fp);
            fclose(fb);
            printf("Debugging Level %d: Wrote some scattering amplitudes to %s and %s\n", wr->debugLvl, fn_p, fn_b);
        }
#endif

    } /* end t-loop */

    waxs_debug("Leaving calculate_I_dkI()");
}


/* Compute total force and torque on solute atoms and write to log file */
void
write_total_force_torque(t_waxsrec *wr, rvec *x)
{
    int p, nprot, ix;
    rvec fsum, torque, c;
    real faver = 0.;

    nprot = wr->nindA_prot;
    clear_rvec(fsum);
    clear_rvec(torque);

    for(p = 0; p < nprot; p++)
    {
        faver += sqrt(norm2(wr->fLast[p]));
        rvec_inc(fsum, wr->fLast[p]);

        ix = wr->indA_prot[p];
        cprod(x[ix], wr->fLast[p], c);
        rvec_inc(torque, c);
    }
    faver /= nprot;
    fprintf(wr->wo->fpLog, "Average absolute WAXS force on atoms [kJ/(mol nm)] = %8.4g\n", faver);
    fprintf(wr->wo->fpLog, "Sum of WAXS forces [kJ/(mol nm)] = ( %10.5g  %10.6g  %10.5g )\n",
            fsum[XX], fsum[YY], fsum[ZZ]);
    fprintf(wr->wo->fpLog, "Total torque [kJ/(mol nm) nm]    = ( %10.5g  %10.6g  %10.5g )\n",
            torque[XX], torque[YY], torque[ZZ]);
}

static double
switch_fn(double x, double min, double max, gmx_bool bOn )
{
    if (bOn)
    {
        if (x<min)
            return 0;
        if (x>max)
            return 1;

        return 0.5 * (1. - cos(M_PI*(x-min)/(max-min)));
    }
    else
    {
        if (x<min)
            return 1;
        if (x>max)
            return 0;
        return 0.5 * cos(M_PI * (x-min)/(max-min));
    }
}


/* Return difference between calculated and experiemental intensity on log or linear scale */
static inline double
waxs_intensity_Idiff(t_waxsrec *wr, t_waxsrecType *wt, int iq, gmx_bool bAllowNegativeIcalc, gmx_bool bVerbose)
{
    double Idiff = 0, IexpFitted;

    /* Works with or without fitting. Without fitting, f=1 and c=0, with fitting only the scale, c=0 */
    IexpFitted = wt->f_ml * wt->Iexp[iq] + wt->c_ml;
    switch (wr->potentialType)
    {
    case ewaxsPotentialLOG:
        waxs_debug("case ewaxsPotentialLOG");
        if (IexpFitted <= 0 && bAllowNegativeIcalc == FALSE)
        {
            gmx_fatal(FARGS, "After fitting the experimental curve I(fitted) = f*I(exp)+c to the calculated curve,\n"
                      "we found negative fitted I(q) (%g) at q = %g. This will not work with a WAXS potential on the log scale.\n"
                      "Use either a linear scale, try a lower q-range, or fit only the scale f but not the offset c. (f = %g  c = %g)",
                      IexpFitted, wt->qvecs->abs[iq], wt->f_ml, wt->c_ml);
        }
        if (wt->wd->I[iq] <= 0 && bAllowNegativeIcalc == FALSE)
        {
            gmx_fatal(FARGS, "Found a negative computed intensity at q = %g (iq = %d): Icalc = %g.\n"
                      "This will not work when coupling on a log scale. Choose a linear coupling scale instead.\n",
                      wt->qvecs->abs[iq], iq, wt->wd->I[iq]);
        }
        if ( (wt->wd->I[iq] <= 0 || IexpFitted <= 0) && bAllowNegativeIcalc)
        {
            if (bVerbose)
            {
                printf("\nNOTE!!!\nFound a negative Icalc = %g or negative fitted Iexp = %g at q = %g (iq = %d), while coupling on the log scale.\n"
                       "This is ok at the beginning of the simulation (t < waxs_tau), when Icalc might not yet be converged.\n\n",
                       wt->wd->I[iq], IexpFitted, wt->qvecs->abs[iq], iq);
            }
            Idiff = 0;
        }
        else
        {
            Idiff = log(wt->wd->I[iq]/IexpFitted);
        }
        break;
    case ewaxsPotentialLINEAR:
        Idiff = wt->wd->I[iq] - IexpFitted;
        break;
    default:
        gmx_fatal(FARGS, "Unknown type of WAXS potential (type %d, should be < %d)\n",
                  wr->potentialType, ewaxsPotentialNR);
    }

    if (isnan(Idiff) || isinf(Idiff))
    {
        gmx_fatal(FARGS, "Error in waxs_intensity_Idiff(), with potential type = %s :\n"
                  "Found for iq = %d: I = %g -- Idiff = %g (NaN/inf) -- IexpFitted = %g -- f_ml = %g -- c_ml = %g\n",
                  EWAXSPOTENTIAL(wr->potentialType), iq, wt->wd->I[iq], Idiff, IexpFitted, wt->f_ml, wt->c_ml);
    }

    return Idiff;
}

/* Return sigma that goes into the WAXS potential, depending on scale (log or linear) and weights type
   This function is only used for non-bayesian MD.
 */
static inline double
waxs_intensity_sigma(t_waxsrec *wr, t_waxsrecType *wt, double *varIcalc, int iq)
{
    double sigma = 0;

    /* Choose the variance that goes into Ewaxs */
    switch (wr->weightsType)
    {
    case ewaxsWeightsUNIFORM:
        sigma = 1;
        break;
    case ewaxsWeightsEXPERIMENT:
        sigma = wt->f_ml * wt->Iexp_sigma[iq];
        break;
    case ewaxsWeightsEXP_plus_CALC:
        sigma = sqrt( dsqr(wt->f_ml * wt->Iexp_sigma[iq]) + varIcalc[iq] );
        break;
    case ewaxsWeightsEXP_plus_SOLVDENS:
        sigma = sqrt( dsqr(wt->f_ml * wt->Iexp_sigma[iq]) + dsqr(wr->epsilon_buff * wt->wd->I_errSolvDens[iq] ));
        break;
    case ewaxsWeightsEXP_plus_CALC_plus_SOLVDENS:
        sigma = sqrt( dsqr(wt->f_ml * wt->Iexp_sigma[iq]) + varIcalc[iq] + dsqr(wr->epsilon_buff * wt->wd->I_errSolvDens[iq] ));
        break;
    default:
        gmx_fatal(FARGS, "Unknown type of WAXS weights type (type %d, should be < %d)\n",
                wr->weightsType, ewaxsWeightsNR);
    }

    if (wr->potentialType == ewaxsPotentialLOG && wr->weightsType != ewaxsWeightsUNIFORM)
    {
        /* With a potential on a log scale, the uncertainty is delta[log I] = (delta I)/I */
        sigma /=  wt->Iexp[iq];
    }

    return sigma;
}

/* Maxiumum-likelihodd estimate for f and c, for fitting f*Iexp+c to Icalc.
   Eqs. 13 and 14 in Shevchuk & Hub, Plos Comp Biol https://doi.org/10.1371/journal.pcbi.1005800
 */
static void
maximum_likelihood_est_fc(double *Icalc, double *Iexp, double *tau, int nq,
                          double *f_ml, double *c_ml, double *sumSquaredResiduals)
{
    int    i;
    double av_c, av_e, sig_c, sig_e, P, sum_tau;

    /* These functions also work with tau == NULL */
    average_stddev_d(Icalc, nq, &av_c, &sig_c, tau);
    average_stddev_d(Iexp,  nq, &av_e, &sig_e, tau);
    P = pearson_d(nq, Icalc, Iexp, av_c, av_e, sig_c, sig_e, tau);

    /* Maximum likelihood estimates for f and c */
    *f_ml = P * sig_c / sig_e;
    *c_ml = av_c - (*f_ml) * av_e;
    if (sumSquaredResiduals)
    {
        sum_tau              = tau ? sum_array_d(nq, tau) : nq;
        /* this sumSquaredResiduals calculation is accurate up to at least 10 digits, checked. */
        *sumSquaredResiduals = sum_tau * ( sig_c*sig_c * (1.0 - P*P) );
    }
}

/* ML estimate for f only (but not c), for fitting f*Iexp to Icalc.
   Eq. above Eq. 23 in Shevchuk & Hub, Plos Comp Biol https://doi.org/10.1371/journal.pcbi.1005800
 */
static void
maximum_likelihood_est_f(double *Icalc, double *Iexp, double *tau, int nq,
                         double *f_ml, double *sumSquaredResiduals)
{
    int    i;
    double avY2, avXY, avX2, sum_tau;

    /* These functions also work with tau == NULL */
    average_x2_d(Iexp,        nq, &avY2, tau);
    average_xy_d(Iexp, Icalc, nq, &avXY, tau);

    /*  Maximum likelihood estimates for scale
        Note that we here fit the experiment to the calculated curve. Then we have:
     */
    *f_ml = avXY / avY2;
    if (sumSquaredResiduals)
    {
        average_x2_d(Icalc, nq, &avX2, tau);
        sum_tau              = tau ? sum_array_d(nq, tau) : nq;
        *sumSquaredResiduals = sum_tau * ( avX2 - (*f_ml) * avXY );
    }
}

static void
maximum_likelihood_est_fc_generic(t_waxsrec *wr,
                                  double *Icalc, double *Iexp, double *tau,
                                  int nq, double *f_ml, double *c_ml, double *sumSquaredResiduals)
{
    switch (wr->ewaxs_Iexp_fit)
    {
    case ewaxsIexpFit_NO:
        if (sumSquaredResiduals)
        {
            sum_squared_residual_d(Icalc, Iexp, nq, sumSquaredResiduals, tau);
        }
        *f_ml = 1;
        *c_ml = 0;
        break;
    case ewaxsIexpFit_SCALE:
        maximum_likelihood_est_f(Icalc, Iexp, tau, nq, f_ml, sumSquaredResiduals);
        *c_ml = 0;
        break;
    case ewaxsIexpFit_SCALE_AND_OFFSET:
        maximum_likelihood_est_fc(Icalc, Iexp, tau, nq, f_ml, c_ml, sumSquaredResiduals);
        break;
    default:
        gmx_fatal(FARGS, "Invalid value for wr->ewaxs_Iexp_fit (%d)\n", wr->ewaxs_Iexp_fit);
        break;
    }
}

/* Computes the log Likelihood, after marginalizing the fitting parameters (f and c, only f, or none).
   - tau-array must be allocated before entry.
   - Icalc / IcalcVar is the overall calculated intensities, which may contain contributions
     from a heterogenous ensmble or not.
 */
static double
bayesian_md_calc_logLikelihood(t_waxsrec *wr, int t, double *Icalc, double *IcalcVar, double *tau)
{
    int             i;
    double          sumRes2 = 0;
    double          diffI, sigma, sumSquaredResiduals = 0, zeta, logL;
    t_waxsrecType  *wt;

    if (wr->nTypes > 1)
    {
        gmx_fatal(FARGS, "Do now allow more than one scattering type with Bayeisan stuff\n");
    }
    if (!tau)
    {
        gmx_fatal(FARGS, "Found tau == NULL in bayesian_md_calc_logLikelihood()\n");
    }
    if (t >= wr->nTypes)
    {
        gmx_fatal(FARGS, "Invalid waxs type number (%d)\n", t);
    }

    wt   = &wr->wt[t];
    zeta = wt->nShannon / wt->nq;
    for (i = 0; i< wt->nq; i++)
    {
        /* Compute the statistical "precisions", or 1/sigma^2 */
        tau[i]    = 1./dsqr(waxs_intensity_sigma(wr, wt, IcalcVar, i));
    }

    /* get \hat{chi}^2, that is the sum of squared residuals (weighted by tau) */
    maximum_likelihood_est_fc_generic(wr, Icalc, wt->Iexp, tau, wt->nq, &wt->f_ml, &wt->c_ml, &sumSquaredResiduals);
    logL = -0.5 * zeta * sumSquaredResiduals;

    if (isnan(logL) || isinf(logL))
    {
        gmx_fatal(FARGS, "Encountered an invalid log-likilihood (logL = %g)\n", logL);
    }

    return logL;
}


/* Do Gibbs sampling of the uncertainty epsilon_buff in the buffer density

   epsilon_buff = 0  -> means no uncertainty due to the buffer density
   epsilon_buff = 1  -> means the uncertainty is the one given by the mdp option "waxs-solvdens-uncert"
   epsilon_buff = 2  -> uncertainty 2 times waxs-solvdens-uncert, etc.

   As prior for epsilon_buff, we use a Gaussian with width = 1
 */
#define WAXS_BAYESIAN_EPSBUFF_MAX 6  /* Maximum epsilon (or number of sigmas) sampled */
#define GIBBS_NST_OUTPUT 10          /* Write to waxs_gibbs.dat ever XX steps         */
static int
bayesian_sample_buffer_uncertainty(t_waxsrec *wr, double simtime, double *Icalc, double *IcalcVar, int nMCmove)
{
    int            i, nAccept = 0, t;
    double         logL, logLprev, eps_prev, r, *tau = NULL, priorFact, pjointRel;
    gmx_bool       bAccept;
    t_waxsrecType *wt;
    FILE          *fp;

    if (wr->nTypes > 1)
    {
        gmx_fatal(FARGS, "Do now allow more than one scattering type with Bayeisan stuff\n");
    }

    t  = 0;
    wt = &wr->wt[t];
    fp = wr->wo->fpGibbsSolvDensRelErr[t];

    snew(tau, wt->nq);
    logLprev = bayesian_md_calc_logLikelihood(wr, t, Icalc, IcalcVar, tau);
    eps_prev = wr->epsilon_buff;

    fprintf(fp, "%8g", simtime);

    for (i = 0; i < nMCmove; i++)
    {
        /* Propose update of epsilon_buff */
        wr->epsilon_buff = gmx_rng_uniform_real(wr->rng) * WAXS_BAYESIAN_EPSBUFF_MAX;

        /* Compute new likelihood */
        logL = bayesian_md_calc_logLikelihood(wr, t, Icalc, IcalcVar, tau);

        /* Prior for epsilon is a Gaussian of width 1: pi(eps) = exp(-eps^2/2) */
        priorFact = exp( - 0.5* (dsqr(wr->epsilon_buff) - dsqr(eps_prev)) );
        pjointRel = exp(logL - logLprev) * priorFact;

        /* Accept according to Metropolis criterium */
        if (pjointRel > 1)
        {
            bAccept = TRUE;
        }
        else
        {
            r = gmx_rng_uniform_real(wr->rng);
            bAccept = (r < pjointRel);
        }
        if (bAccept)
        {
            logLprev    = logL;
            eps_prev = wr->epsilon_buff;
            nAccept++;
        }
        else
        {
            /* put back old value */
            wr->epsilon_buff = eps_prev;
            logL             = logLprev;
        }

        if ((i%GIBBS_NST_OUTPUT) == 0)
        {
            /* Writing the current relative uncertainty of the solvent density (e.g. 0.01 measn 1% uncertainty) */
            fprintf(fp, " %g", wr->epsilon_buff*wr->solventDensRelErr);
        }
    }
    fprintf(fp, "\n");

    sfree(tau);
    return nAccept;
}

/*
  Ensemble-average intensities, averaged over M-1 fixed states plus the state of the MD simulation .
  Weights don't have to be normalized - however, so far they are already normalized upon entry.
 */
static void
intensity_ensemble_average(int nq, int M, double *w, double **I_fixed, double **Ivar_fixed, double *I_MD, double *Ivar_MD,
        double *Icalc, double *IcalcVar)
{
    int i, m;
    double wsum = 0;

    for (m = 0; m < M; m++)
    {
        wsum += w[m];
    }

    /* Loop over q-vectors */
    for (i = 0; i < nq; i++)
    {
        Icalc   [i] = 0;
        IcalcVar[i] = 0;
        /* Loop over fixed states */
        for (m = 0; m < M-1; m++)
        {
            Icalc   [i] += w[m] * I_fixed   [m][i];
            IcalcVar[i] += dsqr(w[m]) * Ivar_fixed[m][i];  /* Gaussian error propagation */
        }
        /* Add I / varI of MD simulation state */
        m = M-1;
        Icalc   [i] += w[m] * I_MD   [i];
        IcalcVar[i] += dsqr(w[m]) * Ivar_MD[i];

        /* Normalize */
        Icalc   [i] /= wsum;
        IcalcVar[i] /= dsqr(wsum);
    }
}

static int
compare_double (const void * a, const void * b)
{
    double diff = *(double*)a - *(double*)b;

    if (diff > 0)
        return 1;
    else if (diff < 0)
        return -1;
    else
        return 0;
}


#define GMX_WAXS_ENSWEIGHT_MOVE_MAX 0.1
/* Allow weights also < 0 or > 1, so to avoid artifacts at the 0/1 boundary in weights space. */
static int
bayesian_sample_ensemble_weights(t_waxsrec *wr, double simtime, int nMCmove)
{
    int            i, nAccept = 0, M, m, t;
    double         logL, logLNew, r, *tau = NULL, pjointRel, *Icalc, *IcalcVar, *wNew, *R, deltaVweights, xi;
    double         wNonScaled;
    gmx_bool       bAccept, bOutsideBounds;
    t_waxsrecType *wt;
    FILE          *fp;

    if (wr->nTypes > 1)
    {
        gmx_fatal(FARGS, "Do now allow more than one scattering type with Bayeisan stuff\n");
    }

    t  = 0;
    wt = &wr->wt[t];
    M  = wr->ensemble_nstates;
    fp = wr->wo->fpGibbsEnsW[t];
    snew(tau,      wt->nq);
    snew(Icalc,    wt->nq);
    snew(IcalcVar, wt->nq);
    snew(R, M+1);

    /* Get placeholder for weights, and initiate it with the old weights */
    snew(wNew, M);
    memcpy(wNew, wr->ensemble_weights, M*sizeof(double));
    intensity_ensemble_average(wt->nq, M, wr->ensemble_weights,
                               wt->ensemble_I, wt->ensemble_Ivar, wt->wd->I, wt->wd->varI,
                               Icalc, IcalcVar);

    logL = bayesian_md_calc_logLikelihood(wr, t, Icalc, IcalcVar, tau);


    /* We need to draw random weights, such that the sum of the weigths is one
       To generate such N uniformly distributed random numbers with a sum of 1,
       one typically does the following:
       1) Draw N-1 random numbers R between 0 and 1 and put them in a list L.
       2) Also add 0 and 1 to the list.
       3) Sort the list.
       4) The N random numbers are given by L1-L0, L2-L1, ... L[N]-L[N-1]

       Note: I tested this against the Python code given on the Wikipedia site, all fine.

       So, in terms of a MC step, we can randomly move the N-1 random numbers R.

       Addition: Due to the memory effect in our calculated SAXS curves, we get artifacts
                 at the hard boundaries at 0 and 1 of the weights. Therefore, extend the
                 flat Dirichlet prior such that weights slightly smalle 0 and larger 1
                 are possible as well. This is achieved as follows:

                 1) Compute the weights \vec{w0} according to the Dirichelet distribution, see above.
                 2) Multiply the weights vector \vec{w0} by (1+xi*N) and shift it back along the vector
                    -xi-(1,....1), such that ||w0||_1 = 1, so the weights remain normalized:

                    \vec{wc} = (1+xi*N) * \vec{w0} - xi*(1,...,1)

                   Then, the new weight vector \vec{wc} is still normalized, but its elements are within
                   the element:

                       wc_i \in [-xi, 1 + N*xi - xi]

                   for example with N = 2, wc_i \in [-xi, 1 + xi]  (instead of [0,1])
    */

    R[0] = 0;
    R[M] = 1;

    /* Allow state weights in interval [-xi, 1 + M*xi - xi] */
    xi   = wr->stateWeightTolerance;

    for (i = 0; i < nMCmove; i++)
    {
        /* Set back weights */
        memcpy(wNew, wr->ensemble_weights, M*sizeof(double));

        /* Set the list of random numbers between 0 and 1
           R[1]      is the weight of state 0
           R[2]-R[1] is the weight of state 1, etc
         */

        for (m = 0; m < M-1; m++)
        {
            /* Weights wNew[m] can be <0 or >1 if wr->stateWeightTolerance > 0, so map back
               to the [0,1] interval .
            */
            wNonScaled = (wNew[m] + xi)/(1 + M*xi);
            R[m+1] = R[m] + wNonScaled;
            if (R[m+1] > 1.001 || R[m+1] < -0.001)
            {
                gmx_fatal(FARGS, "Error while initiating R (should be 0<= R <= 1), found %g\n", R[m+1]);
            }
        }

        /* Do random moves of the R */
        bOutsideBounds = FALSE;
        for (m = 0; m < M-1; m++)
        {
            R[m+1] += (2*gmx_rng_uniform_real(wr->rng) - 1) * GMX_WAXS_ENSWEIGHT_MOVE_MAX;
            if (R[m+1] >= 1. || R[m+1] < 0.)
            {
                bOutsideBounds = TRUE;
            }
        }
        if (bOutsideBounds)
        {
            /* Reject move when one of the R[] is outside [0,1] */
            continue;
        }

        /* Sort R - use qsort_threadsafe() defined in gmx_sort.h */
        qsort_threadsafe(R, M+1, sizeof(double), compare_double);

        /* Pick new weights as differences between neighboring elements in sorted R */
        for (m = 0; m < M; m++)
        {
            wNew[m] = R[m+1] - R[m];
        }

        /* Allow weights slightly larger 1 or smaller 0, so to avoid artifacts
           at the boundary. This still yields a normalized weight vector (sum_i w_i = 1),
           that is still uniformly distributed.
        */
        for (m = 0; m < M; m++)
        {
            wNew[m] = (1 + M*xi) * wNew[m] - xi;
        }

        /* Compute new calculated intensity */
        intensity_ensemble_average(wt->nq, M, wNew,
                                   wt->ensemble_I, wt->ensemble_Ivar, wt->wd->I, wt->wd->varI,
                                   Icalc, IcalcVar);

        /* Compute new likelihood */
        logLNew = bayesian_md_calc_logLikelihood(wr, t, Icalc, IcalcVar, tau);

        /* Relative probabilities = P[new]/P[old] */
        pjointRel = exp(logLNew - logL);

        /* Add contribution to P[new]/P[old] from umbrella potential on weights */
        if (wr->ensemble_weights_fc > 0)
        {
            deltaVweights = 0;
            /* We sum over M-1 states only, since the M'th state is fixed by the other M-1 states anyway */
            for (m = 0; m < M-1; m++)
            {
                /* Change in umbrella potential fc/2*(w-w0)^2 on weights */
                deltaVweights += dsqr(wNew[m] - wr->ensemble_weights_init[m]) - dsqr(wr->ensemble_weights[m] - wr->ensemble_weights_init[m]);
            }
            deltaVweights *= 0.5*wr->ensemble_weights_fc;
            pjointRel     *= exp(-deltaVweights/wr->kT);
        }

        /* Accept according to Metropolis criterium */
        if (pjointRel > 1)
        {
            bAccept = TRUE;
        }
        else
        {
            r = gmx_rng_uniform_real(wr->rng);
            bAccept = (r < pjointRel);
        }
        if (bAccept)
        {
            logL    = logLNew;
            memcpy(wr->ensemble_weights, wNew, M * sizeof(double));
            nAccept++;
        }

        if ((i%GIBBS_NST_OUTPUT) == 0)
        {
            fprintf(fp, "%g ", simtime);
            for (m = 0; m < M; m++)
            {
                fprintf(fp, " %g", wr->ensemble_weights[m]);
            }
            fprintf(fp, "\n");
        }
        /* printf("\tGibbs weights: %g %g | %g - %d | w =", logL, logLNew, pjointRel, bAccept); */
        /* for (int m=0; m<M; m++) printf(" %g", wr->ensemble_weights[m]); printf("\n"); */
        /* printf("\tTry was: "); */
        /* for (int m=0; m<M; m++) printf(" %g", wNew[m]); printf("\n"); */
    }

    sfree(tau);
    sfree(Icalc);
    sfree(IcalcVar);
    sfree(wNew);
    sfree(R);

    return nAccept;
}


/* Specify number of Monte-Carlo Moves */
#define WAXS_GIBBS_N_MONTECARLO_MOVES 100       /* Buffer density only or ensemble only */
#define WAXS_GIBBS_BOTH_N_ROUNDS       20       /* Rounds of both density and ensemble */
static void
bayesian_gibbs_sampling(t_waxsrec *wr, double simtime, t_commrec *cr)
{
    int            i, nAcceptW, nAcceptEps, m, t;
    double         relAcceptEps, relAcceptW;
    t_waxsrecType *wt;
    gmx_bool       bDoEnsmbleW, bDoSolvDens;

    if (wr->nTypes > 1)
    {
        gmx_fatal(FARGS, "Do now allow more than one scattering type for a Bayesian treatment of\n"
                  "systematic errors and/or for refining heterogeneous ensembles.");
    }

    waxsTimingDo(wr, waxsTimeGibbs, waxsTimingAction_start, 0, cr);

    t  = 0;
    wt = &wr->wt[t];

    bDoEnsmbleW = (wr->ewaxs_ensemble_type == ewaxsEnsemble_BayesianOneRefined);
    bDoSolvDens = wr->bBayesianSolvDensUncert;

    if (wt->wd->I_errSolvDens == NULL && bDoSolvDens)
    {
        gmx_incons("bayesian_gibbs_sampling(): I_errSolvDens not set, but requesting bDoSolvDens");
    }
    if (bDoSolvDens && wr->weightsType != ewaxsWeightsEXP_plus_CALC_plus_SOLVDENS)
    {
        gmx_fatal(FARGS, "Requested Bayesian sampling of solvent density uncertainty, but not using solvent\n"
                  "density uncertiainty for potential and likelihood. Use waxs-weights = exp+calc+solvdens\n");
    }

    /* Make sure that we have */
    if (MASTER(cr))
    {
        if (bDoSolvDens && !bDoEnsmbleW)
        {
            /* Gibbs sampling of the uncertainty of the buffer density */
            nAcceptEps   = bayesian_sample_buffer_uncertainty(wr, simtime,
                                                              wt->wd->I, wt->wd->varI, WAXS_GIBBS_N_MONTECARLO_MOVES);
            relAcceptEps = 1.0*nAcceptEps/WAXS_GIBBS_N_MONTECARLO_MOVES;
            printf("Gibbs sampling, final bufEps = %12g - (acceptance rate %g %%)\n", wr->epsilon_buff, 100.*relAcceptEps);
        }
        else if (!bDoSolvDens && bDoEnsmbleW)
        {
            /* Gibbs sampling of only ensemble weights */
            nAcceptW   = bayesian_sample_ensemble_weights(wr, simtime, WAXS_GIBBS_N_MONTECARLO_MOVES);
            relAcceptW = 1.0*nAcceptW/WAXS_GIBBS_N_MONTECARLO_MOVES;
            printf("Gibbs sampling of ensemble weigths (acceptance rate %g %%). Final weights =", 100.*relAcceptW);
            for (m = 0; m < wr->ensemble_nstates; m++)
            {
                printf(" %10g", wr->ensemble_weights[m]);
            }
            printf("\n");
        }
        else if (bDoSolvDens && bDoEnsmbleW)
        {
            /* Doing several rounds of Gibbs sampling of ensemble weights and buffer density uncertainty */
            nAcceptEps = 0;
            nAcceptW   = 0;
            for (i = 0; i < WAXS_GIBBS_BOTH_N_ROUNDS; i++)
            {
                /* Doing a few rounds of Gibbs sampling of the buffer uncertainty */
                intensity_ensemble_average(wt->nq, wr->ensemble_nstates, wr->ensemble_weights,
                                           wt->ensemble_I, wt->ensemble_Ivar, wt->wd->I, wt->wd->varI,
                                           wt->ensembleSum_I, wt->ensembleSum_Ivar);
                nAcceptEps += bayesian_sample_buffer_uncertainty(wr, simtime, wt->ensembleSum_I, wt->ensembleSum_Ivar,
                                                                 WAXS_GIBBS_BOTH_N_ROUNDS);

                /* Doing a few rounds of Gibbs sampling on the weights of the states of the ensemble */
                nAcceptW += bayesian_sample_ensemble_weights(wr, simtime, WAXS_GIBBS_BOTH_N_ROUNDS);
            }

            relAcceptEps = 1.0*nAcceptEps / (WAXS_GIBBS_BOTH_N_ROUNDS * WAXS_GIBBS_BOTH_N_ROUNDS);
            relAcceptW   = 1.0*nAcceptW   / (WAXS_GIBBS_BOTH_N_ROUNDS * WAXS_GIBBS_BOTH_N_ROUNDS);

            printf("Gibbs sampling. Accept. rates: bufEps: %6.2f %%   Weights: %6.2f %%\n", 100.*relAcceptEps, 100.*relAcceptW);
            printf("Final bufEps = %10g      Final weights =", wr->epsilon_buff);
            for (m = 0; m < wr->ensemble_nstates; m++)
            {
                printf(" %10g", wr->ensemble_weights[m]);
            }
            printf("\n");
        }
        else
        {
            gmx_incons("Inconsistency in bayesian_gibbs_sampling()");
        }

        if (bDoEnsmbleW)
        {
            /* Update the ensemble average with the new weights */
            intensity_ensemble_average(wt->nq, wr->ensemble_nstates, wr->ensemble_weights,
                                       wt->ensemble_I,    wt->ensemble_Ivar, wt->wd->I, wt->wd->varI,
                                       wt->ensembleSum_I, wt->ensembleSum_Ivar);
        }

    } /* end if MASTER */

    /* brodcast final variables after Gibbs sampling */
    if (PAR(cr))
    {
        if (bDoEnsmbleW)
        {
            gmx_bcast(wr->ensemble_nstates*sizeof(double), wr->ensemble_weights, cr);
            gmx_bcast(wt->nq*              sizeof(double), wt->ensembleSum_I,    cr);
            gmx_bcast(wt->nq*              sizeof(double), wt->ensembleSum_Ivar, cr);
        }
        if (bDoSolvDens)
        {
            gmx_bcast(                     sizeof(double), &wr->epsilon_buff,    cr);
        }
    }

    waxsTimingDo(wr, waxsTimeGibbs, waxsTimingAction_end, 0, cr);
}

static void
clear_vLast_fLast(t_waxsrec *wr)
{
    int p, t, nprot = wr->nindA_prot;

    /* Clear waxs potential */
    wr->vLast = 0;
    for (t = 0; t < wr->nTypes; t++)
    {
        wr->wt[t].vLast = 0;
    }

    /* Clear waxs forces */
    if (wr->fLast)
    {
        for(p = 0; p < nprot; p++)
        {
            clear_rvec(wr->fLast[p]);
            for (t = 0; t < wr->nTypes; t++)
            {
                clear_rvec(wr->wt[t].fLast[p]);
            }
        }
    }
}


/* Comute the SAXS/SANS-derived potentials and forces */
static void
waxs_md_pot_forces(t_commrec *cr, t_waxsrec *wr, real simtime, matrix *Rinv)
{
    t_waxs_datablock wd;
    t_waxs_eavrmsd   we = wr->wrmsd;
    t_waxsrecType   *wt;
    real             fact1, fForceSwitch,fact0, thisv, fact_target_switch, contrastFactorUsed;
    double           fac1, fac2, fac3 = 1.0, *diffI = NULL, *sigma = NULL, this_simtime;
    double          *Icalc, *IcalcVar, sumSquaredResidualsRecomputed, nShannonFactorUsed;
    int              p, ii, d, nabs, nprot, i, l, t;
    rvec             ftmp, ftmp2;
    double           avX, avY, sigX, sigY, pearson, sumSquaredResiduals = -1, *tau = NULL, rel;
    static int       nWarnNotFittedIexp = 0;
    gmx_bool         bAllowNegativeIcalc = FALSE;

    waxs_debug("Entering waxs_md_pot_forces()");

    if (WAXS_ENSEMBLE(wr) && wr->nTypes > 1)
    {
        gmx_fatal(FARGS, "Ensemble refinemnt so far only with one scattering type (found wr->nTypes = %d)\n", wr->nTypes);
    }
    if (!wr->bCalcPot)
    {
        gmx_incons("wr->bCalcPot false in waxs_md_pot_forces()");
    }

    for (t = 0; t < wr->nTypes; t++)
    {
        wt = &wr->wt[t];
        wd = wt->wd;

        nprot = wr->nindA_prot;
        nabs  = wt->nq;

        /*
         *  Depending on whether we do ensemble refinement, the calculated intensity
         *  is the ensemble-averaged I(q) or the simply I(q) computed from the MD.
         *  So let's define two pointer on the respective arrays.
         */
        if (WAXS_ENSEMBLE(wr))
        {
            Icalc    = wt->ensembleSum_I;
            IcalcVar = wt->ensembleSum_Ivar;
        }
        else
        {
            Icalc    = wd->I;
            IcalcVar = wd->varI;
        }

        snew(diffI, nabs);
        snew(sigma, nabs);
        snew(tau,   nabs);

        /* For cumulative (non-weighted) average of potentials vAver */
        fac1 = 1.0*(wr->waxsStep)/(wr->waxsStep+1);
        fac2 = 1.0/(wr->waxsStep+1);

        /* Moderate the forces if rapid conformational change occurs
         * Ignore case when sd ~ zero, e.g. on first step. */
        if (wr->bDampenForces && we->sd_now > GMX_FLOAT_EPS )
        {
            fac3 = switch_fn(fabs(we->rmsd_now - we->rmsd_av)/we->sd_now,
                             wr->damp_min, wr->damp_max, FALSE);
        }
        else
        {
            fac3 = 1;
        }

        /* Smoothly switching on the force between tau < (simtime-simtime0) < 2tau. This
           makes sure that we don't apply force while the I(q) are not yet converged */
        this_simtime = simtime - wr->simtime0;
        if (this_simtime > 2.0*wr->tau || !wr->bSwitchOnForce)
        {
            fForceSwitch = 1.;
        }
        else if (this_simtime < wr->tau)
        {
            fForceSwitch = 0.;
        }
        else
        {
            fForceSwitch = fac3*0.5*(1. - cos(M_PI*(this_simtime-wr->tau)/(wr->tau) ));
        }
        if (MASTER(cr) && wr->wo->fpLog && fForceSwitch<1.0)
            fprintf(wr->wo->fpLog,"While switching on potential and force: using factor of %g (damp factor due to RMSD = %g)\n",
                    fForceSwitch, fac3);

        /* Get relative weights of current I(q) and target I(q), which enter the I(q) to which we couple.
           Here we use the total simulation time, not the simulation time from the last restart.
         */
        if (simtime > wr->t_target || wr->t_target <= 0.)
        {
            /* Only use input pattern */
            fact_target_switch = 1.;
        }
        else
        {
            /* Linearly switch from current to target pattern.
             * Note:  I_couple = (1-f)I_sim + f*I_target.
             *        -> V  = k*(I_sim-I_couple)^2 = k*f^2*(I_sim-I_target)^2
             *        -> So we only get an additional factor of f^2
             */
            fact_target_switch = simtime / wr->t_target;
        }
        if (MASTER(cr) && wr->wo->fpLog && fact_target_switch<1.0)
        {
            fprintf(wr->wo->fpLog, "Relative weight of target pattern vs. current pattern %g\n",
                    fact_target_switch);
        }

        /*
         * Get maximum likelihood estimate for f (overall scale) and c (offset)
         *
         * At the begninnig, we don't have any estiate for f yet, so we cannot compute the scaling factor entering
         * the experimental errors. Therefore, get a first good guess for f using uniform weights:
         */
        if (wt->f_ml == 0)
        {
            maximum_likelihood_est_fc_generic(wr, Icalc, wt->Iexp, NULL, wt->nq, &wt->f_ml, &wt->c_ml, NULL);
        }

        /* Now we have a reasonable estimate for f, and we can compute the statistical weights (or "precisions")
           tau_i = 1/sigma_i^2 */
        for (i = 0; i < nabs; i++)
        {
            tau[i] = 1./dsqr(waxs_intensity_sigma(wr, wt, IcalcVar, i));
        }

        /* Get fitting parameters f/c, or purely f, or no fitting (dependig on waxs-Iexp-fit) */
        maximum_likelihood_est_fc_generic(wr, Icalc, wt->Iexp, tau, wt->nq, &wt->f_ml, &wt->c_ml, &sumSquaredResiduals);

        nShannonFactorUsed = ( wr->bDoNotCorrectForceByNindep ? 1. : wt->nShannon );

        /* overall factor for pot and forces
         * Update in Dec. 2017. Multiply with Nindep, such that we follow a Bayesian logic if the force constant is fc == 1
         */
        fact0 = fForceSwitch * wt->fc * wr->kT * sqr(fact_target_switch) * nShannonFactorUsed / wt->nq;

        /* At the beginning of the simulation (t < waxs_tau), the Icalc may be negative simply because Icalc is not yet
           converged. However, negative Icalc would lead to an error when coupling on a log scale. To avoid this error,
           allow negative Icalc while t < waxs_tau, while fact0 is anyway still zero: */
        bAllowNegativeIcalc = (fForceSwitch == 0.);

        /* Get delta_I = I_calc - I_exp and sigmas */
        for (i = 0; i < nabs; i++)
        {
            diffI[i] = waxs_intensity_Idiff(wr, wt, i, bAllowNegativeIcalc, TRUE);
            sigma[i] = waxs_intensity_sigma(wr, wt, IcalcVar, i);
        }

        /* WAXS potential energy is only computed on the master and =0 on all slave nodes */
        if (MASTER(cr) && wr->bCalcPot)
        {
            if (wr->wo->fpPot[t])
            {
                fprintf(wr->wo->fpPot[t], "%8g", simtime);
            }
            /* Recomputing the residuals, to make sure they agree with the result from maximum_likelihood_est_fc_generic() */
            sumSquaredResidualsRecomputed = 0;
            for (i = 0; i < nabs; i++)
            {
                /*
                 * The SAXS/SANS potential, and cumulative averages
                 *
                 *  The factor of 0.5 comes from the 1/2 in Eq. 17, Shevchuk & Hub, Plos Comp Biol 2017.
                 *  This factor ensures that we follow the Bayesian refinemnt with a force constant of fc=1.
                 */
                thisv            = 0.5 * fact0 * dsqr(diffI[i]/sigma[i]);
                wt->vLast       += thisv;
                wd->vAver [i]    = d_accum_avg(wd->vAver [i], (double)thisv, fac1, fac2 );
                wd->vAver2[i]    = d_accum_avg(wd->vAver2[i], dsqr(thisv),   fac1, fac2 );
                sumSquaredResidualsRecomputed += dsqr(diffI[i]/sigma[i]);

                if (wr->wo->fpPot[t])
                {
                    fprintf(wr->wo->fpPot[t], "  %8g", thisv);
                }
            }
            if (wr->wo->fpPot[t])
            {
                fprintf(wr->wo->fpPot[t], "\n");
                fflush(wr->wo->fpPot[t]);
            }
            printf("Type %1d (%s): Fit params: f = %8g  c = %8g  mean residual = %8g  V [kJ/mol] = %8g\n", t, wt->saxssansStr,
                   wt->f_ml, wt->c_ml, sqrt(sumSquaredResiduals/wt->nq), wt->vLast);
            if (wr->potentialType == ewaxsPotentialLINEAR &&
                fabs(sumSquaredResidualsRecomputed/nabs - sumSquaredResiduals/nabs) > 0.2 && wr->waxsStep > 30)
            {
                /* Residuals computed in two ways may slightly differ, because we update the
                   f_ml*sigma[exp] after fitting f_ml */
                fprintf(stderr, "\nWARNING, the residuals computed in two ways are inconsistent. This should not happen:\n"
                        "Via max-likelihood calculation: %g  ---  directly: %g\n",
                        sqrt(sumSquaredResidualsRecomputed/nabs), sqrt(sumSquaredResiduals/nabs));
            }

            /* Add potential of this scattering type to total potential */
            wr->vLast += wt->vLast;
        }

        /* Each node computes all forces at the moment */
        if (wr->bCalcForces)
        {
            /*  At negative contrast, moving an atom by +deltaR will shift the contrast by -deltaR,
             *  since the empty space will be filled by water. This should approximately lead to
             *  a reduction of the force relative to the contrast. For more details, see the comment
             *  at the function updateSoluteContrast().
             */
            if (wr->bDoNotCorrectForceByContrast)
            {
                contrastFactorUsed = 1;
            }
            else
            {
                contrastFactorUsed = wt->contrastFactor;
            }

            /* Should this be parallelized? - But then we need another communication. What is faster? */
            for (p = 0; p < nprot; p++)
            {
                clear_rvec(ftmp);
                for (i = 0; i < nabs; i++)
                {
                    /*
                     *   We used to have a factor of 2 here - this was removed to follow the Bayesian formalism
                     *   if the force constant is fc=1.
                     */
                    fact1 = -fact0 * contrastFactorUsed * diffI[i] / dsqr(sigma[i]);
                    if (wr->potentialType == ewaxsPotentialLOG)
                    {
                        fact1 /= Icalc[i];
                    }

                    l = p * nabs + i;
                    for (d = 0; d < DIM; d++)
                    {
                        ftmp[d] += fact1 * wd->dkI[l][d];
                    }
                }

                if (wr->bRotFit)
                {
                    copy_rvec(ftmp, ftmp2);
                    mvmul(*Rinv, ftmp2, ftmp);
                }

                /* Store current force of this scattering type, and add to the total force */
                copy_rvec(ftmp, wt->fLast[p]);
                rvec_inc(wr->fLast[p], ftmp);
            }
        }
        sfree(diffI);
        sfree(sigma);
        sfree(tau);
    }
    waxs_debug("Leaving waxs_md_pot_forces()");
}

void
do_waxs_md_low (t_commrec *cr, rvec x[], double simtime,
        gmx_large_int_t step, t_waxsrec *wr,
        gmx_mtop_t *mtop, matrix box, int ePBC, gmx_bool bDoLog)
{
    waxs_debug("do_waxs_md_low");
    int              i, j, qstart, qhomenr, d, p, t;
    int              nDeuter, nHyd;
    rvec             tmpvec, tmpvec2, cent, *x_red = NULL;
    t_waxs_datablock wd = NULL;
    t_waxsrecType   *wt = NULL;
    char             fn[1024];
    real             nelecA = 0., nelecB = 0., v = 0., densA, densB = 0., scaleB, solDen, tmp;
    double           Rg, time_cpu_this_step, time_ms, vol;
    gmx_bool         bTmp, bTimeOK, bVerbose;
    static gmx_bool  bFirst = TRUE;
    matrix           Rinv;

#ifdef RERUN_CalcForces
    wr->bCalcForces = TRUE ;
#endif

    waxs_debug("Entering do_waxs_md_low()");
    bVerbose = (wr->debugLvl > 1);

    /* Doing this step? */
    bTimeOK = ( (wr->calcWAXS_begin == -1) || ((simtime + 1e-5) >= wr->calcWAXS_begin) ) &&
              ( (wr->calcWAXS_end   == -1) || ((simtime - 1e-5) <= wr->calcWAXS_end  ) );
    if (PAR(cr))
    {
        gmx_bcast(sizeof(gmx_bool), &bTimeOK, cr);
    }
    if (!bTimeOK)
    {
        if (MASTER(cr))
        {
            fprintf(wr->wo->fpLog, "Skipping time %f - outside of time boundaries: begin %g, end %g\n",
                    simtime, wr->calcWAXS_begin, wr->calcWAXS_end);
            printf("Skipping time %f - outside of time boundaries: begin %g, end %g\n",
                    simtime, wr->calcWAXS_begin, wr->calcWAXS_end);
        }
        clear_vLast_fLast(wr);
        return;
    }

    waxsTimingDo(wr, waxsTimeStep, waxsTimingAction_start, 0, cr);

    /* Doing solvent in this frame? */
    wr->bDoingSolvent = (!wr->bVacuum && wr->waxsStep < wr->nfrsolvent);
    if (!wr->bVacuum && wr->waxsStep == wr->nfrsolvent && MASTER(cr))
    {
        printf("\nNote: Reached the requested nr of pure solvent frames (%d)."
                "\n      Will not compute any more pure solvent scattering amplitudes.\n\n", wr->nfrsolvent);
    }

    /* = = = = = MASTER does all the preparations before bcast. = = = = = */
    if (MASTER(cr))
    {
        waxsTimingDo(wr, waxsTimePrepareSolute, waxsTimingAction_start, 0, cr);

        printf("\nComputing SWAXS/SANS intensity in step ");
        fprintf(wr->wo->fpLog, "\nDoing WAXS intensity in step ");
        printf(gmx_large_int_pfmt, step);
        fprintf(wr->wo->fpLog, gmx_large_int_pfmt, step);
        printf(" (time %6g ps). SWAXS step %d\n",     simtime, wr->waxsStep);
        fprintf(wr->wo->fpLog, " (time %g). SWAXS step %d\n", simtime, wr->waxsStep);

        /* Do the required steps with the solute before getting the solvation shell:
           1. Make solute/protein whole.
           2. Get bounding sphere around solute
           3. Move this sphere to the center of the box - this maximises the solvent coverage within the unit cell.
           3. Put all solvent atoms into the compact box.
           4. Obtain the vector between box_center and solute COG. This will become the newBoxCent a the end.
           5. Shift fit-group COG to origin.
           6. If requested, do a rotation of the solute (using the fitgroup atoms such as Backbone),
              and return the inverse rotation matrix.
           7. After rotation, move the solute COG to the origin.
         */
        waxs_prepareSoluteFrame(wr, mtop, x, box, ePBC, wr->wo->fpLog, Rinv);

        waxsTimingDo(wr, waxsTimePrepareSolute, waxsTimingAction_end, 0, cr);

        /* Pick the pure water box and shift coordinates it to the center of the protein box */
        if (wr->bDoingSolvent)
        {
            waxsTimingDo(wr, waxsTimePrepareSolvent, waxsTimingAction_start, 0, cr);

            /* Shift pure-solvent frame onto envelope and store in 'ws->xPrepared' */
            preparePureSolventFrame(wr->waxs_solv, wr->waxsStep, wr->wt[0].envelope, wr->debugLvl);

            waxsTimingDo(wr, waxsTimePrepareSolvent, waxsTimingAction_end, 0, cr);
        }
        /* = = = = = All atoms in place for buffer subtraction = = = = = */

        /* = = = = = Obtain the atoms inside the envelope = = = = =
           Search solvent molecules in solvation layer and in the excluded volume and
           store in index_A (protein+solvation layer) and index_B (excluded solvent),
           respectively. */
        if (! wr->bVacuum)
        {
            get_solvation_shell(x, wr->waxs_solv->xPrepared, wr, mtop, wr->waxs_solv->mtop, ePBC, box);

            fprintf(wr->wo->fpLog, "%d atoms in protein\n",         wr->nindA_prot);
            fprintf(wr->wo->fpLog, "%d atoms in solvation shell\n", wr->isizeA-wr->nindA_prot);
            fprintf(wr->wo->fpLog, "%d atoms in excluded volume\n", wr->isizeB);

            /* Check if solvation layer and excluded solvent fill roughly the same volume */
            if (wr->bDoingSolvent)
            {
                check_selected_solvent(x, wr->waxs_solv->xPrepared,
                                       wr->indexA, wr->isizeA, wr->indexB, wr->isizeB, TRUE);
            }
        }
        /* for wr->bVacuum==TRUE, nothing must be done because indexA and isizeA were initialized to
           the protein in prep_md_scattering_data() */

        /*
         * Get number of electrons and electron density of A and B systems
         */
        densA = droplet_density(wr, wr->indexA, wr->isizeA, wr->nElectrons, &nelecA);
        fprintf(wr->wo->fpLog, "Density of A [e/nm3] = %10g  (%g electrons)\n", densA, nelecA);
        fprintf(wr->wo->fpLog, "%g electrons in solute+solvation layer, i.e., A(q=0) = %g\n",
                nelecA, nelecA*nelecA);

        if (wr->bDoingSolvent)
        {
            densB = droplet_density(wr, wr->indexB, wr->isizeB,  wr->waxs_solv->nElectrons, &nelecB);
            fprintf(wr->wo->fpLog, "Density of B [e/nm3] = %10g (%g electrons)\n", densB, nelecB);
            fprintf(wr->wo->fpLog, "%g electrons in excluded solvent,       i.e., B(q=0) = %g\n",
                    nelecB, nelecB*nelecB);
        }

        /* If applicable, calculate RMSD of fit group to past ensemble. This can be used to
           dampten the forces after a conformational transition, so we don't "overshoot". */
        if (wr->bDampenForces)
        {
            if (!bFirst)
            {
                update_waxs_eavrmsds(wr);
            }
            if (wr->wo->fpExpRMSD)
            {
                fprintf(wr->wo->fpExpRMSD,"%g %g %g %g\n",
                        simtime, wr->wrmsd->rmsd_now, wr->wrmsd->rmsd_av, wr->wrmsd->sd_now);
            }
        }

        /* = = = = = Statistics: MASTER calculates solvation shell stats assuming equal averages = = = = = */
        /* Get Radius of Gyration of Solute (without solvation layer) */
        Rg = soluteRadiusOfGyration(wr, x);
        UPDATE_CUMUL_AVERAGE(wr->RgAv, wr->waxsStep+1, Rg);

        /* Averages of nr of atoms in surface layer and excluded volume */
        UPDATE_CUMUL_AVERAGE(wr->nAtomsLayerAver,   wr->waxsStep + 1, wr->isizeA - wr->nindA_prot);
        UPDATE_CUMUL_AVERAGE(wr->nAtomsExwaterAver, wr->waxsStep + 1, wr->isizeB);
        UPDATE_CUMUL_AVERAGE(wr->nElecAvA,          wr->waxsStep + 1, nelecA);
        UPDATE_CUMUL_AVERAGE(wr->nElecAv2A,         wr->waxsStep + 1, nelecA*nelecA);
        UPDATE_CUMUL_AVERAGE(wr->nElecAv4A,         wr->waxsStep + 1, nelecA*nelecA*nelecA*nelecA);
        if (wr->bDoingSolvent)
        {
            UPDATE_CUMUL_AVERAGE(wr->nElecAvB,  wr->waxsStep + 1, nelecB);
            UPDATE_CUMUL_AVERAGE(wr->nElecAv2B, wr->waxsStep + 1, nelecB*nelecB);
            UPDATE_CUMUL_AVERAGE(wr->nElecAv4B, wr->waxsStep + 1, nelecB*nelecB*nelecB*nelecB);
        }
        if (!wr->bVacuum)
        {            // printf("Current volume of box = %g nm3\n", det(wr->local_state->box));
            solDen = (wr->nElecTotA - nelecA) / (det(wr->local_state->box) - droplet_volume(wr));
            UPDATE_CUMUL_AVERAGE(wr->solElecDensAv,  wr->waxsStep + 1, solDen);
            UPDATE_CUMUL_AVERAGE(wr->solElecDensAv2, wr->waxsStep + 1, solDen*solDen);            /* needed for fix_solvent_density */
            UPDATE_CUMUL_AVERAGE(wr->solElecDensAv4, wr->waxsStep + 1, solDen*solDen*solDen*solDen);
            wr->soluteVolAv = droplet_volume(wr) - (wr->nElecAvA - wr->nElecProtA)/wr->solElecDensAv;
        }
        /* = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = */

    }
    /* = = = = = END of MASTER preparations = = = = = */

    /* Distribute atoms inside the envelope over the nodes */
    if (PAR(cr))
    {
        gmx_bcast(sizeof(int), &wr->isizeA,        cr);
        gmx_bcast(sizeof(int), &wr->isizeB,        cr);
        gmx_bcast(sizeof(int), &wr->indexA_nalloc, cr);
        gmx_bcast(sizeof(int), &wr->indexB_nalloc, cr);
        gmx_bcast(sizeof(double), &wr->soluteVolAv,   cr);
    }

    /* Broadcast the Rotational Matrix for force rotation. */
    if (wr->bRotFit && PAR(cr))
    {
        gmx_bcast(sizeof(matrix), Rinv, cr);
    }

    /* Possilby expand allocated memory for atomic coordinates and scattering types */
    srenew(wr->atomEnvelope_coord_A, wr->indexA_nalloc);
    for (t = 0; t < wr->nTypes; t++)
    {
        srenew(wr->wt[t].atomEnvelope_scatType_A, wr->indexA_nalloc);
    }
    if (wr->bDoingSolvent)
    {
        srenew(wr->atomEnvelope_coord_B, wr->indexB_nalloc);
        for (t = 0; t < wr->nTypes; t++)
        {
            srenew(wr->wt[t].atomEnvelope_scatType_B, wr->indexB_nalloc);
        }
    }


    for (t = 0; t < wr->nTypes; t++)
    {
        /* Compute Fourier Transforms of solvent density, used for solvent density correction in
           solute system */
        if (wr->bFixSolventDensity || wr->bScaleI0)
        {
            update_envelope_solvent_density(wr, cr, x, t);
        }
    }

#ifdef GMX_GPU
    /* The envelope of the solvent density is needed on the GPU already in the first WAXS-step.
     * Attention: On the GPU we never recalculate the FT we only pass the values of the solvent FT by reference!
    */
    if (wr->bUseGPU && wr->bRecalcSolventFT_GPU)
    {
        update_envelope_solvent_density_GPU( wr );
    }
#endif

    waxsTimingDo(wr, waxsTimeScattAmplitude, waxsTimingAction_start, 0, cr);

    /*
     *  The main loop over the scattering groups (xray, neutron1, neutron2,...)
     */
    for (t = 0; t < wr->nTypes; t++)
    {
        wt = &wr->wt[t];
        wd = wt->wd;

        if (MASTER(cr))
        {
            nDeuter = nHyd = 0;
            for (i = 0; i < wr->isizeA; i++)
            {
                j = wr->indexA[i];
                if (wt->type == escatterXRAY)
                {
                    assert_scattering_type_is_set(wr->cmtypeList[j], wt->type, j);
                    wt->atomEnvelope_scatType_A[i] = wr->cmtypeList[j];
                }
                else
                {
                    /* get the NSL type, taking this deuterium conc into account */
                    assert_scattering_type_is_set(wr->nsltypeList[j], wt->type, j);

                    wt->atomEnvelope_scatType_A[i] = get_nsl_type(wr, t,  wr->nsltypeList[j]);
                    nDeuter += (wt->atomEnvelope_scatType_A[i] == wt->nnsl_table-1);
                    nHyd    += (wt->atomEnvelope_scatType_A[i] == wt->nnsl_table-2);

                    // printf("atom %d, nsltype = %d - t = %d\n", j, wr->redA[i].t, t);
                }
                copy_rvec(x[j], wr->atomEnvelope_coord_A[i]);
            }
            UPDATE_CUMUL_AVERAGE(wt->n2HAv_A,  wr->waxsStep+1, nDeuter);
            UPDATE_CUMUL_AVERAGE(wt->n1HAv_A,  wr->waxsStep+1, nHyd);
            UPDATE_CUMUL_AVERAGE(wt->nHydAv_A, wr->waxsStep+1, nHyd+nDeuter);
        }
        if (PAR(cr))
        {
            gmx_bcast(wr->isizeA*sizeof(rvec), wr->atomEnvelope_coord_A,    cr);
            gmx_bcast(wr->isizeA*sizeof(int),  wt->atomEnvelope_scatType_A, cr);
        }

        qstart  = wt->qvecs->qstart;
        qhomenr = wt->qvecs->qhomenr;

        /* B-part: scattering of excluded solvent */
        if (wr->bDoingSolvent)
        {
            if (MASTER(cr))
            {
                nDeuter = nHyd = 0;
                for (i = 0; i < wr->isizeB; i++)
                {
                    j = wr->indexB[i];
                    if (wt->type == escatterXRAY)
                    {
                        assert_scattering_type_is_set(wr->waxs_solv->cmtypeList[j], wt->type, j);
                        wt->atomEnvelope_scatType_B[i] = wr->waxs_solv->cmtypeList[j];
                    }
                    else
                    {
                        /* get the NSL type, taking this deuterium conc into account */
                        assert_scattering_type_is_set(wr->waxs_solv->nsltypeList[j], wt->type, j);

                        wt->atomEnvelope_scatType_B[i] = get_nsl_type(wr, t, wr->waxs_solv->nsltypeList[j]);
                        nDeuter += (wt->atomEnvelope_scatType_B[i] == wt->nnsl_table-1);
                        nHyd    += (wt->atomEnvelope_scatType_B[i] == wt->nnsl_table-2);

                    }
                    copy_rvec(wr->waxs_solv->xPrepared[j], wr->atomEnvelope_coord_B[i]);
                }
                UPDATE_CUMUL_AVERAGE(wt->n2HAv_B,  wr->waxsStep+1, nDeuter);
                UPDATE_CUMUL_AVERAGE(wt->n1HAv_B,  wr->waxsStep+1, nHyd);
                UPDATE_CUMUL_AVERAGE(wt->nHydAv_B, wr->waxsStep+1, nHyd+nDeuter);
            }

            if (PAR(cr))
            {
                gmx_bcast(wr->isizeB*sizeof(rvec), wr->atomEnvelope_coord_B, cr);
                gmx_bcast(wr->isizeB*sizeof(int), wt->atomEnvelope_scatType_B, cr);
            }


            /* Compute instantanous scattering amplitude of pure-solvent sytem, B(\vec{q}) */
            if (wr->bUseGPU)
            {
#ifdef GMX_GPU
                compute_scattering_amplitude_cuda (wr,cr,  t, wd->B, wr->atomEnvelope_coord_B , wt->atomEnvelope_scatType_B , wr->isizeB, 0,
                                                           qhomenr,
                                                           wt->type == escatterXRAY    ? wt->aff_table : NULL,
                                                           wt->type == escatterNEUTRON ? wt->nsl_table : NULL,
                                                           wt->qvecs->nabs,
                                                           wd->normA, wd->normB, wr->scale, wr->bCalcForces, wr->bFixSolventDensity,
                                                           &time_ms);
#endif
            }
            else
            {
                compute_scattering_amplitude (wd->B, wr->atomEnvelope_coord_B, wt->atomEnvelope_scatType_B ,  wr->isizeB, 0,
                                                     wt->qvecs->q+qstart, wt->qvecs->iTable+qstart, qhomenr,
                                                     wt->type == escatterXRAY    ? wt->aff_table : NULL,
                                                     wt->type == escatterNEUTRON ? wt->nsl_table : NULL,
                                                     NULL, &time_ms);
            }

            waxsTimingDo(wr, waxsTimeOneScattAmplitude, waxsTimingAction_add, time_ms/1000, cr);
        }

        /* Compute instantanous scattering amplitude of protein + hydration layer, A(\vec{q}) */
        if (wr->bUseGPU)
        {
#ifdef GMX_GPU
            compute_scattering_amplitude_cuda (wr,cr, t, wd->A , wr->atomEnvelope_coord_A, wt->atomEnvelope_scatType_A, wr->isizeA, wr->nindA_prot,
                                                      qhomenr,
                                                      wt->type == escatterXRAY    ? wt->aff_table : NULL,
                                                      wt->type == escatterNEUTRON ? wt->nsl_table : NULL,
                                                      wt->qvecs->nabs,
                                                      wd->normA , wd->normB , wr->scale , wr->bCalcForces, wr->bFixSolventDensity,
                                                      &time_ms);
#endif
        }
        else
        {
            compute_scattering_amplitude (wd->A, wr->atomEnvelope_coord_A, wt->atomEnvelope_scatType_A , wr->isizeA, wr->nindA_prot,
                                                 wt->qvecs->q+qstart, wt->qvecs->iTable+qstart, qhomenr,
                                                 wt->type == escatterXRAY    ? wt->aff_table : NULL,
                                                 wt->type == escatterNEUTRON ? wt->nsl_table : NULL,
                                                 wd->dkAbar,  &time_ms);
        }

        waxsTimingDo(wr, waxsTimeOneScattAmplitude, waxsTimingAction_add, time_ms/1000, cr);

        /* Extra-check: Make sure that A[0]/B[0] is in agreement with the total number of electrons in the envelope */
        if (wt->qvecs->qstart == 0 && norm2(wt->qvecs->q[0]) == 0.0 && wt->type == escatterXRAY && wr->bUseGPU == FALSE)
        {
            if (fabs(wd->A[0].re - nelecA)/nelecA > 1e-6 && MASTER(cr))
            {
                gmx_fatal(FARGS, "A(q=0) = %g is inconsistent with # of electrons in A: %g\n", wd->A[0].re,
                          nelecA);
            }
            if (wr->bDoingSolvent && MASTER(cr) && fabs(wd->B[0].re - nelecB)/nelecB > 1e-6)
            {
                gmx_fatal(FARGS, "B(q=0) is inconsistent with # of electrons in B: %g / %g\n", wd->B[0].re,
                          nelecB);
            }
        }

        /* Upate total number of electrons or NSLs of A and B systems */
        tmp = number_of_electrons_or_NSLs(wt, wt->atomEnvelope_scatType_A, wr->isizeA, mtop );
        UPDATE_CUMUL_AVERAGE(wd->avAsum, wr->waxsStep + 1, tmp);
        if (wr->bDoingSolvent)
        {
            tmp = number_of_electrons_or_NSLs(wt, wt->atomEnvelope_scatType_B, wr->isizeB, mtop);
            UPDATE_CUMUL_AVERAGE(wd->avBsum, wr->waxsStep + 1, tmp);
        }

        wd = NULL;
        wt = NULL;
    }

    waxsTimingDo(wr, waxsTimeScattAmplitude, waxsTimingAction_end, 0, cr);

    waxsTimingDo(wr, waxsTimeScattUpdates, waxsTimingAction_start, 0, cr);

    waxsTimingDo(wr, waxsTimeSolvDensCorr, waxsTimingAction_start, 0, cr);
    for (t = 0; t < wr->nTypes; t++)
    {
        if (wr->bFixSolventDensity && !wr->bVacuum)
        {
            /* This function we need also to execute in case of GPU calculation to get estimation of added electrons etc..
             * However, in case of GPU calculation the actual fix_solvent_density correction will be done on the GPU!"
             */
            fix_solvent_density(wr, cr, t);
        }
    }
    waxsTimingDo(wr, waxsTimeSolvDensCorr, waxsTimingAction_end, 0, cr);

    if (MASTER(cr) && wr->bGridDensity)
    {
        update_envelope_griddensity(wr, x);
    }

    /* Update average of scattering amplitudes A(q) and B(q) */
    update_AB_averages(wr);

    waxsTimingDo(wr, waxsTimeComputeIdkI, waxsTimingAction_start, 0, cr);
    if (wr->bScaleI0)
    {
        scaleI0_getAddedDensity(wr, cr);
    }
    waxsTimingDo(wr, waxsTimeComputeIdkI, waxsTimingAction_end, 0, cr);

    /* Compute I(q) and gradients of I(q) wrt. atomic positions */
    calculate_I_dkI(wr, cr);

    /* Updatting estimates for the solute contrast wrt. to the buffer */
    updateSoluteContrast(wr, cr);

    waxsTimingDo(wr, waxsTimeScattUpdates, waxsTimingAction_end, 0, cr);

#ifndef RERUN_CalcForces
    if (wr->bCalcForces || wr->bCalcPot)
    {
        waxsTimingDo(wr, waxsTimePotForces, waxsTimingAction_start, 0, cr);

        if (wr->ewaxsaniso != ewaxsanisoNO)
        {
            gmx_fatal(FARGS, "Computing WAXS potentials or forces is not yet supported with anisotropic patterns\n");
        }

        /* Clear old potential and forces */
        clear_vLast_fLast(wr);

        /* Update number of independent data points (Shannon-Nyquist theorem) */
        nIndep_Shannon_Nyquist(wr, x, cr, wr->waxsStep == 0 && (MASTER(cr)));

        /*
         * GIBBS SAMPLING:
         *    - Sample the uncertainty factor for the buffer density using Gibbs sampling
         *    - With ensemble refinment, in addition sample the weights of the different states
         *
         * On exit: New buffer epsilon and state weights present, already broadcastet over the nodes
         *          wr->ensembleSum_I and wr->ensembleSum_Ivar are updated.
         */
        if (wr->ewaxs_ensemble_type == ewaxsEnsemble_BayesianOneRefined || wr->bBayesianSolvDensUncert)
        {
            bayesian_gibbs_sampling(wr, simtime, cr);
        }

        /* Compute SAXS-drived potential and forces */
        waxs_md_pot_forces(cr, wr, simtime, &Rinv);

        if (MASTER(cr) && wr->bCalcForces)
        {
            write_total_force_torque(wr, x);
        }
        if (wr->debugLvl>1 && wr->bRotFit)
        {
            fprintf(stderr, "Node rotations: Rinv(row1) = %g %g %g\n", Rinv[0][0], Rinv[0][1], Rinv[0][2] );
        }

        waxsTimingDo(wr, waxsTimePotForces, waxsTimingAction_end, 0, cr);
    }
#endif

    /* Output to log file */
    if (bDoLog && MASTER(cr))
    {

        for (t = 0; t < wr->nTypes; t++)
        {
            wt = &wr->wt[t];

            fprintf(wr->wo->fpSpectra[t], "\n&\n# Intensity %d at simulation step ", wr->waxsStep);
            fprintf(wr->wo->fpSpectra[t], gmx_large_int_pfmt, step);

            write_intensity(wr->wo->fpSpectra[t], wr, t);
            if (wr->bCalcPot && wr->weightsType != ewaxsWeightsUNIFORM)
            {
                write_stddevs(wr, step, t);
            }
            if ((wr->waxsStep % 10) == 0)
            {
                fflush(wr->wo->fpSpectra[t]);
            }

            if (WAXS_ENSEMBLE(wr))
            {
                /* Write ensemble-averaged I(q) */
                fprintf(wr->wo->fpSpecEns[t], "\n&\n# Intensity %d at simulation step ", wr->waxsStep);
                fprintf(wr->wo->fpSpecEns[t], gmx_large_int_pfmt, step);
                fprintf(wr->wo->fpSpecEns[t], "\n@type xydy\n");
                for (i = 0; i < wt->nq; i++)
                {
                    fprintf(wr->wo->fpSpecEns[t], "%8g  %12g %12g\n", wt->qvecs->abs[i], wt->ensembleSum_I[i],
                            sqrt(wt->ensembleSum_Ivar[i]));
                }
                if ((wr->waxsStep % 10) == 0)
                {
                    fflush(wr->wo->fpSpecEns[t]);
                }
            }
        }

        if (wr->bPrintForces)
        {
            snew(x_red, wr->nindA_prot);
            if (wr->bRotFit)
            {
                for (j = 0; j < wr->nindA_prot; j++)
                {
                    /* Rotate x to the original frame */
                    rvec_sub(x[wr->indA_prot[j]], wr->origin, tmpvec);
                    /* v' = R(v0-c0)+c' ; Rotate and re-center. */
                    mvmul(Rinv, tmpvec, tmpvec2);
                    rvec_add(tmpvec2, wr->origin, x_red[j]);
                }
            }
            else
            {
                for (j = 0; j<wr->nindA_prot; j++)
                {
                    copy_rvec(x[wr->indA_prot[j]], x_red[j]);
                }
            }

            fprintf(stderr,"Writing positions and forces of frame %i into trr file\n", wr->waxsStep);
            for (t = 0; t < wr->nTypes; t++)
            {
                /* Write coordinate and forces from all scattering groups into separate trr files */
                fwrite_trn(wr->wo->xfout[t], wr->waxsStep, simtime, wr->wt[t].vLast, box, wr->nindA_prot, x_red, NULL, wr->wt[t].fLast );
            }
            sfree(x_red);
        }

        if (wr->bDampenForces)
        {
            printf("WAXS step & RMSD to ensemble: %d %g , %g +- %g\n",
                    wr->waxsStep, wr->wrmsd->rmsd_now, wr->wrmsd->rmsd_av,
                    wr->wrmsd->sd_now);
        }
    }

    if (PAR(cr)) gmx_barrier(cr);

    waxsTimingDo(wr, waxsTimeStep, waxsTimingAction_end, 0, cr);

    if (MASTER(cr) && wr->wo->fpLog && wr->waxsStep >= WAXS_STEPS_RESET_TIME_AVERAGES)
    {
        waxsTimingWriteLast(wr->compTime, wr->wo->fpLog);
    }

    wr->waxsStep++;
    bFirst = FALSE;

    waxs_debug("Leaving do_waxs_md_low()");
}
