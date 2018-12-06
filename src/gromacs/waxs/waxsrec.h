/* -*- mode: c; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; c-file-style: "stroustrup"; -*-
 *
 *
 *                This source code is part of
 *
 *                 G   R   O   M   A   C   S
 *
 *          GROningen MAchine for Chemical Simulations
 *
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2008, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 *
 * For more info, check our website at http://www.gromacs.org
 *
 * And Hey:
 * Gallium Rubidium Oxygen Manganese Argon Carbon Silicon
 */

#ifndef _waxsrec_h
#define _waxsrec_h

#include "simple.h"
// #include "topology.h"
#include "state.h"
#include "oenv.h"
#include "gmx_envelope.h"
#include "gmx_random.h"

#include "../gmxcomplex.h"


#ifdef __cplusplus
extern "C" {
#endif

/* This could be abstract and get declared in sfactor.c */
typedef struct {
    int nabs  ;  /* number of distinct absolutes */
    real *abs ;  /* abs. values for redundant reference */
    int  *ref ;  /* table reference for each value. Designed to support remapping.  */
    int  *ind ;  /* Indices of first vector for each |x| terminated by abs. of size nabs + 1 */
    int  n    ;  /* total number of distinct vectors */
    rvec *q   ;  /* distinct vectors of size max(ind), e.g. nabs*J */
    int *iTable; /* array of size n. Used in atomic form factor table aff_table[sftype][iTable]
                    (NULL if this is a neutron grp) */
    int qstart;  /* qstart <= i < qstart+qhomenr indices of q's calculated on this node */
    int qhomenr;
} t_spherical_map;

/* The rvec struct of this sphere is sorted such the the number of vectors is defined by the differnce in indices.
 *    |q|[0]   |     |q|[1]  |... |q|[nabs]
 *    w
 *    ref[0]   |     ref[1]  |... ref[nabs]
 * ind[0]   ind[1]         ind[2] ... ind[nabs+1]
 *   |         |             |           |
 * x[0]x[1]--x[i1]x[i1+1]--x[i2]  ...  x[nx]
 * */

/* Abstract data type for WAXS calculations (declared in waxsmd.c) */
typedef struct waxs_datablock *t_waxs_datablock;

/* Abstract type for on-the-fly calculation of RMSD, can be used todampen
   the forces after a conformational transition */
typedef struct waxs_eavrmsd *t_waxs_eavrmsd;

/* WAXS computing time stuff */
typedef struct waxsTiming *t_waxsTiming;

/* WAXS output stuff */
typedef struct waxs_output *t_waxs_output;

/* GPU data which need to be allocated just once */
typedef struct {
    int      *iTable_GPU;
    real     *aff_table_linearized;
    real     *q_linearized;
    gmx_bool useGPU;
} t_waxs_GPU;

/* Abstract data type for excluded solvent. This was originally declared in waxsmd.c, but now we declare it here.
   Hence, we could make this non-abstract in the future. */
typedef struct waxs_solvent *t_waxs_solvent;

struct waxs_solvent {
    gmx_mtop_t *mtop;              /* water topology */
    int        *cmtypeList;        /* Cromer-Mann types, array of size mtop->natoms */
    int        *nsltypeList;       /* NSL types, array of size mtop->natoms */
    int         nframes;           /* nr of frames in **x */
    rvec      **x;                 /* coordinates of water frames */
    rvec       *xPrepared;         /* Allocated array for coordinates shifted onto the envelope */
    matrix     *box;               /* boxes of water frames */
    real        avInvVol;          /* Average inverse volume */
    real        avDensity;         /* Average density, computed from complete box (e/nm^3) */
    real        avNumerDensity;    /* Average number density (1/nm^3) */
    real        avDropletDensity;  /* Average density of droplet (hopefully the same as avDensity) */
    // real        densCorr;       /* homogeneous density added to fix density */
    double     *nElectrons;        /* Number of electrons per atom, size mtop->natoms */
    real        nelec;             /* Total number of electrons (from Cromer-Mann parameters) */
    int         ePBC;
    gmx_bool    bHaveNeutron;      /* Neutron scattering lengths found in solvent tpr file */
    double     *avScattLenDens;    /* Average scattering length density (in electrons or NSLs per nm3)
                                      for each scattering type (SAXS and SANS with different D2O) */
};


/* WAXS record stuff that is specific for each coupling type, such as each xray or neutron type */
typedef struct {
    int type;                /* escatterXRAY or escatterNEUTRON */
    t_waxs_datablock wd;     /* WAXS data averages during MD    */
    char saxssansStr[10];    /* String holding "SAXS" or "SANS" */
    char scattLenUnit[15];   /* String holding "electrons" or "NSLs" */

    int *atomEnvelope_scatType_A; /* Scattering type of each molecule */
    int *atomEnvelope_scatType_B; /* Scattering type of each molecule */

    real fc;                  /* Extra overall force constant (=1 by default).   */
    int  nq;                 /* nr of q values, start and end */
    real minq, maxq;
    real nShannon;           /* number of independent data points according to Shannon-Nyquist */
    t_spherical_map *qvecs ; /* Collection of vectors relative to box and reference to sf_table */

    real vLast;              /* Last potential, this waxs type only */
    rvec *fLast;             /* Last forces,    this waxs type only */

    double *Iexp;            /* Intensity I(q) to which we couple (e.g. experimental I)  */
    double *Iexp_sigma;      /* Uncertainty (or coupling strength) of I_exp */

    int naff_table;          /* Nr of elements in atomic form factor table */
    real **aff_table;        /* The atomic form factors vs. |q| for each type  (NULL if this is a neutron grp) */

    int  nnsl_table;         /* Nr of elements in nsl table */
    real *nsl_table;         /* The NSL, taking deuteration into account (NULL if this is a xray grp) */
    real deuter_conc;        /* Deuterrium concentration (for neutron scattering only) */

    double *Ipuresolv;       /* Intensity of pure solvent, used to add back oversubtracted buffer */
    double f_ml, c_ml;       /* With Bayesian MD: Maximum likelihood estimtates for offset and scale of calculated curve */
    real     targetI0;       /* When scaling I(q=0), scale to this I0 */

    double **ensemble_I;           /* I(q) and variance of fixed states of ensemble                          */
    double **ensemble_Ivar;
    double *ensembleSum_I;         /* I(q) and variance of the complete ensemble, weighted by current weights */
    double *ensembleSum_Ivar;
    double *ensemble_I_mh;    /* long array with all I(q) for all replicas r, ordering: I(0)(0), I(1)(0)... I(qabs)(0), I(qabs)(1)... I(qabs)(r)
                                 This approach is easier for MPI-Communications*/


    gmx_envelope_t envelope; /* pointer to envelope. Required by each waxsrecType because it stores the Fourier transform and
                                solvent density in the envelope, which are different for each type. */

    real givenSolventDensity;  /* Solvent density given in the mdp file (waxs-solvens).
                                  Electron density if type == xray
                                  NSL      density if type == neutron (dependent on deuter_conc)       */

    /* Stuff for accounting for the solute contrast wrt. to bulk solvent */
    double soluteContrast;          /* Approximate contrast of the solute (in electron or NSL density) */
    double contrastFactor;          /* SAXS/SANS-derived forces are reduced by this factor to account for the contrast.
                                       See comment at function updateSoluteContrast(). */
    double soluteSumOfScattLengths; /* Sum of scattering lengths of the solute (in electrons or NSLs)  */

    /* Average number of 2H, 1H, and total hydrogen inside the envelope */
    double n2HAv_A, n1HAv_A, nHydAv_A;
    double n2HAv_B, n1HAv_B, nHydAv_B;
} t_waxsrecType;


typedef struct {
    int             nTypes;                 /* Number of scattering groups: xray or neutron types */
    t_waxsrecType  *wt;                     /* records to variables that are specific to the scattering type (size nTypes) */

    t_waxs_GPU     *GPU_data;               /* Initialize GPU data */
    gmx_bool        bUseGPU;                /* To use GPU for scattering amplitude calculation */

    int             nstlog;                 /* Log output frequency for WAXS stuff          */

    /* Stuff for neutron scattering */
    gmx_bool        bDoingNeutron;          /* Doing (also) Neutron scattering              */
    gmx_bool        bStochasticDeuteration; /* Assining 1H and 2H randomly to deuteratable hydrogen atoms */
    real            backboneDeuterProb;     /* Probability that the backbone H is deuterated at 100% D2O. Defaul = 0.9 */

    real            solv_warn_lay;          /* Warn if layer gets thinner that this (with sphere) */
    int             nSolvWarn;              /* Nr of warnings due to thin solvation layer   */
    int             nWarnCloseBox;          /* Nr of warnings due to solute getting too close to box boundary  */
    real            xray_energy;            /* Energy of incoming Xray (keV)                */
    real            qbeam;                  /* q of incoming beam */
    int             ewaxsaniso;             /* Anisotropic averaging: No, yes, [cos(alpha)]^2 */

    /* Short-hand lists of Cromer-Mann scattering types and NSL scattering types */
    int            *cmtypeList;             /* array (size mtop->natoms) of atoms.atom[].cmtype    */
    int            *nsltypeList;            /* array (size mtop->natoms) of atoms.atom[].nsltype   */

    real            tau, tausteps;          /* coupling time constant, in ps and in waxs steps */
    int             nstcalc;                /* Calculating I(q) every these simulation steps   */
    int             nfrsolvent;             /* # of pure solvent structure factors to compute (-1 = take from xtc) */
    gmx_bool        bDoingSolvent;          /* ==FALSE if bVacuum or waxsStep >= nfrsolvent */
    gmx_bool        bVacuum;                /* No solvent scattering */
    double          scale;                  /* factor used for cumulative averages - allows non-weighted, instantaneous, exponential avarge */
    int             potentialType;          /* Enum for type of potential (log or linear scale) */
    int             weightsType;            /* Enum for coupling weights (uniform, experiment, experiment+calculated uncertainties) */
    int             bBayesianSolvDensUncert;/* Bayesian sampling of the uncertainty of the solvent density */

    real            t_target;               /* Switch the target curve linearly from current to the experimental curve within this time (e.g. 2ns)
                                               This reduces very large forces just after waxs_tau and allows for more gradual transitions.
                                               Implemented simply such that the force constant is scaled by (t/t_target)^2 if t < t_target.
                                            */
    int             ewaxs_Iexp_fit;         /* Fitting I(exp) to I(calc) on the fly, or maginializing the fitting parameters
                                               no / scale-and-offset / scale */

    real            kT;                     /* Boltzmann const times temperature (in units (kJ/mol K)).
                                               To get a natural energy scale for WAXS-potential */

    int             J;                      /* Nr of pointes on unit sphere in q-space for spherical quadrature  */
    int             Jmin;                   /* If J==0, determine J automatically by J = max(Jmin, alpha*(D*q)^2 */
    real            Jalpha;                 /* where alpha ~ 0.05 and D is the maximum diameter of the envelope.
                                               Jmin and Jalpha can be modified by environment variables.         */

    int             nAverI;                 /* Nr of previous I(q) used to compute uncertainty of I(q). Will use 2*tau/nstcalc. */
    int             stepCalcNindep;         /* calc number of independent D(q) very # steps */
    gmx_bool        bHaveNindep;            /* Did we already compute Nindep? */

    int             waxsStep;               /* Nr of WAXS calculations done so far. */
    double          calcWAXS_begin, calcWAXS_end;  /* Time to begin and end WAXS calculations - useful to avoid trjconv before rerun */

    /* Some statistics on atoms and electrons inside the envelope, and on density */
    double         *nElectrons;                     /* Array (size mtop->natoms) with number of electrons in each atom */
    double          nAtomsExwaterAver;              /* Av. number of atoms in excluded volume */
    double          nAtomsLayerAver;                /* Av. number of atoms in hydration layer */
    double          nElecAvA, nElecAv2A, nElecAv4A; /* Average # of electrons in A and B systems (and squares) */
    double          nElecAvB, nElecAv2B, nElecAv4B;
    double          nElecTotA;                                     /* # of electrons in system A */
    double          nElecProtA;                                    /* # of electrons in solute */
    double          solElecDensAv, solElecDensAv2, solElecDensAv4; /* Average of electron density of Solvent in System A */
    real            solElecDensAv_SysB;             /* Average solvent density in pure solvent box */
    double          RgAv;                           /* Average radius of gyration */

    gmx_bool        bDoingMD;       /* Within mdrun or, if not, within g_waxs */
    gmx_bool        bCalcForces;    /* Calculate forces form WAXS coupling (normal MD, not in rerun) */
    gmx_bool        bPrintForces;   /* Controls for printing out WAXS-forces.*/
    gmx_bool        bSwitchOnForce; /* Smooth switching-on of force until t=tau (so we have some averaging already) */
    gmx_bool        bCalcPot;       /* Calculate potential from WAXS-coupling (in normal MD, or in rerun if Iexp provided) */
    gmx_bool        bScaleI0;       /* Scaling I(q=0) to target pattern (assumes that the density of the protein is
                                       constant upon conformational transitions - useful at low contrast) */
    gmx_bool        bDoNotCorrectForceByContrast;  /* Do not scale the forces with factor = contrast / solute-density */
    gmx_bool        bDoNotCorrectForceByNindep;    /* Do not scale the forces by number of independet data points */

    gmx_bool        bGridDensity;    /* Average density on a grid around the envelope */
    int             gridDensityMode; /* Specifies which densit is averaged: 0=solute+hydration layer, 1=solute, 2=hydration layer */

    gmx_bool        bFixSolventDensity;        /* Fixing the solvent density to a given (experimental) value in waxs-solvdens */
    real            givenElecSolventDensity;   /* Given electron density of solvent (= 0 means use xtc) (mdp option waxs-solvdens) */
    real            solventDensRelErr;         /* Relative uncertainty of the solvent density - this can be translated into an
                                                  uncertainty of I(q), which may enter the E[waxs] */
    double          nElecAddedA, nElecAddedB;  /* Average number of added electrons to A and B systems */

    /* If the buffer subtraction was reduced by the volue of the solute, we need: */
    gmx_bool        bCorrectBuffer;            /* Add back oversubtracted buffer     */
    double          soluteVolAv;               /* volume of the solute, used to add back oversubtracted buffer, and to get solute
                                                  contrast wrt. to buffer. */

    /* Stuff for Bayesian methods / Gibbs samping */
    gmx_rng_t       rng;                       /* Random number generator */
    double          epsilon_buff;              /* Uncertainty factor in the buffer density - to be sampled by Gibbs sampling */

    /* Stuff for ensemble refinement */
    int             ewaxs_ensemble_type;   /* WAXS ensemble type (none, bayesian_fixed, bayesian_allopt) */
    int             ensemble_nstates;      /* Number of states (in addition to the simulated state)               */
    double         *ensemble_weights;      /* Weights of states: size real[ensemble_nstates].
                                              Weight of the MD simulation is ensemble_weights[ensemble_nstates-1] */
    double         *ensemble_weights_init; /* Initial weights                                                     */
    double          ensemble_weights_fc;   /* Force constant to restrain weights to initial value                 */
    double          stateWeightTolerance;   /* Tolerance for weights, allowing state weights to be <0 or >1, such that
                                               weights are [-tol, 1+N*tol -tol]. Default is 0.1. */

    /* More WAXS-MD controls */
    gmx_bool        bHaveFittedTraj;
    gmx_bool        bRotFit;
    gmx_bool        bMassWeightedFit;     /* Do a mass-weighted fit */
    int             debugLvl;             /* Control verbosity */

    /* Turn down position restraints during tau and 2tau. This features can be used to restrain the protein
       during the first tau, so the simulation has time to recollect the averages after a restart.
       This is useful because we cannot restart from a checkpoint file at present. */
    gmx_bool       bRemovePosresWithTau;     /* Slowly turn down position restraints between tau and 2*tau */
    double         simtime0;                 /* Simulation time at first WAXS step */
    real           posres_fconst;            /* Force constant for position restraints */

    /* Global lists of solute(protein), solvent, and excluded solvent. (global atom numbers) */
    int            nindA_prot ;
    atom_id       *indA_prot;               /* Protein atoms (mdp group waxs-solute)        */
    int            nindA_sol ;
    atom_id       *indA_sol;                /* All solvent atoms in protein system (mdp group waxs-solvent) */
    int            nindB_sol ;
    atom_id       *indB_sol;                /* All solvent atoms in pure-solvnet box (mdp group waxs-solvent in
                                         pure solvent sytem */

    /* Variables used for fitting the protein into the enelope */
    char          *x_ref_file;          /* Reference coordinates for fitting into the envelope (e.g. envelope-ref.gro) */
    gmx_bool       bHaveRefFile;
    int            nind_RotFit;         /* Number of atoms used for rotational fit, to fit the protein into the envelope */
    atom_id       *ind_RotFit;          /* Atom index used for rotational fit */
    rvec          *x_ref;               /* Only on Master. Used in MD to rotationally fit the protein into the reference frame. */
    rvec           origin;              /* The designated centre to shift all WAXS frames to eliminate fluctuations, origin=(0,0,0). */
    real          *w_fit;               /* Fitting weights (mass-weighted or uniformly-weighted) */


    /* Variables for making the solute whole */
    atom_id       *pbcatoms;           /* WAXS PBC atoms. Analogous to pull pbc atom. At best near the solute's COM */
    int            npbcatom;           /* Edited to include several candidates to faciliate a COG calculation thereof. */
    gmx_bool       bHaveWholeSolute;   /* Solute is already whole (e.g., due to posres simulation). No need to make
                                          select WAXS PBC atoms */
    int            nSoluteMols;        /* # of unconnected molecules in Solute/Protein. If ==1, I can use distances between
                                          numberwise atomic neighbors to make the solute whole. */

    /* List of atoms inside the envelope. Using *global* atom numbers. Updated every WAXS step. */
    int            isizeA;                 /* Nr of atoms in Protein + solvation layer (updated) */
    int           *indexA;                 /* Protein + solvation layer index list                */
    int            indexA_nalloc;          /* memory allocated for indexA and atoms inside Envelope         */
    int            isizeB;                 /* Nr of atoms in excluded solvent              */
    int           *indexB;                 /* Excluded solvent                             */
    int            indexB_nalloc;          /* memory allocated for indexB and redB         */
    rvec          *atomEnvelope_coord_A;   /* Coordinates of atoms inside envelope in solvent system A */
    rvec          *atomEnvelope_coord_B;   /* Coordinates of atoms insdie envelope in solute  system B */


    t_waxs_solvent  waxs_solv;             /* Topology and coordinates for pure-solvent box */

    t_state        *local_state;           /* Keep a pointer to the local state, since we need this for dd_collect_vec()
                                              within do_force */
    rvec           *x;                     /* Coordinates of all atoms, only on Master, size of total nr of atoms.
                                              Used to a) make protein whole, b) shift to center, c) neighbor searching */

    real           vLast;                 /* Last calculated potential energy       (sum over all waxs types)  */
    rvec          *fLast;                 /* Last calcuated forces. Size nindA_prot (sum over all waxs types) */

    /* Stuff for damping the SAXS-derived forces after a conformational transitions, to
       avoid "overshooting". Dez. 2017: It seems that we don't need this any more, we can instead
       avoid overshooting with mdp option waxs-t-target */
    t_waxs_eavrmsd wrmsd;               /* Compute past ensemble position as exponential average */
    gmx_bool       bDampenForces;       /* Uses the RMSD average to dampen the force */
    real           damp_min, damp_max;  /* Range over which to apply the switching function. */


    /* OUTPUT STUFF */
    t_waxs_output wo;         /* WAXS output stuff */

    gmx_bool bRecalcSolventFT_GPU ;   /* Set True if Solvent FT has been recalculated and therefore has to be
                                         pushed on GPU in case of GPU support */


    /* For measuring compute times */
    t_waxsTiming compTime;

} t_waxsrec;
/* This container has all variables that are to be read for MD and g_waxs calculations.
 * i.e. to be self-contained, using no structures specific to MD/g_waxs.
 *
 * All sources must be converted to this shared format to do calculations.
 * this results it being usable by both g_waxs and mdrun.
 *
 * NB: sf-tables and qvecs calculated with this.
 * NB: rearrange_atoms converts g_waxs -> reducedatoms. Need MD-version.
 * */

/* Length (relative to box dimension) by which we shift the pure-solvent frame if the number of
   requested frames (by mdp file) is larger than the number of frames in the pure-solvent xtc
   file. If there are sufficient frames in the pure-solvent xtc, this variable is never used. */
#define WAXS_SHIFT_WATERFRAME_BOXRATIO (20.0/191)


#ifdef __cplusplus
}
#endif

#endif
