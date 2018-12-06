/* -*- mode: c; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; c-file-style: "stroustrup"; -*-
 *
 *
 *                This source code is part of
 *
 *                 G   R   O   M   A   C   S
 *
 *          GROningen MAchine for Chemical Simulations
 *
 *                        VERSION 3.2.0
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
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
 * GROningen Mixture of Alchemy and Childrens' Stories
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/* Do we need some #ifdef statement here (such as GMX_THREAD_SHM_FDECOMP) ?? */
#include <pthread.h>

#include <ctype.h>
#include "sysstuff.h"
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
#include "tpxio.h"
#include "typedefs.h"
#include "statutil.h"
#include "oenv.h"
#include "gmxfio.h"
#include "xvgr.h"
#include "matio.h"
#include "sftypeio.h"
#include "waxsmd.h"

/* This section contains all base structure factor operations. */

/* t_cromer_mann * init_cmsf() */
/* { */
/*     t_cromer_mann * temp; */
/*     snew(temp, 1); */
/*     temp->a[0] = temp->a[1] = temp->a[2] = temp->a[3] = 0.; */
/*     temp->b[0] = temp->b[1] = temp->b[2] = temp->b[3] = 0.; */
/*     temp->c = 0; */
/*     return temp; */
/* } */

void print_cmsf(FILE *out, t_cromer_mann *cm, int nTab)
{
    int i;
    for (i=0; i<nTab; i++)
        fprintf(out,"\t");
    for (i=0; i<4; i++)
        fprintf(out," %8g", cm->a[i]);
    for (i=0; i<4; i++)
        fprintf(out," %8g", cm->b[i]);
    fprintf(out," %8g\n", cm->c);
}

void init_scattering_types( t_scatt_types *t )
{
    t->ncm = t->nnsl = 0;
    t->cm  = NULL;
    t->nsl = NULL;
}

void done_scattering_types( t_scatt_types * t )
{
    sfree(t->cm);
    sfree(t->nsl);
    t->ncm = t->nnsl = 0;
}

/* A shorthand way of setting individual factors */
t_cromer_mann cromer_mann(real a[], real b[], real c)
{
    t_cromer_mann cm;
    int i;

    for (i=0 ; i<4; i++)
    {
        cm.a[i] = a[i];
        cm.b[i] = b[i];
    }
    cm.c = c;
    return cm;
}

static int
is_eq_cmsf ( t_cromer_mann *x, t_cromer_mann *y)
{
    int i;
    const double eps = 1e-5;

    for(i=0; i<4; i++)
    {
        if( fabs(x->a[i]-y->a[i]) > eps || fabs(x->b[i]-y->b[i]) > eps )
        {
            return FALSE;
        }
    }
    if ( fabs(x->c-y->c) > eps )
    {
        return FALSE;
    }

    return TRUE;
}
static
int is_eq_nsl ( t_neutron_sl *x, t_neutron_sl *y)
{
    int i;
    const double eps = 1e-6;

    if (fabs(x->cohb - y->cohb) > eps)
    {
        return FALSE;
    }

    return TRUE;
}


/* Look for the Cromer-Mann types cm_new in the list scattTypes */
int search_scattTypes_by_cm( t_scatt_types *scattTypes, t_cromer_mann * cm_new)
{
    int i;

    if (cm_new == NULL)
    {
        gmx_fatal(FARGS,"Trying to search structure-factor table using a null-CMSF!\n");
    }

    for(i = 0; i < scattTypes->ncm; i++)
    {
        if (is_eq_cmsf( &scattTypes->cm[i], cm_new))
            return i;
    }
    return NOTSET;
}

/* Look for the NSL types nsl_new in the list scattTypes */
int search_scattTypes_by_nsl( t_scatt_types *scattTypes, t_neutron_sl *nsl_new)
{
    int i;

    if (nsl_new == NULL)
    {
        gmx_fatal(FARGS,"Trying to search structure-factor table, but nsl_new = NULL!\n");
    }

    for(i = 0; i < scattTypes->nnsl; i++)
    {
        if (is_eq_nsl( &scattTypes->nsl[i], nsl_new))
        {
            return i;
        }
    }
    return NOTSET;
}


int add_scatteringType( t_scatt_types *scattTypes, t_cromer_mann *cm, t_neutron_sl *nsl)
{
    int icm, insl;


    if (cm && nsl)
    {
        gmx_fatal(FARGS, "Can only add a Cromer-Mann or a NSL in add_scatteringType\n");
    }

    if (cm)
    {
        icm  = scattTypes->ncm;
        scattTypes->ncm++;
        srenew(scattTypes->cm, scattTypes->ncm);

        scattTypes->cm[icm] = *cm;
        return icm;
    }
    else if (nsl)
    {
        insl = scattTypes->nnsl;
        scattTypes->nnsl++;
        srenew(scattTypes->nsl, scattTypes->nnsl);

        scattTypes->nsl[insl] = *nsl;
        return insl;
    }

    return -1;
}



t_spherical_map * init_spherical_map( void )
{
    t_spherical_map * qvecs;
    snew(qvecs, 1);

    qvecs->nabs = 0;
    qvecs->abs  = NULL;
    qvecs->ref  = NULL;
    snew(qvecs->ind, 1);
    qvecs->ind[0] = 0;
    qvecs->n      = 0;
    qvecs->q      = NULL;

    return qvecs;
}

void done_spherical_map( t_spherical_map *qvecs )
{
    sfree(qvecs->abs);
    sfree(qvecs->ref);
    sfree(qvecs->ind);
    sfree(qvecs->q);

    sfree(qvecs);
    qvecs = NULL;
}


/* t_waxs_GPU *init_GPU_data(void) */
/* { */
/*     t_waxs_GPU  *gpu_data; */
/*     snew(gpu_data, 1); */
/* //    gpu_data->xlinear_realloc       = 0; */
/* //    gpu_data-> x_linearized        = NULL; */
/*     gpu_data->iTable_GPU           = NULL; */
/*     gpu_data->aff_table_linearized = NULL; */
/*     gpu_data->q_linearized         = NULL; */
/*     gpu_data->useGPU               = FALSE; */

/*     return gpu_data; */
/* } */

t_waxsrecType init_t_waxsrecType()
{
    t_waxsrecType t;

    t.type           = -1;
    t.fc             = 0;
    t.nq             = 0;
    t.minq = t.maxq  = 0;
    t.nShannon       = 0;
    t.qvecs          = NULL;
    t.Ipuresolv      = NULL;
    t.Iexp           = NULL;
    t.Iexp_sigma     = NULL;
    t.f_ml = t.c_ml  = 0;
    t.targetI0       = 0;
    t.ensemble_I     = NULL;
    t.ensemble_Ivar  = NULL;
    t.ensembleSum_I  = NULL;
    t.ensembleSum_Ivar = NULL;
    t.wd             = NULL;
    strcpy(t.saxssansStr, "");

    t.atomEnvelope_scatType_A = NULL ;
    t.atomEnvelope_scatType_B = NULL ;

    t.naff_table = 0;
    t.aff_table  = NULL;

    t.vLast      = 0;
    t.fLast      = NULL;

    t.nnsl_table  = 0;
    t.nsl_table   = NULL;
    t.deuter_conc = -1;

    t.envelope            = NULL;
    t.givenSolventDensity = 0;
    t.soluteSumOfScattLengths = 0;
    t.soluteContrast = 0;
    t.contrastFactor = 0;

    t.n2HAv_A  = 0;
    t.n1HAv_A  = 0;
    t.nHydAv_A = 0;
    t.n2HAv_B  = 0;
    t.n1HAv_B  = 0;
    t.nHydAv_B = 0;

    return t;
}


/* waxs records. */
t_waxsrec * init_t_waxsrec(void )
{
    t_waxsrec * t;
    snew(t, 1);

    t->nTypes      = 0;
    t->cmtypeList  = NULL;
    t->nsltypeList = NULL;

    t->x           = NULL;
    t->x_ref       = NULL;

    /* Make origin 0,0,0 - this is where the protein is eventually shifted
       to fit into the envelope */
    clear_rvec(t->origin);
    t->w_fit       = NULL;

    t->nstlog      = 0;
    t->xray_energy = 0;
    t->wt          = NULL;
    t->tau         = 0;
    t->tausteps    = 0;
    t->nstcalc     = 0;
    t->potentialType = 0;
    t->weightsType   = 0;
    t->ewaxs_ensemble_type = ewaxsEnsembleNone;
    t->bBayesianSolvDensUncert = FALSE;
    t->J           = 0;
    t->Jmin        = GMX_WAXS_JMIN;
    t->Jalpha      = GMX_WAXS_J_ALPHA;
    t->scale       = 0.;
    t->stepCalcNindep = 0;
    t->bHaveNindep = FALSE;
    t->debugLvl    = 1;
    t->simtime0    = 0;

    t->ewaxs_Iexp_fit      = ewaxsIexpFit_NO;
    t->nSolvWarn           = 0;
    t->nWarnCloseBox       = 0;
    t->bScaleI0            = FALSE;
    t->soluteVolAv         = 0;
    t->bCorrectBuffer      = FALSE;
    t->bFixSolventDensity  = FALSE;
    t->givenElecSolventDensity = 0.;
    t->solventDensRelErr   = 0.;
    t->solElecDensAv       = 0;
    t->nElectrons          = NULL;
    t->nElecAddedA         = t->nElecAddedB = 0.;
    t->nElecTotA           = 0.;
    t->nElecProtA          = 0.;
    t->bMassWeightedFit    = TRUE;
    t->bHaveFittedTraj     = FALSE;
    t->stateWeightTolerance= GMX_WAXS_WEIGHT_TOLERANCE_DEFAULT;
    t->bGridDensity        = FALSE;
    t->gridDensityMode     = 0;
    t->bRotFit             = FALSE;
    t->bRemovePosresWithTau= TRUE;
    t->posres_fconst       = 500;
    t->bCalcPot            = FALSE;
    t->bCalcForces         = FALSE;
    t->bDampenForces       = FALSE;
    t->damp_min            = 2.5;
    t->damp_max            = 3.5;
    t->backboneDeuterProb   = BACKBONE_DEUTERATED_PROB_DEFAULT;
    t->bStochasticDeuteration = FALSE;
    t->bDoNotCorrectForceByContrast = FALSE;
    t->bDoNotCorrectForceByNindep   = FALSE;

    t->epsilon_buff         = 1;
    t->rng                  = NULL;

    t->waxsStep            = 0;
    t->calcWAXS_begin      = -1;
    t->calcWAXS_end        = -1;
    t->nAtomsExwaterAver = t->nAtomsLayerAver = 0;
    t->nElecAvA = t->nElecAv2A = t->nElecAv4A = 0.;
    t->nElecAvB = t->nElecAv2B = t->nElecAv4B = 0.;
    t->solElecDensAv = t->solElecDensAv2 = t->solElecDensAv4 = 0.;
    t->RgAv = 0;

    t->isizeA           = t->isizeB = 0;
    t->npbcatom         = 0;
    t->pbcatoms         = NULL;
    t->bHaveWholeSolute = FALSE;

    t->atomEnvelope_coord_A = t->atomEnvelope_coord_B = NULL ;

    t->indexA = t->indexB = NULL;
    t->indexA_nalloc = t->indexB_nalloc = 0;

    t->ensemble_nstates      = 0;
    t->ensemble_weights      = NULL;
    t->ensemble_weights_init = NULL;
    t->ensemble_weights_fc   = 0;

    t->indA_prot   = NULL;
    t->indA_sol    = NULL;
    t->indB_sol    = NULL;
    t->nindA_prot  = 0;
    t->nindA_sol   = 0;
    t->nindB_sol   = 0;
    t->nSoluteMols = 0;

    t->ind_RotFit  = NULL;
    t->nind_RotFit = 0;

    t->vLast      = 0;
    t->fLast      = NULL;

    t->bVacuum = FALSE;

    t->waxs_solv   = NULL;
    t->local_state = NULL;
    t->wo          = NULL;
    t->wrmsd       = NULL;
    t->compTime    = NULL;

    t->bUseGPU     = FALSE;

    return t;
}

void done_t_waxsrec( t_waxsrec * t )
{
    /* This function is totally incomplete, but we don't really need it anyway */

    if (t->x) sfree(t->x);
    if (t->x_ref) sfree(t->x_ref);
    if (t->w_fit) sfree(t->w_fit);
    // if ( t->wt ) done_sim_sf_types( t->wt );
    // if ( t->qvecs ) done_spherical_map(t->qvecs );
    if ( t->pbcatoms ) sfree( t->pbcatoms );
    // if ( t->sf_table ) done_sf_table( t->sf_table, t->ntypes );

    //if( t->Icalc ) sfree( t->Icalc ) ;
    //if( t->Icalc_sigma ) sfree(t->Icalc_sigma );
    // if( t->Iexp ) sfree( t->Iexp);

    // if( t->Iexp_sigma) sfree(t->Iexp_sigma) ;

    /* if( t->protein_index) sfree(t->protein_index) ; */

    sfree(t->indexA);
    if (t->indexB)     sfree(t->indexB);
    if (t->indA_prot)  sfree(t->indA_prot);
    if (t->indA_sol)   sfree(t->indA_sol);
    if (t->indB_sol)   sfree(t->indB_sol);
    if (t->ind_RotFit) sfree(t->ind_RotFit);

    sfree(t);
    t = NULL;
}

int get_actual_ngroups( gmx_groups_t *groups, int egc )
{
    int n;

    n=groups->grps[egc].nr ;
    if (n == 1)
    {
        /* Can be either all or nothing. */
        if ( strncmp("rest",*(groups->grpname[groups->grps[egc].nm_ind[0]]),4) != 0 )
        {
            /* Found real group. */
            n++;
        } else {
            /* Is FAKE. */
        }
    }
    if ( n <= 0 )
        gmx_fatal(FARGS,"Death Horror in get_actual_n_groups (sftypeio.c) - ngrps < 0 !\n");

    return n-1 ; /* Remove fake group. */
}
