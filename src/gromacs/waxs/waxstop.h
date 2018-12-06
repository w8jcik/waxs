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

#ifndef _waxstop_h
#define _waxstop_h

#include "gromacs/math/gmxcomplex.h"
// #include "topology.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/utility/real.h"

// enum { SF_FUNCTYPE_NULL, SF_FUNCTYPE_CROMER_MANN, SF_FUNCTYPE_NUCLEAR };

/* #define SF_TYPE_CM    1 */
/* #define SF_TYPE_NU    1<<1 */
/* #define SF_TYPE_CMSOL 1<<2 */
/* #define SF_TYPE_NUSOL 1<<3 */


#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    real a[4];
    real b[4];
    real c;
} t_cromer_mann;

/* Neutron scattering length (could later add incoh. b, ) */
typedef struct {
    real cohb;
} t_neutron_sl;

/* typedef union { */
/*     struct { real a[4]; real b[4]; real c; } cm; */
/*     struct { real nsl; } nsl; */
/* } t_scattering_params; */

/* typedef struct { */
/*     int ncmtypes;                /\* Nr of Cromer-Mann scattering types *\/ */
/*     char **cmname;               /\* Atom type names *\/ */
/*     t_cromer_mann *cmtypes;      /\* The cromer-mann types from [ cromer_mann_types ] *\/ */

/*     int ncmunique;               /\* Nr of unique CM types, to be added to tpr *\/ */
/*     t_cromer_mann *cmunique;     /\* The unique Cromer-Mann *\/ */

/*     int nnsltypes;                /\* Nr of NSL types from [ nuclear_types ] *\/ */
/*     char **nslname;               /\* Atom type names *\/ */
/*     t_neutron_sl *nsltypes;       /\* The NSL types   *\/ */

/*     int nnslunique;               /\* Nr of unique NSL types, to be added to tpr *\/ */
/*     t_neutron_sl *nslunique;      /\* The unique NSL *\/ */

/*     int ntot; */
/*     char ***name; /\* Type identifier for searching*\/ */
/*     int *p;       /\* Proton number *\/ */
/*     int *n;       /\* Neutron number *\/ */
/*     real *mass;   /\* Mass *\/ */
/*     real *charge; /\* Charge *\/ */
/*     int *ftype;   /\* Whether the struct contains CMSFd, NSFs, or both. *\/ */
/*     t_cromer_mann *cm; */
/*     t_neutron_sl  *nu; */

/*     gmx_bool * bInUse ; */
/* } t_gpp_sf_types; */
/* The type used in grompp, to accurately determine what data to put into the next one. */

typedef struct {
    int ncm;             /* Nr of Cromer-Mann paramers */
    int nnsl;            /* Nr of NSLs                 */

    t_cromer_mann *cm;   /* Cromer-Mann parameters     */
    t_neutron_sl *nsl;   /* Neutron scattering lengths */
} t_scatt_types;
/* The data type written into the tpr and used in simulation. Minimised. */

#ifdef __cplusplus
}
#endif

#endif
