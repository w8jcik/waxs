/*
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
 * Gromacs Runs On Most of All Computer Systems
 */

/* Authors' Notes:
 *  We would like this file to contain the main I/O and basic utilities of
 *  the types defined in waxsrec.h. One would ideally include just this file to gain
 *  the basic reading, writing, searching, and editing of the types.
 *
 *  Whereas, sfactors.c contains more complex routines more directly related to
 *  scattering factor calculations.
 *
 *  This makes it easier to fork general mathematical constructs in the future:
 *  - t_spherical_maps
 *  - vector_maps
 *  - t_reduced_atoms
 * */

#ifndef _sftypeio_h
#define _sftypeio_h

#include <stdio.h>
#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/waxs/waxstop.h"
#include "gromacs/waxs/waxsrec.h"
#include "gromacs/topology/topology.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int size;
    int *index;
} int_map ;

// t_cromer_mann * init_cmsf(void);
// t_nuclear_sf * init_nusf(void);

void init_scattering_types( t_scatt_types * t );
void done_scattering_types( t_scatt_types * t );

// t_gpp_sf_types * init_gpp_sf_types( void );
// void done_gpp_sf_types( t_gpp_sf_types * t );
/* Creation-destruction operators. */

// void set_cmsf ( t_cromer_mann *cm, real *a, real *b, real *c);

void print_cmsf(FILE *out, t_cromer_mann *cm, int nTab);

// void incr_gpp_sf_mem( t_gpp_sf_types *gsf ); /* Handles the memory renewal after init. */

//int set_gpp_sf_type( t_gpp_sf_types *sflist, int index, char * name, int p, int n, real mass, real charge,
//           int ftype, t_cromer_mann * cmsf, t_nuclear_sf * nusf, gmx_bool bUse );
/* Returns the new index as a short cut, NOTSET on out-of-bounds error.*/

// int search_gppsft_byname( t_gpp_sf_types *sflist, char * qname, int ftype ); /* EXACT search. */
// int search_gpp_cromermann_byname( t_gpp_sf_types *sflist, char * qname, int ftype );

// int search_gppsft_byletter( t_gpp_sf_types *sflist, const char * qname, const int ftype ); /* Search by 1st and second letters. */
// int search_gppsft_bymass( t_gpp_sf_types *sflist, real qmass, const real rel_mass_diff, int ftype );
int search_scattTypes_by_cm( t_scatt_types *scattTypes, t_cromer_mann * cm_new);
int search_scattTypes_by_nsl( t_scatt_types *scattTypes, t_neutron_sl *nsl_new);
/* Returns the index of the existing table if found, and NOTSET if not found. */

//t_cromer_mann get_cromermanntype_by_atype(t_gpp_sf_types *sflist,  gpp_atomtype_t atype );
// int search_gppsft_byatomprop( t_gpp_sf_types *sflist, char * qname, real qmass, real mdiff, real qq, real qdiff );

// void add_gpp_cromermann_type(t_gpp_sf_types *gsf, t_cromer_mann cm, char* attype);
/* Adds this type to the type-list. Int returns error on non-zero. */

int add_scatteringType( t_scatt_types* sim_sf, t_cromer_mann *cmsf, t_neutron_sl *nusf);
/* Generic adder for sorted-unque list for simulations. */

// void clone_gpp2sim_sftype( t_scatt_types *sim_sf, t_gpp_sf_types *gpp_sf );
/* Clones the relevant data from a prepared grompp list to the sim list.*/

// int clone_sfindex2state( t_state *state, int_map *src );

// int add_gsftypes_from_file( t_gpp_sf_types *gsftypes, char *cmsf_fn, int ftype );
/* Reads all lines from file and adds a gsftype for each valid type found. No init. */
/* We expect to define handing of different types in this function. */

void done_sf_table( real **t, int ntypes );

t_spherical_map * init_spherical_map( void );

void done_spherical_map( t_spherical_map * qvecs);

// void add_qabs_to_spheremap( t_spherical_map * qvecs, real qabs, int iref, int nJ );

// int populate_spheremap_with_rvecs( t_spherical_map *qvecs, int index, rvec *x_in, int nx_in);
/* rvec is copied from x_in[0]~x_in[J-1] to qvecs[index]~ */

t_waxsrec * init_t_waxsrec();

void done_t_waxsrec( t_waxsrec *t );

t_waxsrecType init_t_waxsrecType();
    
int get_actual_ngroups( gmx_groups_t *groups, int egc );
/* Because the current group numbering system is ambiguous. */

#ifdef __cplusplus
}
#endif

#endif    /* _gpp_atomtype_h */








