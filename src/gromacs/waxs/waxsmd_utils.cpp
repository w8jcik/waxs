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

//#include "sysstuff.h"
//#include "typedefs.h"
#include <cstring>
#include "gromacs/math/invertmatrix.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/futil.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/groio.h"
#include "gromacs/waxs/waxsmd_utils.h"
#include "gromacs/waxs/sftypeio.h"
#include "gromacs/math/do_fit.h"
#include "gromacs/waxs/gmx_miniball.h"
#include "gromacs/waxs/gmx_envelope.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/statistics/statistics.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/mdtypes/md_enums.h" //md_enums.h
#include "gromacs/gmxpreprocess/notset.h"

/*
 *  This source file was written by Jochen Hub.
 */

//  #define waxs_debug(x) _waxs_debug(x, __FILE__, __LINE__)
#define waxs_debug(x)

double CMSF_q(t_cromer_mann cmsf, real q)
{
    real f, k2;
    int i;

    f = cmsf.c;
    /* Note: Cromer-Mann parameters use inverse Angstroem, but our q is in inv. nm.
       Also note: q/4pi = sin(theta)/lambda
    */
    k2  = gmx::square(q/(4*M_PI*10));
    for (i = 0; (i < 4); i++)
    {
        f += cmsf.a[i] * exp (-cmsf.b[i] * k2);
    }

    return f;
}


/* Non-weighted center of mass */
static void
get_cog(rvec x[], atom_id *index, int n, rvec cent)
{
    int i;
    rvec temp = {0. ,0. ,0.};
    for (i=0; i<n; i++)
    {
        rvec_inc(temp, x[index[i]]);
    }
    svmul(1.0/n, temp, cent);
}

/* Get the atom with the largest distance from a, store index in imax, and return this
   largest distance */
static real
largest_dist_atom(rvec x[], atom_id *index, int n, rvec a, int *imax)
{
    int i;
    real d2, d2max = 0;
    rvec v;

    *imax = 0;
    for (i = 0; i < n; i++)
    {
        rvec_sub(x[index[i]], a, v);
        d2 = norm2(v);
        if (d2 > d2max)
        {
            d2max = d2;
            *imax  = i;
        }
    }

    return sqrt(d2max);
}

/* Draw a sphere that includes the sphere around cent with radius R as well as point x.
   Upate cent and R. */
static void
update_bounding_sphere(rvec cent, real *R, rvec x)
{
    rvec v, vinv, tmp;
    real l;

    rvec_sub(cent, x, v);
    l = sqrt(norm2(v));
    svmul(1./l, v, v);

    /* update R and center */
    *R = (l + *R)/2;
    svmul(*R, v, v);
    rvec_add(x, v, cent);
}


void
get_bounding_sphere_COM(rvec x[], atom_id *index, int n, rvec cent, real *R, gmx_bool bVerbose)
{
    rvec xcom, v1;
    int imax1 = 0;

    if (n == 0)
    {
        gmx_fatal(FARGS, "Error while getting bounding sphere: zero atoms in index\n");
    }
    if (n == 1)
    {
        copy_rvec(x[index[0]], cent);
        *R = 0;
        return;
    }

    clear_rvec(xcom);
    /* Need COM of protein for both versions */
    get_cog(x, index, n, xcom);

    /* Get largest distance from COM */
    largest_dist_atom(x, index, n, xcom, &imax1);
    rvec_sub(x[index[imax1]], xcom, v1);
    *R = sqrt(norm2(v1));
    copy_rvec(xcom, cent);
    if (bVerbose)
    {
        fprintf(stderr,"COM-method: %g from [%g, %g, %g]\n", *R, xcom[0], xcom[1], xcom[2]);
    }

}

/* Return an estimate for the center and the radius of the bounding sphere
   around atoms x[]. Is a simplified version of Ritter's algorithm. */
void
get_bounding_sphere_Ritter_COM(rvec x[], atom_id *index, int n, rvec cent, real *R, gmx_bool bVerbose)
{
    int imax1 = 0, imax2 = 0, nit = 0;
    rvec midpoint, v1, v2, xcom, centRitter;
    real d, Rcom, Rritter;
    const real eps = 1e-6;
    static gmx_bool bDoRitter = TRUE, bDoCOM = TRUE, bFirst = TRUE;
    static int ritterAtom1 = -1, ritterAtom2 = -1;

    if (n == 0)
    {
        gmx_fatal(FARGS, "Error while getting bounding sphere: zero atoms in index\n");
    }
    if (n == 1)
    {
        copy_rvec(x[index[0]], cent);
        *R = 0;
        return;
    }


    /* On the first call of this function, we get the bounding sphere around the COM AND
       from Ritter's algorithm. Then we take the smaller sphere, and stick to this method
       for the rest of the simulation. */
    Rcom = Rritter = 1e20;
    clear_rvec(xcom);
    clear_rvec(centRitter);
    /* Need COM of protein for both versions */
    get_cog(x, index, n, xcom);

    if (bDoCOM)
    {
        /* Get largest distance from COM */
        largest_dist_atom(x, index, n, xcom, &imax1);
        rvec_sub(x[index[imax1]], xcom, v1);
        Rcom = sqrt(norm2(v1));
        if (bVerbose)
        {
            fprintf(stderr,"COM-method: %g from [%g, %g, %g]\n", Rcom, xcom[0], xcom[1], xcom[2]);
        }
    }

    if (bDoRitter)
    {
        /* Ritter: */
        if (ritterAtom1 == -1)
        {
            /* When this function is called the first time, we choose the two
               initial atoms for Ritter's algorithm. In the next calls, we start
               with the same two atoms. We hope that this will make the Ritter
               bounding sphere more stable */

            /* point X with largest distance from COM */
            largest_dist_atom(x, index, n, xcom, &imax1);
            /* point Y with largest distance from X */
            largest_dist_atom(x, index, n, x[index[imax1]], &imax2);
            ritterAtom1 = index[imax1];
            ritterAtom2 = index[imax2];
        }

        /* sphere around center of X-Y line */
        rvec_add(x[ritterAtom1], x[ritterAtom2], v1);
        svmul(0.5, v1, centRitter);
        rvec_sub(x[ritterAtom1], x[ritterAtom2], v1);
        Rritter = sqrt(norm2(v1)) / 2;

        /* Ritter's algorithm would here do a loop over all atoms, and expand the sphere each time the atom
           is outside. Here, we simply pick the atom with the largest distance and expand the sphere accordingly */
        while ( (d = largest_dist_atom(x, index, n, centRitter, &imax1)) > (Rritter + eps))
        {
            /* printf("Update bounding sphere imax %d, d %g (R = %g)\n", imax, d, *R); */
            update_bounding_sphere(centRitter, &Rritter, x[index[imax1]]);
            if (nit++ > 10000)
            {
                gmx_fatal(FARGS, "Error while getting bounding sphere: No convergence after %d iterations\n", nit);
            }
        }
        if (bVerbose)
        {
            fprintf(stderr,"Ritter-sph: %g from [%g, %g, %g]\n", Rritter, centRitter[0], centRitter[1], centRitter[2]);
        }
    }

    if (Rritter < Rcom)
    {
        *R = Rritter;
        copy_rvec(centRitter, cent);
        bDoCOM = FALSE;
    }
    else
    {
        *R = Rcom;
        copy_rvec(xcom, cent);
        bDoRitter = FALSE;
    }

    if (bFirst)
    {
        // printf("\tContructed bounding sphere based on %s.\n", bDoRitter ? "Ritter's algorithm" : "center-of-mass");
        printf("\tR(Ritter)         = %g\n\tR(center-of-mass) = %g\n", Rritter, Rcom);
        bFirst = FALSE;
    }

    if (bVerbose && bFirst)
    {
        FILE *fp = fopen("bounding_sphere.py", "w");
        fprintf(fp, "from pymol.cgo import *\nfrom pymol import cmd\n\n");
        fprintf(fp, "obj = [\n\nALPHA,  0.4,\n");
        fprintf(fp, "COLOR, %g, %g, %g,\n\n", 0.1, 0.8, 0.4);
        /* Draw bounding sphere around origin, since the protein will be shifted there */
        fprintf(fp, "SPHERE, %g,%g,%g, %g,\n", 10*cent[0], 10*cent[1], 10*cent[2], 10 * (*R));
        fprintf(fp, "]\n\ncmd.load_cgo(obj, 'boundsphere')\n\n");
        fclose(fp);
        printf("\nWrote bounding sphere to pymol CGO file %s\n\n", "bounding_sphere.py");
    }
}

void
get_bounding_sphere(rvec x[], atom_id *index, int n, rvec cent, real *R, gmx_bool bVerbose)
{
    /* NT is the floating point variable in gmx_miniball.c. It may be either real or double */
    NT *centNT, **xsoluteNT, relerr, subopt, *xNT = NULL;
    rvec xsolute, centtmp;
    int i, d;
    gmx_miniball_t mb;
    static gmx_bool bFirst = TRUE;
    double miniball_time;
    real Rrittercom;
    FILE *fp;

    if (n == 0)
    {
        gmx_fatal(FARGS, "Error while getting bounding sphere: zero atoms in index\n");
    }
    if (n == 1)
    {
        copy_rvec(x[index[0]], cent);
        *R = 0;
        return;
    }

    if (sizeof(NT) != sizeof(real))
    {
        snew(xNT, DIM*n);
        for (i=0; i<n; i++)
        {
            for (d = 0; d < DIM; d++)
            {
                /* Copy real array x into linear NT array xNT */
                xNT[DIM*i+d] = x[index[i]][d];
            }
        }
        /* gmx_fatal(FARGS, "You are trying to run the Miniball code in a different precision than Gromacs\n" */
        /*           "This is possible, but then you need to copy the solute atom coords to an array of the" */
        /*           "NT precision\n"); */
    }

    /* Need a linear NT* array with pointers to the solute atoms */
    snew(xsoluteNT, n);
    for (i=0; i<n; i++)
    {
        if (xNT == NULL)
        {
            /* make pointers on the x array - used if NT is type real - do explicit cast to avoid warning if
               NT is double and x is real (so if this line is not used anyway) */
            xsoluteNT[i] = (NT*) x[index[i]];
        }
        else
        {
            /* make pointers on the xNT array - used if NT is not of type real real */
            xsoluteNT[i] = xNT + DIM*i;
        }
    }

    mb = gmx_miniball_init(DIM, xsoluteNT, n);

    if (!gmx_miniball_is_valid(mb, -1))
    {
        relerr = gmx_miniball_relative_error (mb, &subopt);

        if (relerr < 1e-3 && subopt < 0.1)
        {
            fprintf(stderr, "\nWARNING, Miniball did not converged to the default of 10 x machine precision\n"
                    "\tbut to %f. This is not a problem (but worth mentioning).\n\n", relerr);
        }
        else if (relerr < 1e-3)
        {
            fprintf(stderr, "\nWARNING, Miniball generated a slighly sub-optimal bounding sphere (%f instead of 0).\n"
                    "\tThis is not a problem (but worth mentioning).\n\n", subopt);
        }
        else
        {
            fflush(stdout);
            fprintf(stderr, "\n\nThe generated Miniball is invalid.\n");
            fp = fopen("miniball_coords_error.dat", "w");
            for (i=0; i<n; i++)
            {
                fprintf(fp, "%.15f %.15f %.15f\n", xsoluteNT[i][XX], xsoluteNT[i][YY], xsoluteNT[i][ZZ]);
            }
            fclose(fp);
            fprintf(stderr, "Dumped %d coordinates that made an invalid Miniball to miniball_coords_error.dat\n", n);
            fprintf(stderr, "Relative error = %g, suboptimality = %g\n", relerr, subopt);

            gmx_fatal(FARGS, "Generating the bounding sphere with Miniball failed.\n");
        }
    }

    miniball_time = gmx_miniball_get_time(mb);

    centNT = gmx_miniball_center(mb);
    for (d=0; d<DIM; d++)
    {
        cent[d] = centNT[d];
    }
    *R = sqrt(gmx_miniball_squared_radius(mb));

    /* printf("Center = %g %g %g\n", */
    /*            gmx_miniball_center(mb)[XX], gmx_miniball_center(mb)[YY], gmx_miniball_center(mb)[ZZ]); */
    /* printf("Radius^2 = %g\n", gmx_miniball_squared_radius(mb));                                              */
    /* printf("Radius   = %g\n", sqrt(gmx_miniball_squared_radius(mb))); */

    /* printf("Number of support points = %d\n", gmx_miniball_nr_support_points(mb)); */
    /* printf("Support point indices = "); */
    /* Sit it; */
    /* int diff; */
    /* for (it = gmx_miniball_support_points_begin(mb); it!=gmx_miniball_support_points_end(mb);  it++) */
    /* { */
    /*         diff = *it - xsolute; */
    /*         printf("%d  ", diff); */
    /* } */
    /* printf("\n"); */

    /* NT subopt; */
    /* printf("Relative Error = %g\n", gmx_miniball_relative_error (mb, &subopt)); */
    /* printf("Suboptimiality = %g\n", subopt); */
    /* printf("Is valid = %d\n", gmx_miniball_is_valid(mb)); */

    /*for (i=0; i<n; i++)
      {
      sfree(xsolute[i]);
      }*/

    sfree(xsoluteNT);
    if (xNT)
    {
        sfree(xNT);
    }
    gmx_miniball_destroy(&mb);

    if (bFirst)
    {
        printf("Contructed bounding sphere using Miniball.\n");
        printf("\tR(Miniball) = %g\n\tcenter = %g %g %g\n", *R, cent[XX], cent[YY], cent[ZZ]);
        /* Do for comparison also Ritter / COM */
        get_bounding_sphere_Ritter_COM(x, index, n, centtmp, &Rrittercom, FALSE);
        printf("For comparison: Ritter/COM gives R = %g, which is %.2f %% larger than Miniball)\n",
               Rrittercom, (Rrittercom-*R)/(*R)*100);

        bFirst = FALSE;
        printf("Construction of bounding sphere took %g sec.\n", miniball_time);
    }

    if (bVerbose || bFirst)
    {
        FILE *fp = fopen("bounding_sphere.py", "w");
        fprintf(fp, "from pymol.cgo import *\nfrom pymol import cmd\n\n");
        fprintf(fp, "obj = [\n\nALPHA,  0.4,\n");
        fprintf(fp, "COLOR, %g, %g, %g,\n\n", 0.1, 0.8, 0.4);
        /* Draw bounding sphere around origin, since the protein will be shifted there */
        fprintf(fp, "SPHERE, %g,%g,%g, %g,\n", 10*cent[0], 10*cent[1], 10*cent[2], 10 * (*R));
        fprintf(fp, "]\n\ncmd.load_cgo(obj, 'boundsphere')\n\n");
        fclose(fp);
        printf("\nWrote bounding sphere to pymol CGO file %s\n\n", "bounding_sphere.py");
    }
}

#define WAXS_WARN_BOX_DIST 0.5
void
check_prot_box_distance(rvec x[], atom_id *index, int isize, matrix box,
                        t_waxsrec *wr, gmx_bool bPrintWarn, real *mindistReturn)
{
    int      i, d, imin = -1, imax = -1;
    rvec    *ptr_x, boxcenter, min = {1e20, 1e20, 1e20}, max = {-1e20, -1e20, -1e20}, rad;
    real     max2 = -1, rad2, mindist = 1e20, thisdist;
    gmx_bool bWarn = FALSE, bTric;

    bTric = TRICLINIC(box);

    calc_box_center(ecenterTRIC, box, boxcenter);

    for (i = 0; i < isize; i++)
    {
        ptr_x = &(x[index[i]]);

        /* Take note of the maximum radius and dimensions. */
        if (!bTric)
        {
            /* Rectangular box */
            for (d = 0; d < DIM; d++)
            {
                if ((*ptr_x)[d] < min[d])
                {
                    min[d] = (*ptr_x)[d];
                    imin = i;
                }
                if ((*ptr_x)[d] > max[d])
                {
                    max[d] = (*ptr_x)[d];
                    imax = i;
                }

            }
        }
        else
        {
            rvec_sub(*ptr_x, boxcenter, rad);
            rad2 = norm2(rad);
            if (rad2 > max2)
            {
                max2 = rad2;
                imax = i;
            }
        }
    }

    /* Pick minimum distance to box from all atoms */
    if (!bTric)
    {
        for (d = 0; d < DIM; d++)
        {
            thisdist = (box[d][d] - (max[d]-min[d]))/2;
            if (thisdist < mindist)
            {
                mindist = thisdist;
            }
        }
    }
    else
    {
        /* GROMACS convention stores x as (d,0,0), the maximum dimension of rhombic dodecahedron and trunc. octahedron cells. */
        mindist = box[0][0]/2 - sqrt(max2);
    }

    /* Now check if the protein is still not whole... */
    if (!bTric)
    {
        for (d = 0; d < DIM; d++)
        {
            if (mindist < WAXS_WARN_BOX_DIST)
            {
                if (bPrintWarn)
                {
                    fprintf(stderr, "\n\n** WARNING **\n\n"
                            "Solute spans between %g and %g in %c direction (box length = %g).\n"
                            "Seems like the solute is either not whole, or very big.\n\n"
                            " ** Hint: maybe you have not chosen a good waxs-pbc atom ? **\n",
                            min[d], max[d], 'X'+d, box[d][d]);
                }
                bWarn = TRUE;
            }
        }
    }
    else
    {
        /* GROMACS convention stores x as (d,0,0), the maximum dimension of rhombic dodecahedron and trunc. octahedron cells. */
        if (mindist < WAXS_WARN_BOX_DIST)
        {
            if (bPrintWarn)
            {
                fprintf(stderr, "\n\n** WARNING **\n\n"
                        "Solute is reaching dangerously close the edge of the compact PBC-box! (max. distance from "
                        "box center: %g, box radius = %g).\n"
                        "\tbox center   at %8g %8g %8g\n"
                        "\tatom %4d    at %8g %8g %8g\n"
                        "\tSeems like the solute is either not whole, or too big.\n\n",
                        sqrt(max2), box[0][0]/2.0,
                        boxcenter[0], boxcenter[1], boxcenter[2],
                        index[imax], x[index[imax]][0], x[index[imax]][1], x[index[imax]][2]);
            }
            bWarn = TRUE;
        }
    }
    if (bWarn && wr)
    {
        wr->nWarnCloseBox++;
    }
    if (mindistReturn)
    {
        *mindistReturn = mindist;
    }
}

/* Check if all atoms are inside the comapact unit cell */
static gmx_bool
assert_all_atoms_inside_compact_box(rvec x[], atom_id *index, int isize, matrix box, t_pbc *pbc)
{
    int i, d;
    rvec boxcenter, dxpbc, dxdirect;
    real small = box[XX][XX]/10;

    calc_box_center(ecenterTRIC, box, boxcenter);

    for (i = 0; i < isize; i++)
    {
        /* Use that shortest distance (retured by pbc_dx()) from the box center is (by definition) inside the
           compact box */
        pbc_dx(pbc, x[index[i]], boxcenter, dxpbc);
        rvec_sub(   x[index[i]], boxcenter, dxdirect);
        for (d = 0; d < DIM; d++)
        {
            if (fabs(dxpbc[d] - dxdirect[d]) > small)
            {
                printf("\n\nAtom %d (solute atom nr %d) is outside the compact box (x/y/z = %g %g %g) after shifting the bounding sphere to the box center.\n"
                       "This means that your solute is not whole.\n\n"
                       "Probably you (a) need to choose a better waxs-pbc atom, or (b) atoms of your waxs-solute group"
                       "are freely diffusing around (such as ions)\n", index[i], i, x[index[i]][XX], x[index[i]][YY], x[index[i]][ZZ]);
                return FALSE;
            }
        }
    }
    return TRUE;
}

/* mv center of geometry to origin */
void
mv_cog_to_rvec(int natoms, rvec x[], int *index, int isize, rvec target, rvec cog)
{
    int i;
    rvec cent, diff, shift;

    /* center of geometry */
    get_cog(x, index, isize, cent);

    /* shift geomtric center to orgin */
    rvec_sub(target, cent, shift);
    for (i=0; i<natoms; i++)
    {
        rvec_inc(x[i], shift);
    }
    if (cog)
    {
        copy_rvec(cent, cog);
    }
}


/* Move the center of the bounding sphere to target */
static void
mv_boundingSphere_to_rvec(int natoms, rvec x[], atom_id *index, int isize, rvec target, gmx_bool bVerbose, rvec centBS)
{
    int i;
    rvec center, shift;
    real R = 0;
    static int bFirst = TRUE, bCentCOM = FALSE;


    if (bFirst)
    {
        bFirst = FALSE;
        if (getenv("GMX_WAXS_BOUNDING_SPHERE_METHOD") != NULL)
        {
            /* fprintf(stderr,"GMX_WAXS_BOUNDING_SPHERE_METHOD has been deprecated. It will no longer function.\n");*/
            bCentCOM = TRUE;
        }
    }

    if (!bCentCOM)
    {
        get_bounding_sphere(x, index, isize, center, &R, bVerbose);
    }
    else
    {
        get_cog(x, index, isize, center);
    }

    rvec_sub(target, center, shift);
    for (i=0; i<natoms; i++)
    {
        rvec_inc(x[i], shift);
    }
    if (centBS)
    {
        copy_rvec(center, centBS);
    }
}

void
mv_boxcenter_to_rvec(int natoms, rvec x[], rvec cent, matrix box)
{
    int i;
    rvec box_center, diff;

    calc_box_center(ecenterTRIC, box, box_center);
    rvec_sub(cent, box_center, diff);
    for (i = 0; i < natoms; i++)
    {
        rvec_inc(x[i], diff);
    }
}


/* This is adopted from rm_gropbc() in src/gmxlib/rmpbc.c
   A simple routine to make the protein whole based on the disnace between number-wise
   atomic neighbors. */
static void
waxs_rm_gropbc(rvec *x, atom_id *index, int isize, matrix box)
{
    real dist, absdist;
    int  n, m, d;
    const real maxAllowedDist = 2.0;
    char *dimLett[] = {"X", "Y", "Z"};

    /* check periodic boundary */
    for (n = 1; (n < isize); n++)
    {
        for (m = DIM-1; m >= 0; m--)
        {
            dist = x[index[n]][m]-x[index[n-1]][m];
            if (fabs(dist) > 0.5*box[m][m])
            {
                if (dist >  0)
                {
                    for (d = 0; d <= m; d++)
                    {
                        x[index[n]][d] -= box[m][d];
                    }
                }
                else
                {
                    for (d = 0; d <= m; d++)
                    {
                        x[index[n]][d] += box[m][d];
                    }
                }
            }
            /* Check if the two atoms are now close to each other */
            absdist = fabs(x[index[n]][m]-x[index[n-1]][m]);
            if (absdist > maxAllowedDist || absdist > 0.4*box[m][m])
            {
                gmx_fatal(FARGS, "Trying to make the solute whole based on distances between number-wise atom\n"
                          "neighbors. However, found a distance of of %g between atoms %d and %d in direction %s\n", fabs(dist),
                          index[n], index[n-1], dimLett[m]);
            }
        }
    }
}

static void
waxs_mk_prot_whole(rvec *x, atom_id *index, int isize, atom_id *ipbcatom, int npbcatom, t_pbc *pbc,
                   gmx_bool bVerbose)
{
    int i, d, ipbc_global, ipbc_global0;
    rvec x_pbc,*ptr_x, dx;
    rvec max={-1e20, -1e20, -1e20}, min={ 1e20,  1e20,  1e20};

    waxs_debug("Begin of waxs_mk_prot_whole()\n");

    clear_rvec(x_pbc);
    /* ipbcatom is a global index, independent of *index. */
    /* pbcatoms are a set of atoms that are close together to ensure a correct and consistent box centering from the solute frame */
    /* The first atom is used here to define a center. */
    for (i = 0; i < npbcatom; i++)
    {
        /* Note: Out of bounds checks should have been done in tpr generation. */
        ipbc_global = ipbcatom[i];
        /* Locate shortest pbc-distance of this sub-collection.*/
        if (bVerbose)
        {
            fprintf(stderr,"pbcatom[%d] = %d, x[pbcatom[%d]] before-shift: %g %g %g .\n", i, ipbc_global, i,
                    x[ipbc_global][0], x[ipbc_global][1], x[ipbc_global][2]);
        }
        if (i > 0)
        {
            ipbc_global0 = ipbcatom[0];
            pbc_dx(pbc, x[ipbc_global], x[ipbc_global0], dx);
            rvec_add(x[ipbc_global0], dx, x[ipbc_global]);
            if (bVerbose)
            {
                fprintf(stderr,"pbcatom[%d] = %d, x[pbcatom[%d]] after-shift: %g %g %g .\n", i, ipbc_global, i,
                        x[ipbc_global][0], x[ipbc_global][1], x[ipbc_global][2]);
            }
        }
        /* Add the correct PBC image to the COM. */
        rvec_inc(x_pbc, x[ipbc_global]);
    }
    svmul(1.0/npbcatom, x_pbc, x_pbc);

    if (bVerbose)
    {
        fprintf(stderr,"x[pbc_center]: %g %g %g .\n", x_pbc[0], x_pbc[1], x_pbc[2]);
    }

    /* Shift the protein by dx */
    for (i = 0; i < isize; i++)
    {
        ptr_x = &x[index[i]];
        pbc_dx(pbc, *ptr_x, x_pbc, dx);
        rvec_add(x_pbc, dx, *ptr_x);
    }
    waxs_debug("End of waxs_mk_prot_whole()\n");
}

#define WAXS_WRITE_FRAME_PDB_DEBUG(req, fname) {                        \
        if (req){                                                       \
            sprintf(fn, fname"_%d.pdb", wr->waxsStep);                 \
            write_sto_conf_mtop(fn, *mtop->name, mtop, x, NULL, ePBC, box); \
            printf("Wrote %s\n", fn); }                                 \
    }

/* Do the required steps with the solute before getting the solvation shell:
   1. Make solute/protein whole, using pbcatom if given.
   2. If requested, shift fit-group COG to origin, calculate rotation matrix of the solute
      (using the fitgroup atoms such as Backbone), and store the inverse rotation matrix.
   2.5. Build envelope, if needed. First step: Calculate envelope bounding sphere. Allways: Check if envelope fits into box.
   3. Move the center of the envelope in the reference frame of the protein to the center of the box.
   4. Check distance between solute and box walls.
   5. put all solvent atoms into the compact box.
   6. If requested, move fit-group cog to center and do a rotation of the solute.
   7. Move the solute COG to the origin (origin).
*/
void
waxs_prepareSoluteFrame(t_waxsrec *wr, gmx_mtop_t *mtop, rvec x[], matrix box, int ePBC,
                        FILE *fpLog, matrix Rinv)
{
    matrix      Rfit = {{1,0,0}, {0,1,0}, {0,0,1}};
    int         i, j, d, nWarnCloseBoxLast;
    rvec        tmpvec2, box_center, cog, env_cent_solv, protRefInBox, env_rgb = {0.8, 0.8, 0.8}, env_cent;
    char        fn[1024];
    real        min = 1e20, bvec, env_R2;
    const char *warn = NULL;
    t_pbc       pbc;
    auto xArrayRef = gmx::arrayRefFromArray(reinterpret_cast<gmx::RVec *>(x), mtop->natoms);


    if (ePBC >= 0 && ePBC != epbcXYZ)
    {
        gmx_fatal(FARGS, "The PBC of your pure-solvent system should be %s, but I found %s\n",
                  epbc_names[epbcXYZ], epbc_names[ePBC]);
    }
    set_pbc(&pbc, ePBC, box);

    /**** 0 ****/
    WAXS_WRITE_FRAME_PDB_DEBUG( wr->debugLvl>2 || (wr->debugLvl == 2 && wr->waxsStep == 0), "before_shift");

    /**** 1 ****/
    if (wr->bHaveWholeSolute == FALSE)
    {
        if (wr->nSoluteMols == 1 && wr->pbcatoms[0] < -1)
        {
            /* Make whole using distances number-wise atomic neighbors */
            if (wr->waxsStep == 0)
            {
                printf("\nWAXS-MD: Will make solute whole using distances between number-wise atomic neighbors\n");
            }
            waxs_rm_gropbc(x, wr->indA_prot, wr->nindA_prot, box);
        }
        else
        {
            /* Make whole using waxs-pbc atom(s) */
            waxs_mk_prot_whole(x, wr->indA_prot, wr->nindA_prot, wr->pbcatoms, wr->npbcatom, &pbc, wr->debugLvl>1);
        }
    }
    else
    {
        if (wr->waxsStep == 0)
        {
            printf("\nNote: Assuming that the solute is already whole.\n");
        }
        if (!wr->bHaveFittedTraj)
        {
            /* With wr->bHaveWholeSolute == TRUE, we assume that the solute is whole in the COMPACT unit cell.
               So in case that the solute is boken in the triclinic/rectangular unit cell, first put atoms in the compact box */
            put_atoms_in_compact_unitcell(ePBC, ecenterTRIC, box, xArrayRef);
        }
        if (wr->waxsStep == 0)
        {
            WAXS_WRITE_FRAME_PDB_DEBUG( wr->debugLvl>2 || (wr->debugLvl == 1 && wr->waxsStep == 0), "withWholeSolute_checkIfWhole");
        }
    }

    WAXS_WRITE_FRAME_PDB_DEBUG( (wr->debugLvl>2 || (wr->debugLvl == 2 && wr->waxsStep == 0)), "protein_whole");

    /**** 2 ****/

    /* find the rotation matrix based on the fitgroup atoms */
    if (wr->bRotFit && !wr->bHaveFittedTraj)
    {
        reset_x(wr->nind_RotFit, wr->ind_RotFit, mtop->natoms, NULL, x, wr->w_fit);

        if (wr->waxsStep == 0 && !wr->bHaveRefFile )
        {
            fprintf(stderr,"Writing first frame into x_ref...\n");
            /* Store the reference coordinates */
            for(j = 0; j < mtop->natoms; j++)
            {
                copy_rvec(x[j], wr->x_ref[j]);
            }
            /* Inverse matrix is the idendity matrix */
            copy_mat(Rfit, Rinv);
        }
        else
        {
            calc_fit_R(DIM, mtop->natoms, wr->w_fit, wr->x_ref, x, Rfit);
            /* Store the inverse-fit to rotate forces. R^T = R^-1 for rotation matrices */
            transpose(Rfit, Rinv);
        }
        if (wr->debugLvl > 0)
        {
            fprintf(fpLog, "       [ %10g %10g %10g ]\nRfit = [ %10g %10g %10g ]\n       [ %10g %10g %10g ]\n",
                    Rfit[0][0], Rfit[0][1], Rfit[0][2],
                    Rfit[1][0], Rfit[1][1], Rfit[1][2],
                    Rfit[2][0], Rfit[2][1], Rfit[2][2]);
        }
        WAXS_WRITE_FRAME_PDB_DEBUG(wr->debugLvl > 2 || (wr->debugLvl >1 && wr->waxsStep == 0), "after_to_rotmat");
    }
    else
    {
        /* Inverse matrix is the idendity matrix */
        copy_mat(Rfit, Rinv);
    }

    /**** 2.5 ****/

    /* Envelope must now be built before starting mdrun */
    if (!gmx_envelope_bHaveSurf(wr->wt[0].envelope))
    {
        gmx_fatal(FARGS, "No envelope was defined. Provide the envelope and the envelope reference coordinate file with\n"
                  "environment variables GMX_ENVELOPE_FILE and GMX_WAXS_FIT_REFFILE\n");
        /* mv_cog_to_rvec(mtop->natoms, x, wr->indA_prot, wr->nindA_prot, wr->origin, NULL); */
        /* gmx_envelope_buildEnvelope(wr->wt[0].envelope, x, wr->indA_prot, wr->nindA_prot, wr->solv_lay, */
        /*                            GMX_ENVELOPE_SMOOTH_SIGMA); */
        /* gmx_envelope_writePymolCGO(wr->wt[0].envelope, "envelope.py", "envelope", env_rgb, env_rgb, 0.5); */
        /* gmx_envelope_writeToFile(wr->wt[0].envelope, "env.dat"); */
        /* WAXS_WRITE_FRAME_PDB_DEBUG(wr->debugLvl > 2 || (wr->debugLvl >1 && wr->waxsStep == 0), "build_envelope"); */
    }

    if (TRICLINIC(box))
    {
        /* If the box vectors are skewed (dodecahedron or so), then center the bounding sphere
           of the envelope at the center of the box */
        gmx_envelope_bounding_sphere(wr->wt[0].envelope, env_cent, &env_R2);
        if (wr->waxsStep == 0)
        {
            fprintf(stderr, "Radius of the envelopes bounding sphere = %g \n", sqrt(env_R2));
        }

        /* Check if envelope fits into box. */
        for (d = 0; d < DIM; d++)
        {
            bvec = 0.5 * sqrt(norm2(wr->local_state->box[d]));
            // printf("XXXXX %g %g %g %g\n", bvec, wr->local_state->box[d][0], wr->local_state->box[d][1], wr->local_state->box[d][2]);
            if (bvec < min)
            {
                min = bvec;
            }
        }
        if (env_R2 >= min*min)
        {
            if (!wr->bHaveFittedTraj)
            {
                fprintf(stderr, "\n"
                        "*******************************************************************************\n"
                        "WARNING, the Bounding sphere of envelope is larger than the smallest box vector\n"
                        "If your protein rotates, the envelope might not fit into the unit cell any more\n"
                        "So better choose a larger simulation box.\n"
                        "********************************************************************************\n");
            }
            fprintf(stderr, "   Radius of the bounding sphere = %g \n", sqrt(env_R2));
            fprintf(stderr, "   Max. radius of sphere fitting box = %g \n", sqrt(min));
        }
    }
    else
    {
        /* For a cubic or cuboid box (alpha = beta = gamma = 90 degree), center the maximum extension of
           the envlelope IN EACH DIRECTION (x/y/z) separately */
        gmx_envelope_center_xyz(wr->wt[0].envelope, Rinv, env_cent);
    }


    /**** 3 ****/
    clear_rvec(env_cent_solv);
    clear_rvec(protRefInBox);
    mvmul(Rinv, env_cent, env_cent_solv);
    calc_box_center(ecenterTRIC, box, box_center);
    rvec_sub(box_center, env_cent_solv, protRefInBox);
    if (wr->debugLvl >1 && wr->waxsStep == 0)
    {
        fprintf(stderr, "\nBox center: %g %g %g",box_center[0],box_center[1],box_center[2]);
        fprintf(stderr, "\nCog center: %g %g %g",protRefInBox[0],protRefInBox[1],protRefInBox[2]);
    }
    mv_cog_to_rvec(mtop->natoms, x, wr->indA_prot, wr->nindA_prot, protRefInBox, NULL);

    WAXS_WRITE_FRAME_PDB_DEBUG(wr->debugLvl > 2 || (wr->debugLvl >1 && wr->waxsStep == 0), "envcenter_to_boxcenter");

    /**** 4 ****/

    /* Safety check: are solute atoms are far enough from the box walls? */
    nWarnCloseBoxLast = wr->nWarnCloseBox;
    check_prot_box_distance(x, wr->indA_prot, wr->nindA_prot, box, wr, TRUE, NULL);
    WAXS_WRITE_FRAME_PDB_DEBUG(wr->nWarnCloseBox > nWarnCloseBoxLast && wr->debugLvl > 1, "box_dist_warning");

    /* Assert that all solue atoms are now inside the compact unit cell */
    if (! assert_all_atoms_inside_compact_box(x, wr->indA_prot, wr->nindA_prot, box, &pbc))
    {
        WAXS_WRITE_FRAME_PDB_DEBUG(TRUE, "atoms_outside_box_error");
        gmx_fatal(FARGS, "Some atoms are outside of the compact box - check error message above for more details.\n"
                  "Wrote problematic frame into atoms_outside_box_error_%d.pdb\n",  wr->waxsStep);
    }

    /* Check if envelope will fit into Compact Box */
    if (! gmx_envelope_bInsideCompactBox(wr->wt[0].envelope, Rinv, box, protRefInBox, &pbc, TRUE, 0.))
    {
        gmx_fatal(FARGS, "The envelope does not fit into the compact unitcell\n"
                  "Choose larger simulation box\n");
    }

    /**** 5 ****/
    if (!wr->bHaveFittedTraj)
    {
        put_atoms_in_compact_unitcell(ePBC, ecenterTRIC, box, xArrayRef);
    }

    WAXS_WRITE_FRAME_PDB_DEBUG ( wr->debugLvl > 2 || (wr->debugLvl > 1 && wr->waxsStep == 0), "all_in_box");

    /**** 6 ****/
    /* rotate system */
    if (wr->bRotFit && !wr->bHaveFittedTraj)
    {
        reset_x(wr->nind_RotFit, wr->ind_RotFit, mtop->natoms, NULL, x, wr->w_fit);
        WAXS_WRITE_FRAME_PDB_DEBUG(wr->debugLvl > 2 || (wr->debugLvl >1 && wr->waxsStep == 0), "before_rotate");
        if (wr->waxsStep == 0 || wr->bHaveRefFile )
        {
            for(j = 0; j < mtop->natoms; j++)
            {
                /* v' = R(v0-c0)' ; Rotate. Centered around COM (0,0,0) */
                mvmul(Rfit, x[j], tmpvec2);
                copy_rvec(tmpvec2, x[j]);
            }
        }
        WAXS_WRITE_FRAME_PDB_DEBUG(wr->debugLvl > 2 || (wr->debugLvl >1 && wr->waxsStep == 0), "after_rotate");
    }

    /**** 7 ****/
    /* Move all atoms such that the protein is at the origin, so the protein is inside the envelope */
    mv_cog_to_rvec(mtop->natoms, x, wr->indA_prot, wr->nindA_prot, wr->origin, NULL);

    WAXS_WRITE_FRAME_PDB_DEBUG (wr->debugLvl > 2 || (wr->debugLvl > 1 && wr->waxsStep == 0), "in_center");
    waxs_debug("Centering of solute box done.\n");
}


/* Preparing a frame 'ws->xPrepared' of the pure-solvent simulation:
   1. If the waxs step is larger than the number of frames: shift the water system so we get an independent 
      water configuration inside the envelope.
   2. Move waterbox to the center of the envelope.
   3. Assert that the enevelope fully fits into the box.

   On exit, ws->xPrepared is on the envelope.
*/
void
preparePureSolventFrame(t_waxs_solvent ws, int waxsStep, gmx_envelope_t envelope, int debugLvl)
{
    int              iWaterframe, natoms;
    real             rBoundingSphere;
    rvec             envelopeCenter, waterRefInBox, box_center;
    t_pbc            pbc;
    matrix           box, idM = {{1,0,0}, {0,1,0}, {0,0,1}};

    iWaterframe = waxsStep % ws->nframes;

    /* Water frames used */
    natoms      = ws->mtop->natoms;

    if (!ws->xPrepared)
    {
        gmx_incons("Error in preparePureSolventFrame, memory for ws->xPrepared not yet allocated.");
    }
    if (ws->ePBC >=0 && ws->ePBC != epbcXYZ)
    {
        gmx_fatal(FARGS, "The PBC of your pure-solvent system should be %s, but I found %s (%d)\n",
                  epbc_names[epbcXYZ], epbc_names[ws->ePBC], ws->ePBC);
    }

    /* Make a copy of the solvent coordinates of the used frame */
    memcpy(ws->xPrepared, ws->x[iWaterframe], natoms*sizeof(rvec));

    /* ...and a local pointer to the box */
    copy_mat(ws->box[iWaterframe], box);

    /* Shift water box when waxsStep >= ws->nframes, (to get independent pure-water frame) */
    if (waxsStep >= ws->nframes)
    {
        real rshift;
        rvec tmpvec;

        rshift  = (waxsStep/ws->nframes)*WAXS_SHIFT_WATERFRAME_BOXRATIO;
        rshift -= floor(rshift);
        int d;
        for (d = 0; d < DIM; d++)
        {
            tmpvec[d] = rshift * box[d][d];
        }
        // printf("Note: shifting water frame # %d coodinates by %g, %g, %g to get independent coordinates\n",
        //       iWaterframe, tmpvec[XX], tmpvec[YY], tmpvec[ZZ]);
        int i;
        for (i = 0; i < natoms; i++)
        {
            rvec_inc(ws->xPrepared[i], tmpvec);
        }
    }

    /*
     *  The envelope is always located around the origin. Hence, we need to shift the pure-solvent frame
     *  (which is in the unit cell) onto the envelope, such that the pure-solvent frame encloses the envelope
     *  as good as possible.
     */
    if (TRICLINIC(box))
    {
        /* With a tricilinic box, we shift the 'center of the solvent box' to the
           'center of the bounding sphere' of the envelope */
        gmx_envelope_bounding_sphere(envelope, envelopeCenter, &rBoundingSphere);
    }
    else
    {
        /* With a cuboid box, we shift the 'center of the solvent box' to the
           'center in x/y/z' of the envelope */
        gmx_envelope_center_xyz(envelope, idM, envelopeCenter);
    }

    /* All solvent atoms into the compact box */
    auto xPreparedArrayRef = gmx::arrayRefFromArray(reinterpret_cast<gmx::RVec *>(ws->xPrepared), natoms);
    put_atoms_in_compact_unitcell(ws->ePBC, ecenterTRIC, box, xPreparedArrayRef);

    /* Shift pure-solvent atoms to the center of the envelope */
    mv_boxcenter_to_rvec(ws->mtop->natoms, ws->xPrepared, envelopeCenter, box);

    /*
     * Check if envelope fits into the compact box
     */
    set_pbc(&pbc, ws->ePBC, box);
    calc_box_center(ecenterTRIC, box, box_center);
    rvec_sub(box_center, envelopeCenter, waterRefInBox);
    if (! gmx_envelope_bInsideCompactBox(envelope, idM, box, waterRefInBox, &pbc, TRUE, 0.))
    {
        gmx_fatal(FARGS, "The envelope does not fit into the compact unitcell for the PURE-SOLVENT box.\n"
                  "Provide a larger PURE-SOLVENT simulation system with -sw and -fw\n");
    }

    /* Write the shifted solvent coordinates for debugging */
    if (waxsStep == 0 && debugLvl > 1)
    {
        write_sto_conf_mtop("water_in_center_step0.pdb", *(ws->mtop->name),
                            ws->mtop, ws->xPrepared, NULL, ws->ePBC, box);
    }
    waxs_debug("Centering of solvent box done.\n");
}

void
read_fit_reference(const char* fn, rvec x_ref[], int nsys, atom_id* isol, int nsol, atom_id* ifit, int nfit)
{
    int nread, i;
    rvec *xtemp = NULL;
    char *dumtitle;
    t_atoms *dumatoms;
    matrix dummybox;

    snew(dumtitle, STRLEN);

    if (!x_ref)
    {
        snew(x_ref, nsys);
    }
    for (i=0; i<nsys;i++)
    {
        clear_rvec(x_ref[i]);
    }


    get_coordnum(fn, &nread);
    snew(dumatoms,1);
    init_t_atoms(dumatoms, nread, FALSE);

    if (nread == nsys)
    {
        /* Whole system */
        fprintf(stderr,"\nReading fit-reference coords of: system\n");
        gmx_gro_read_conf(fn, nullptr, &dumtitle, dumatoms, x_ref, nullptr, dummybox);
    }
    else if (nread == nsol)
    {
        /* Solute */
        snew(xtemp, nread);
        for (i=0; i<nsys;i++)
        {
            clear_rvec(x_ref[i]);
        }

        fprintf(stderr,"\nReading fit-reference coords of: solute\n");
        gmx_gro_read_conf(fn, nullptr, &dumtitle, dumatoms, xtemp, nullptr, dummybox);
        for (i=0; i<nread; i++)
        {
            copy_rvec(xtemp[i],x_ref[isol[i]]);
        }
    }
    else if (nread == nfit)
    {
        /* Fitgroup */
        snew(xtemp, nread);
        for (i=0; i<nsys;i++)
        {
            clear_rvec(x_ref[i]);
        }

        fprintf(stderr,"\nReading fit-reference coords of: fit-group\n");
        gmx_gro_read_conf(fn, nullptr, &dumtitle, dumatoms, xtemp, nullptr, dummybox);
        for (i=0; i<nread; i++)
        {
            copy_rvec(xtemp[i],x_ref[ifit[i]]);
        }
    }
    else
    {
        fprintf(stderr,"nread = %i. nfit = %i, nsol = %i, nSystem = %i.\n",
                nread, nfit, nsol, nsys );
        gmx_fatal(FARGS,"Incorrect number of atoms in the reference coordinreades used to fit to the WAXS envelope! Please provide either the whole system, the solute, or the fit group only.\n");
    }

    if (xtemp)
    {
        sfree(xtemp);
    }

    // Free t_atoms
    sfree(dumatoms->atomname);
    sfree(dumatoms->resinfo);
    sfree(dumatoms->atom);
    if (dumatoms->pdbinfo)
    {
        sfree(dumatoms->pdbinfo);
    }
    dumatoms->nr   = 0;
    dumatoms->nres = 0;

    sfree(dumtitle);
}

/* Return x,y and z from the cell index */
static void
waxs_ci2xyz(ivec gridn, int ci, ivec ind)
{
    ind[XX]  = ci / (gridn[YY]*gridn[ZZ]);
    ci      -= (ind[XX])*gridn[YY]*gridn[ZZ];
    ind[YY]  = ci / gridn[ZZ];
    ci      -= (ind[YY])*gridn[ZZ];
    ind[ZZ]  = ci;
}


/* return array containg the cell indices of cell ci *and* ci's unique neighbors.

   For a large simulation boxes, this will always be the central cell + 26 neighbors = 27 cells (3^3)
   For small system, however, there may be less neighbors (the cell to the left may be the same
   as the cell to the right. Therefore, return the number of unique neighbors
*/
static int
waxs_ciNeighbors(int ci, ivec gridn, int *cinb)
{
    ivec k, kk, ind;
    int ncells, d, j, cinew;

    waxs_ci2xyz(gridn, ci, ind);
    // fprintf(stderr," waxs_ciNeighbors: ci %d  ind = %d %d %d\n",ci,ind[XX],ind[YY],ind[ZZ]);

    ncells = 0;
    for (k[XX] = ind[XX]-1; k[XX] <= ind[XX]+1; k[XX]++)
    {
        for (k[YY] = ind[YY]-1; k[YY] <= ind[YY]+1; k[YY]++)
        {
            for (k[ZZ] = ind[ZZ]-1; k[ZZ] <= ind[ZZ]+1; k[ZZ]++)
            {
                /* Check boundaries */
                copy_ivec(k, kk);
                // fprintf(stderr," k = %d %d %d -- kk = %d %d %d\n", k[0],k[1],k[2], kk[0],kk[1],kk[2]);
                for (d=0; d<DIM; d++)
                {
                    /* make sure 0 <= kk < ngrid */
                    kk[d] = (kk[d] + gridn[d]) % gridn[d];

                    /* if (kk[d]<0) */
                    /* { */
                    /*     kk[d] += gridn[d]; */
                    /* } */
                    /* else if (kk[d]>=gridn[d]) */
                    /* { */
                    /*     kk[d] -= gridn[d]; */
                    /* } */
                }
                // fprintf(stderr," k = %d %d %d -- kk = %d %d %d\n", k[0],k[1],k[2], kk[0],kk[1],kk[2]);

                /* get the new cell index, and make sure we don't have the cell yet */
                cinew = gridn[YY]*gridn[ZZ]*kk[XX] + gridn[ZZ]*kk[YY] + kk[ZZ];
                for (j = 0; j<ncells; j++)
                {
                    if (cinew == cinb[j])
                    {
                        continue;
                    }
                }
                cinb[ncells] = cinew;
                ncells++;
                // fprintf(stderr," waxs_ciNeighbors: Found neighbor i=%d : %d\n",i,cinb[i]);
            }
        }
    }

    return ncells;
}

void
atom_types_rdfs_slow(int nFramesLocal, rvec *x[], matrix *box, int nTypes, int *isize, int *index[], double **rdf,
                     int **irdf, int nR, real rmax, gmx_bool bVerbose)
{
    int ifr, j, k, t1, t2, tt, k0, ePBC, ibin;
    t_pbc pbc;
    rvec dx;
    real r2, rmax2, r, dr;
    double incr;

    dr    = rmax/nR;
    rmax2 = gmx::square(rmax);

    if (nFramesLocal > 0)
    {
        ePBC  = guess_ePBC(box[0]);
    }
    for (ifr = 0; ifr < nFramesLocal; ifr++)
    {
        if (bVerbose)
        {
            printf("\rComputing RDFs, %5.1f%% done\n", 100.*ifr/nFramesLocal);
        }
        set_pbc(&pbc, ePBC, box[ifr]);

        /* Loop over atom types */
        for (t1 = 0; t1 < nTypes; t1++)
        {
            /* Loop over atom types */
            for (t2 = t1; t2 < nTypes; t2++)
            {
                /* Index of the RDF */
                tt = irdf[t1][t2];

                for (j = 0; j<isize[t1]; j++)
                {
                    if (t1 == t2)
                    {
                        /* Pairs of the same atom type (t1==t2), do the sum only once */
                        k0   = j+1;
                        incr = 2.;
                    }
                    else
                    {
                        k0   = 0;
                        incr = 1.;
                    }
                    for (k = k0; k<isize[t2]; k++)
                    {
                        // fprintf(stderr, "ifr = %d, t1 / t2 =  %d / %d, j,k = %d %d, indexj = %d, indexk = %d\n",ifr, t1,t2,j,k,index[t1][j],index[t2][k]);
                        pbc_dx(&pbc, x[ifr][index[t1][j]], x[ifr][index[t2][k]], dx);
                        r2 = norm2(dx);
                        if (r2 < rmax2)
                        {
                            r = sqrt(r2);
                            ibin = floor(r/dr);

                            if (ibin < nR && index[t1][j])
                            {
                                rdf[tt][ibin] += incr;
                                // n[tt]         += 1.;
                            }
                        }
                    }
                }
            }
        }
    }
    if (bVerbose)
    {
        printf("\n");
    }
}

void
atom_types_rdfs(int nFramesLocal, rvec *x[], matrix *box, int nTypes, int *isize, int *index[], double **rdf,
                int **irdf, int nR, real rmax, gmx_bool bVerbose)
{
    int ifr, j, k, **t = NULL, ePBC, ibin, tt, t1, t2, irdfDiff, ***indexgrid = NULL, **nAlloc, **nOnGrid;
    int iNeighbors[27], ci, cj, nb, atomci, atomcj, d, ngridTotal, nNeighbors;
    t_pbc pbc;
    rvec dx, veclen, xb, griddx;
    real dr, rmax2, r2, inv_vol;
    double r, *n;
    ivec ngrid, igrid;
    matrix transf, boxvecnorm;
    const int allocBlock = 1000;

    if (nFramesLocal > 0)
    {
        ePBC  = guess_ePBC(box[0]);
    }
    dr    = rmax/nR;
    rmax2 = gmx::square(rmax);

    if (bVerbose)
    {
        printf("Computing RDFs of pure-solvent system, doing ~ %d frames per node\n", nFramesLocal);
    }
    snew(indexgrid, nTypes);
    snew(nOnGrid,   nTypes);
    snew(nAlloc,    nTypes);
    for (ifr = 0; ifr < nFramesLocal; ifr++)
    {
        if (bVerbose)
        {
            printf("\rComputing RDFs, %5.1f%% done\n", 100.*ifr/nFramesLocal);
        }

        set_pbc(&pbc, ePBC, box[ifr]);

        /* Put atoms on a grid */
        for (d = 0; d<DIM; d++)
        {
            veclen[d] = sqrt(norm2(box[ifr][d]));
            ngrid [d] = floor(veclen[d]/rmax);
            griddx[d] = veclen[d]/ngrid[d];
        }
        ngridTotal = ngrid[XX]*ngrid[YY]*ngrid[ZZ];
        if (bVerbose && ifr == 0)
        {
            printf("Using %d cells (%d x %d x %d)\n", ngridTotal, ngrid[XX], ngrid[YY], ngrid[ZZ]);
        }

        /* Tranformation matrix from cartesian to box vector coordinates.
           First get nomralized box vectors: */
        for (d = 0; d < DIM; d++)
        {
            r = sqrt(norm2(box[ifr][d]));
            svmul(1./r, box[ifr][d], boxvecnorm[d]);
        }
        gmx::invertBoxMatrix(boxvecnorm, transf);

        for (t1 = 0; t1 < nTypes; t1++)
        {
            snew(indexgrid[t1], ngridTotal);
            snew(nOnGrid  [t1], ngridTotal);
            snew(nAlloc   [t1], ngridTotal);
            for (j = 0; j<isize[t1]; j++)
            {
                /* Transform x to box coordinates of the box vectors */
                mvmul(transf, x[ifr][index[t1][j]], xb);

                /* get the grid points for x */
                for (d = 0; d<DIM; d++)
                {
                    igrid[d] = round(xb[d]/griddx[d]);
                    while (igrid[d] < 0)
                    {
                        igrid[d] += ngrid[d];
                    }
                    while (igrid[d] >= ngrid[d])
                    {
                        igrid[d] -= ngrid[d];
                    }
                }
                /* get continuous grid number */
                ci = igrid[XX]*ngrid[YY]*ngrid[ZZ] + igrid[YY]*ngrid[ZZ] + igrid[ZZ];
                if (ci >= ngridTotal)
                {
                    gmx_fatal(FARGS, "Invalid igridcont %d (should be < %d)\n", ci, ngridTotal);
                }

                if (nAlloc[t1][ci] == nOnGrid[t1][ci])
                {
                    nAlloc[t1][ci] += allocBlock;
                    srenew(indexgrid[t1][ci], nAlloc[t1][ci]);
                }
                indexgrid[t1][ci][nOnGrid[t1][ci]] = index[t1][j];
                nOnGrid[t1][ci]++;
            }
        }

        /* Now compute the RDFs */
        /* Loop over atom types */
        for (t1 = 0; t1 < nTypes; t1++)
        {
            /* Loop over atom types */
            for (t2 = t1; t2 < nTypes; t2++)
            {
                /* Index of the RDF */
                tt = irdf[t1][t2];

                /* Loop over all grid points (or cells) */
                for (ci = 0; ci < ngridTotal; ci++)
                {
                    /* Get neighbors of cell ci */
                    nNeighbors = waxs_ciNeighbors(ci, ngrid, iNeighbors);

                    /* Loop over the cell ci *and* its unique neighbors (=27 only if cells << simulation box)  */
                    for (nb = 0; nb<nNeighbors; nb++)
                    {
                        /* cell index of the neighbor */
                        cj = iNeighbors[nb];
                        // printf("cell %d (%d atoms), doing its neighbor %d (%d atoms)\n", ci, nOnGrid[t1][ci], cj, nOnGrid[t2][cj]);

                        /* Loop over atoms of type t1 in cell ci */
                        for (k = 0; k < nOnGrid[t1][ci]; k++)
                        {
                            /* Atom index */
                            atomci = indexgrid[t1][ci][k];

                            /* Loop over the atoms of type t2 in cell cj */
                            for (j = 0; j < nOnGrid[t2][cj]; j++)
                            {
                                /* Atom index */
                                atomcj = indexgrid[t2][cj][j];

                                /* increment the RDF */
                                pbc_dx(&pbc, x[ifr][atomci], x[ifr][atomcj], dx);
                                r2 = norm2(dx);
                                if (r2 < rmax2)
                                {
                                    r = sqrt(r2);
                                    ibin = floor(r/dr);

                                    if (ibin < nR && atomci != atomcj)
                                    {
                                        rdf[tt][ibin] += 1.;
                                        // n[tt]         += 1.;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /* for (t1 = 0; t1 < nTypes; t1++) */
        /* { */
        /*     for (t2 = t1; t2 < nTypes; t2++) */
        /*     { */
        /*         tt = t[t1][t2]; */

        /*         set_pbc(&pbc, ePBC, box[ifr]); */

        /*         for (j = 0; j<isize[t1]; j++) */
        /*         { */
        /*             for (k = j+1; k<isize[t2]; k++) */
        /*             { */
        /*                 pbc_dx(&pbc, x[ifr][index[t1][j]], x[ifr][index[t2][k]], dx); */
        /*                 r2 = norm2(dx); */
        /*                 if (r2 < rmax2) */
        /*                 { */
        /*                     r = sqrt(r2); */
        /*                     ibin = floor(r/dr); */

        /*                     if (ibin<nR) */
        /*                     { */
        /*                         rdf[tt][ibin] += 1.; */
        /*                         n[tt]         += 1.; */
        /*                     } */
        /*                 } */
        /*             } */
        /*         } */
        /*     } */
        /* } */

        for (t1 = 0; t1 < nTypes; t1++)
        {
            for (j = 0; j < ngridTotal; j++)
            {
                sfree(indexgrid[t1][j]);
            }
            sfree(indexgrid[t1]);
            sfree(nAlloc[t1]);
            sfree(nOnGrid[t1]);
        }
    }
    sfree(indexgrid);
    sfree(nAlloc);
    sfree(nOnGrid);
    if (bVerbose)
    {
        printf("\n");
    }

}

void
rdf_sine_transform(double *rdf, int nR, real rmax, real qmax, int nq, double *sinetransf)
{
    int i, j;
    double qr, qd, dr, r, sum, fact, dq, q, gamma, sumg, sumgamma, f;

    dr = rmax/nR;
    dq = qmax/(nq-1);

    /* first determine gamma in eq. 2 of  */
    sumg = 0;
    sumgamma = 0;
    for (i = 0; i<nR; i++)
    {
        r = (i+0.5)*dr;
        sumg += r*r*rdf[i]*dr;
        sumgamma += r*r*dr;
    }
    //gamma = 3*sum/(rmax*rmax*rmax);
    gamma = sumg/sumgamma;
    f     = sumgamma/sumg;
    // printf("gamma = %g\n", gamma);

    for (j = 0; j < nq; j++)
    {
        sum = 0;
        q   = dq*j;
        for (i = 0; i < nR; i++)
        {
            r = (i+0.5)*dr;
            qr = q*r;
            if (qr > 0)
            {
                fact = sin(qr)/qr;
            }
            else
            {
                fact = 1.;
            }

            sum += r*r*(rdf[i]-1.)*fact*dr;
        }

        sinetransf[j] = 4*M_PI*sum;
        // printf("Filled sine %d (q = %g) transf by %g\n", j, q, 4*M_PI*sum);
    }
}

void
interpolate_solvent_intensity(double *Igiven, real qmax, int nq, t_waxsrecType *wt)
{
    int i, j;
    double dq, q, left, right, Ileft, Iright, slopeI, dqgiven;
    gmx_bool bFound;

    if (wt->maxq > qmax)
    {
        gmx_fatal(FARGS, "Maximum q passed in mdp file (%g) is larger than the maximum q at which\n"
                  "the pure-solvent intensity is available *%g)\n", wt->maxq, qmax);
    }

    dqgiven = qmax/(nq-1);
    dq      = (wt->nq > 1) ? (wt->maxq-wt->minq)/(wt->nq-1) : 0.0 ;
    for (i = 0; i < wt->nq; i++)
    {
        q = wt->minq + i*dq;
        bFound = FALSE;
        /* Simple linear interpolation. Maybe we can improve this later. */
        j = floor(q/dqgiven);
        if (j == nq-1)
        {
            /* q is exactly == qmax (it cannot be larger, since we checke above) */
            wt->Ipuresolv[i] = Igiven[j];
        }
        else if (j < nq-1)
        {
            left     = j*dqgiven;
            right    = (j+1)*dqgiven;
            Ileft    = Igiven[j];
            Iright   = Igiven[j+1];
            slopeI   = (Iright-Ileft) / (right-left);
            wt->Ipuresolv[i] = Ileft   +(q-left)*slopeI;
        }
        else
        {
            gmx_fatal(FARGS, "Error in interpolate_solvent_intensity(), this should not happen\n");
        }
    }

    /* debugging output */
    FILE *fp = fopen("Ibuffer_interp.dat","w");
    for (i = 0; i<wt->nq; i++)
    {
        fprintf(fp, "%g %g\n", wt->minq + i*dq, wt->Ipuresolv[i]);
    }
    fclose(fp);

}

void
read_pure_solvent_intensity_file(const char *fn, t_waxsrecType *wt)
{
    int i, ncol, nlines;
    double **y;

    nlines = read_xvg(fn, &y, &ncol);
    if (ncol < 2)
    {
        gmx_fatal(FARGS, "Found only %d columns in %s\n", ncol, fn);
    }
    if (nlines < 3)
    {
        gmx_fatal(FARGS, "Found only %d lines in %s\n", nlines, fn);
    }

    if (y[0][0] > wt->minq)
        gmx_fatal(FARGS,"Smallest q-value in %s is %g, but the smallest q requested is %g. Provide a different\n "
                  "scattering intensity file or increase waxs-startq in the mdp file\n", fn, y[0][0], wt->minq);
    if (y[0][nlines-1] < wt->maxq)
        gmx_fatal(FARGS,"Largest q-value in %s is %g, but the largest q requested is %g. Provide a different\n "
                  "scattering intensity file or decrease waxs-endq in the mdp file\n", fn, y[0][nlines-1], wt->maxq);

    /* Check if we have increasing equally-spaced q values */
    for (i = 1; i<nlines-1; i++)
    {
        if ( fabs( (y[0][i]-y[0][i-1]) - (y[0][i+1]-y[0][i]) ) > 1e-4)
        {
            // printf("i = %d, y = %g diffs %g %g\n", i, y[1][i], (y[1][i]-y[1][i-1]), (y[1][i+1]-y[1][i]));
            gmx_fatal(FARGS, "The q values in file\n%s\nare not in equally-spaced increasing order\n", fn);
        }
    }
    if (fabs(y[0][0]) > 1e-5)
    {
        gmx_fatal(FARGS, "The q values in %s don't start at zero\n");
    }

    printf("Read %d pure-solvent intensities between q = %g and %g\n", nlines, y[0][0], y[0][nlines-1]);
    interpolate_solvent_intensity(y[1], y[0][nlines-1], nlines, wt);

    sfree(y[0]);
    sfree(y[1]);
    sfree(y);
}

void
do_pure_solvent_intensity(t_waxsrec *wr, t_commrec *cr, gmx_mtop_t *mtop, rvec **xSolv, int nAtomsTotal, matrix *boxSolv, int nFramesRead,
                          int *cmtypesList, int nTypes, double qmax, int nq, double *intensitySum, char *fnIntensity, int t)
{
    int    *nodeIdFrame = NULL, nFramesLocal = 0, nFramesReceived, nRDFs, ifr, *cmtypesList_B = NULL, ncmtypesList_B = 0;
    int    **index = NULL, *isize = NULL, *nAlloc = NULL, **irdf = NULL, i, j, tt, t1, t2, irdfDiff;
    int    nrow, nFramesUsed = -1;
    real   inv_vol, ri, ro, dr, dv, dens1, dens2, tmp, f1, f2, r, q;
    rvec   **x = NULL;
    matrix *box = NULL;
    char   *buf, *fn;
    double **rdf = NULL, **sinetransf = NULL, **intensity = NULL, *Icalc = NULL;
    MPI_Status     stat;
    gmx_bool       bFound;
    FILE           *fp;

    const int allocBlockSize = 100;
    const double nAtomsTimesFrames = 20e5;
    real rMax        = 1.6;
    real fade        = 0.9;
    int  nR          = 1001;

    if (MASTER(cr))
    {
        /* Specify the number of frames we want to use to compute the RDFs.
           Note: nAtomsTotal include all atoms including Tip4p/Tip5p dummy atoms, therefore nAtomsTotal >= wr->nindB_sol */
        nFramesUsed = nAtomsTimesFrames/wr->nindB_sol;
        if (nFramesUsed == 0)
            nFramesUsed = 1;
        printf("%d atoms in the solvent box - averaging over %d frames should give converged RDFs.\n",
               wr->nindB_sol, nFramesUsed);
        if (nFramesUsed > nFramesRead)
        {
            nFramesUsed = nFramesRead;
            printf("Warning, found only %d frames in solvent xtc. The RDFs may not fully converge\n", nFramesRead);
        }
        if ( (buf = getenv("GMX_WAXS_RDF_RMAX")) != NULL)
        {
            rMax = atof(buf);
            printf("Found envrionment variable GMX_WAXS_RDF_RMAX, computing RDFs up to %g nm\n", rMax);
        }
        if ( (buf = getenv("GMX_WAXS_RDF_NPOINTS")) != NULL)
        {
            nR = atof(buf);
            printf("Found envrionment variable GMX_WAXS_RDF_NPOINTS, using %d bins for RDFs.\n", nR);
        }
        if ( (buf = getenv("GMX_WAXS_RDF_FADE")) != NULL)
        {
            fade = atof(buf);
            printf("Found envrionment variable GMX_WAXS_RDF_FADE, turning RDFs to 1 at %g.\n", fade);
        }
    }

    if (PAR(cr))
    {
        gmx_bcast(sizeof(real), &rMax, cr);
        gmx_bcast(sizeof(int), &nR, cr);
        gmx_bcast(sizeof(int), &nAtomsTotal, cr);
        gmx_bcast(sizeof(int), &nFramesUsed, cr);
        gmx_bcast(sizeof(int), &nTypes, cr);
    }

    if (MASTER(cr))
    {
        snew(index,  nTypes);
        snew(isize,  nTypes);
        snew(nAlloc, nTypes);
        /* First, make a list of the SF types in the pure solvent */
        for (i = 0; i < wr->nindB_sol; i++)
        {
            t1 = cmtypesList[wr->indB_sol[i]];

            /* Check if type t1 is already in the list cmtypesList_B[] */
            bFound = FALSE;
            for (j = 0; j<ncmtypesList_B; j++)
            {
                if (cmtypesList_B[j] == t1)
                {
                    bFound = TRUE;
                    break;
                }
            }
            /* If t1 is not yet in the list, add it */
            if (!bFound)
            {
                srenew(cmtypesList_B, ncmtypesList_B+1);
                cmtypesList_B[ncmtypesList_B] = t1;
                ncmtypesList_B++;
            }
        }
        if (ncmtypesList_B != nTypes)
        {
            gmx_fatal(FARGS, "Found %d SF types in the topology of the pure-water system, but only %d in the solvent group\n",
                      nTypes, ncmtypesList_B);
        }

        /* Make index groups of atoms of the same scattering factor type
           (for pure water this would be a list of O (j==0) and a list of H (j==1) atoms) */
        for (i = 0; i < wr->nindB_sol; i++)
        {
            t1 = cmtypesList[wr->indB_sol[i]];
            for (j = 0; j<ncmtypesList_B; j++)
            {
                if (cmtypesList_B[j] == t1)
                {
                    bFound = TRUE;
                    break;
                }
            }
            if (j >= nTypes)
            {
                gmx_fatal(FARGS, "Invalid SF type for pure solvent: %d, should be < %d\n", t1, nTypes);
            }
            if (nAlloc[j] == isize[j])
            {
                nAlloc[j] += allocBlockSize;
                srenew(index[j], nAlloc[j]);
            }
            index[j][isize[j]] = wr->indB_sol[i];
            if (wr->indB_sol[i] >= nAtomsTotal)
            {
                gmx_fatal(FARGS, "Invalid atom number %d, max is %d (i = %d)\n", wr->indB_sol[i], nAtomsTotal, i);
            }
            isize[j]++;
        }
    }
    if (PAR(cr))
    {
        gmx_barrier(cr);
    }

    /* Send the index groups to the nodes */
    if (PAR(cr))
    {
        if (!MASTER(cr))
        {
            snew(index,  nTypes);
            snew(isize,  nTypes);
        }
        gmx_bcast(sizeof(int)*nTypes, isize, cr);
        if (!MASTER(cr))
        {
            for (j = 0; j < nTypes; j++)
            {
                snew(index[j], isize[j]);
            }
        }

        for (j = 0; j < nTypes; j++)
        {
            gmx_barrier(cr);
            gmx_bcast(sizeof(int)*isize[j], index[j], cr);
        }
        gmx_barrier(cr);
    }

    /* Distribute solvent coordinates and box size over the nodes */
    if (!PAR(cr))
    {
        x            = xSolv;
        box          = boxSolv;
        nFramesLocal = nFramesUsed;
    }
    else
    {
        if (cr->nodeid >= cr->nnodes-cr->npmenodes)
        {
            gmx_fatal(FARGS, "Inconsistency, found too large nodeid = %d\n", cr->nodeid);
        }
        /* Send frames to the nodes */
        snew(nodeIdFrame, nFramesUsed);
        for (i=0; i < nFramesUsed; i++)
        {
            nodeIdFrame[i] = i % cr->nodeid;
            if (nodeIdFrame[i] == cr->nodeid)
            {
                nFramesLocal++;
            }
        }
        snew(x,   nFramesLocal);
        snew(box, nFramesLocal);
        nFramesReceived = 0;
        if (PAR(cr))
        {
            gmx_barrier(cr);
        }
        if (MASTER(cr) && PAR(cr))
        {
            printf("\nSending solvent frames to the nodes ...");
            fflush(stdout);
        }
        for (i=0; i<nFramesUsed; i++)
        {
            if (nodeIdFrame[i] == 0)
            {
                if (MASTER(cr))
                {
                    /* If the master does this frame, there is no need to copy x and box */
                    x[nFramesReceived]   = xSolv[i];
                    copy_mat(boxSolv[i], box[nFramesReceived]);
                    nFramesReceived++;
                }
            }
            else
            {
                // fprintf(stderr, "node %d, frame %d\n", cr->nodeid, i);
                if (MASTER(cr))
                {
                    /*fprintf(stderr, "Sending frame %d\n", i);*/
                    MPI_Send((void*)xSolv[i], nAtomsTotal*sizeof(rvec), MPI_BYTE, nodeIdFrame[i], 0,
                             cr->mpi_comm_mysim);
                    MPI_Send((void*)(&boxSolv[i]), sizeof(matrix), MPI_BYTE, nodeIdFrame[i], 0,
                             cr->mpi_comm_mysim);
                }
                if (!MASTER(cr) && nodeIdFrame[i] == cr->nodeid)
                {
                    snew(x[nFramesReceived], nAtomsTotal);
                    MPI_Recv((void*)x[nFramesReceived], nAtomsTotal*sizeof(rvec), MPI_BYTE, 0, 0,
                             cr->mpi_comm_mysim, &stat);
                    MPI_Recv((void*)(&box[nFramesReceived]), sizeof(matrix), MPI_BYTE, 0, 0, cr->mpi_comm_mysim, &stat);
                    /*fprintf(stderr, "Received!! frame %d on node %d box = %g %g %g\n", i, cr->nodeid,
                      box[nFramesReceived][0][0],
                      box[nFramesReceived][1][1],box[nFramesReceived][2][2]);*/
                    nFramesReceived++;
                }
            }
        }
        if (MASTER(cr) && PAR(cr))
        {
            printf(" done\n");
        }
        if (PAR(cr))
        {
            gmx_barrier(cr);
        }
    }

    /* Number of RDFs to compute */
    nRDFs = nTypes + nTypes*(nTypes-1)/2;
    if (MASTER(cr))
    {
        printf("Found %d atom types in the pure-solvent system; this makes %d RDFs.\n", nTypes, nRDFs);
    }

    snew(rdf, nRDFs);
    for (i = 0; i < nRDFs; i++)
    {
        snew(rdf[i], nR);
    }

    /* Numbering of the rdfs */
    irdfDiff = 0;
    snew(irdf, nTypes);
    for (t1=0; t1<nTypes; t1++)
    {
        snew(irdf[t1], nTypes);
    }
    for (t1 = 0; t1 < nTypes; t1++)
    {
        for (t2 = t1; t2 < nTypes; t2++)
        {
            if (t1 == t2)
            {
                /* t1 == t2: We get the RDF between identical atom types (e.g., O-O or H-H):
                   Store those in the first nTypes entries of rdf[][] */
                irdf[t1][t2] = t1;
            }
            else
            {
                /* RDF between different atom types, e.g. O-H.
                   Store these *after* the first nTypes entries in rdf[][] */
                irdf[t1][t2] = nTypes + irdfDiff;
                irdfDiff++;
            }
        }
    }

    /* Compute the RDFs */
    atom_types_rdfs_slow(nFramesLocal, x, box, nTypes, isize, index, rdf, irdf, nR, rMax, MASTER(cr));

    /* Collect RDFs from the nodes */
    for (t1 = 0; t1 < nRDFs; t1++)
    {
        if (PAR(cr))
        {
            gmx_sumd(nR, rdf[t1], cr);
        }
    }

    /* Get average inverse volume and density */
    if (MASTER(cr))
    {
        inv_vol = 0.;
        for (i=0; i < nFramesUsed; i++)
        {
            inv_vol += 1./det(boxSolv[i]);
        }
        inv_vol /= nFramesUsed;
        for (t1 = 0; t1 < nTypes; t1++)
        {
            printf("Density of atom type %d = %g 1/nm3\n", t1, inv_vol*isize[t1]);
        }

        /* densities from RDFs */
        double *dens, sum;
        snew(dens, nTypes);
        dr = rMax/nR;
        for (t1 = 0; t1 < nTypes; t1++)
        {
            sum = 0;
            tt = irdf[t1][t1];
            for (i = 0; i<nR; i++)
            {
                r  = (i+0.5)*dr;
                /* Before normalizing the rdfs (see below), this gives simply the cumulative RDF: */
                sum += rdf[tt][i] / (nFramesUsed*isize[t1]);
            }
            dens[t1] = sum / (4*M_PI/3.*rMax*rMax*rMax);
            printf("Type %d, density from RDF = %g\n", t1, dens[t1]);
        }

        /* Normalize RDFs */
        dr = rMax/nR;
        for (i=0; i<nR; i++)
        {
            ri = dr*i;
            ro = dr*(i+1);
            r  = (ri+ro)/2;
            dv = 4./3.*M_PI * (ro*ro*ro-ri*ri*ri);
            dv = r*r*4*M_PI*dr;
            for (t1 = 0; t1 < nTypes; t1++)
            {
                for (t2 = t1; t2 < nTypes; t2++)
                {
                    tt = irdf[t1][t2];
                    rdf[tt][i] = rdf[tt][i] / dv / (inv_vol*isize[t1]*isize[t2]) / nFramesUsed;
                    if (fade > 0 && r >= fade)
                    {
                        /* Smoothly switch off RDF behind r = fade */
                        rdf[tt][i] = 1 + (rdf[tt][i]-1)*exp(-16*gmx::square(r/fade-1));
                    }
                }
            }
        }


        /* Do sine transform of RDFs: int_0^infty { r^2 [RDF-1] sin(qr)/(qr) } */
        snew(sinetransf, nRDFs);
        for (t1 = 0; t1 < nRDFs; t1++)
        {
            snew(sinetransf[t1], nq);
        }
        for (t1 = 0; t1 < nTypes; t1++)
        {
            for (t2 = t1; t2 < nTypes; t2++)
            {
                tt = irdf[t1][t2];
                rdf_sine_transform(rdf[tt], nR, rMax, qmax, nq, sinetransf[tt]);
            }
        }

        /* Compute intensity */
        snew(intensity, nRDFs);
        for (t1 = 0; t1 < nRDFs; t1++)
        {
            snew(intensity[t1], nq);
        }
        for (i = 0; i<nq; i++)
        {
            tmp = 0.;
            q   = 1.0*i*qmax/(nq-1);
            for (t1 = 0; t1 < nTypes; t1++)
            {
                for (t2 = t1; t2 < nTypes; t2++)
                {
                    /* atomic scattering factors */
                    f1 = CMSF_q(mtop->scattTypes.cm[cmtypesList_B[t1]], q);
                    f2 = CMSF_q(mtop->scattTypes.cm[cmtypesList_B[t2]], q);

                    dens1 = isize[t1]*inv_vol;
                    dens2 = isize[t2]*inv_vol;
                    //dens1 = dens[t1];
                    //dens2 = dens[t2];

                    tt = irdf[t1][t2];

                    /* eq. 2 in Koefinger & Hummer, Phys Rev E 87 052712 (2013) */
                    tmp = dens1*dens2*sinetransf[tt][i];
                    if (t1 == t2)
                    {
                        tmp += dens1;
                    }
                    if (t1 != t2)
                    {
                        /* We sum only once in case of different atom types, so we need to multiply by 2 */
                        tmp *= 2;
                    }
                    // printf("i %d t = %d / %d (tt %d); f = %g / %g; dens = %g / %g\n", i, t1, t2, tt, f1, f2, dens1, dens2);

                    intensity[tt][i] = f1*f2*tmp;
                    if (fabs(f1*f2*tmp) < 0.01)
                    {
                        printf("Strange int = %g, f1 %g f2 %g, tmp %g dens %g %g\n", intensity[tt][i],
                               f1, f2, tmp, dens1, dens2);
                    }
                    intensitySum[i] += f1*f2*tmp;
                }
            }
        }

        /* Interpolate intensity to the q values in the mdp file */
        interpolate_solvent_intensity(intensitySum, qmax, nq, &wr->wt[t]);
    }


    if (MASTER(cr))
    {
        fp = gmx_ffopen(fnIntensity, "w");
        for (i=0; i<nq; i++)
        {
            fprintf(fp, "%g %g\n", 1.0*i*qmax/(nq-1), intensitySum[i]);
        }
        gmx_ffclose(fp);
        printf("Wrote solvent intensity to %s\n", fnIntensity);
    }

    if (MASTER(cr))
    {
        fp = fopen("rdf.dat", "w");
        FILE *fpr = fopen("rdf_m1_r2.dat", "w");

        FILE *fp1 = fopen("sinetransf.dat", "w");
        FILE *fp2 = fopen("intensities.dat", "w");

        for (tt = 0; tt < nRDFs; tt++)
        {
            for (i=0; i<nR; i++)
            {
                fprintf(fp, "%g %g\n", (i+0.5)*rMax/nR, rdf[tt][i]);
                fprintf(fpr, "%g %g\n", (i+0.5)*rMax/nR, (rdf[tt][i]-1)*gmx::square((i+0.5)*rMax/nR));
            }
            fprintf(fp, "&\n");
            fprintf(fpr, "&\n");
            for (i=0; i<nq; i++)
            {
                fprintf(fp1, "%g %g\n",  1.0*i*qmax/(nq-1), sinetransf[tt][i]);
                fprintf(fp2, "%g %g\n",  1.0*i*qmax/(nq-1), intensity[tt][i]);
            }
            fprintf(fp1, "&\n");
            fprintf(fp2, "&\n");
        }

        fclose(fp);
        fclose(fpr);
        fclose(fp1);
        fclose(fp2);
    }

    if (PAR(cr))
    {
        if (!MASTER(cr))
        {
            for (i = 0; i<nFramesLocal; i++)
            {
                sfree(x[i]);
            }
        }
        if (x)
        {
            sfree(x);
            sfree(box);
        }
        sfree(nodeIdFrame);
    }

}


/* Return the radius of gyration of the solute only (without taking the solvation layer into account)
   Note: solute must be whole.
*/
real
soluteRadiusOfGyration(t_waxsrec *wr, rvec x[])
{
    int i, sft;
    rvec com, tmp, diff;
    real w, nelec, diff2, sum2, rg;

    /* Get electron-weighted center-of-mass */
    clear_rvec(com);
    w = 0;
    for (i = 0; i < wr->nindA_prot; i++)
    {
        nelec = wr->nElectrons[wr->indexA[i]];
        w    += nelec;
        svmul(nelec, x[wr->indA_prot[i]], tmp);
        rvec_inc(com, tmp);
    }
    svmul(1./w, com, com);

    /* Rg^2 = 1/N sum[i] w[i]*(r[i]-com)^2,
       where w[i] is the number of electrons, and N = sum[i] w[i] is the norm
    */
    sum2 = 0;
    for (i = 0; i < wr->nindA_prot; i++)
    {
        nelec = wr->nElectrons[wr->indexA[i]];
        rvec_sub(x[wr->indA_prot[i]], com, diff);
        sum2 += nelec*norm2(diff);
    }

    return sqrt(sum2/w);
}

double
guinierFit(t_waxsrecType *wt, double *I, double *varI, double RgSolute)
{
    const double qRgMin = 1.2;  /* Fitting up to q = 1.2/Rg */
    int n = 0, i;
    real dq;
    real *x, *y, *yerr, a, b, da, db, r, chi2, Rg, dRg, I0;
    gmx_bool bHaveErrors;

    dq       = (wt->nq > 1) ? (wt->maxq-wt->minq)/(wt->nq-1) : 0.0;
    if (RgSolute <= 0)
    {
        printf("WARNING, cannot do Guinier fit, because the aporoximate radius of gyration is unknown\n");
        return -1;
    }
    if (dq == 0.0)
    {
        printf("\n\nNOTE: cannot do Guinier fit, because only %d intensity points available in total\n", wt->nq);
        return -1;
    }

    while (wt->minq + n*dq <= qRgMin/RgSolute && n < wt->nq)
    {
        n++;
    }
    if (n > wt->nq)
    {
        n = wt->nq;
    }

    if (n < 2)
    {
        printf("\n\nNOTE: cannot do Guinier fit, because only %d intensity points available up to q = %g\n", n, qRgMin/RgSolute);
        return -1;
    }
    else if (n == 2)
    {
        printf("\n\nNOTE: only %d intensity points available for Guinier fit up to q = %g\n", n, qRgMin/RgSolute);
    }

    snew(x,    n);
    snew(y,    n);
    snew(yerr, n);

    bHaveErrors = TRUE;
    for (i=0; i<n; i++)
    {
        if (varI[i] <= 0.)
        {
            bHaveErrors = FALSE;
            break;
        }
    }

    for (i=0; i<n; i++)
    {
        x[i]    = gmx::square(wt->minq+i*dq);
        y[i]    = log(I[i]);
        yerr[i] = bHaveErrors ? sqrt(varI[i])/I[i] : 1.0;
    }

    /* Linear least-square fit y = ax+b */
    if (lsq_y_ax_b_error(n, x, y, yerr, &a, &b, &da, &db, &r, &chi2) != 0)
    {
        printf("WARNING, least-square fit for Guinier fit failed\n");
        return -1;
    }
    Rg  = sqrt(-3*a);
    dRg = -3*da/sqrt(-3*a);
    I0  = exp(b);
    if (bHaveErrors)
    {
        printf("\nGuinier fit: Rg [nm]   = %g +- %g", Rg, dRg);
        printf("\n             I(0) [e2] = %g +- %g", I0, exp(b)*db);
    }
    else
    {
        printf("\nGuinier fit: Rg [nm]   = %g", Rg);
        printf("\n             I(0) [e2] = %g", I0);
    }
    printf("\n             (from %d intensity points up to q = %g)\n\n", n, qRgMin/RgSolute);

    if (r < 0.95)
    {
        printf("\nWARNING, the Guinier fit was not successful, found a correlation on only %g\n", r);
    }

    sfree(x);
    sfree(y);
    sfree(yerr);

    return Rg;
}

/* Return average and variance in double prec. */
void
average_stddev_d(double *x, int n, double *av, double *sigma, double *wptr)
{
    int    i;
    double tmp, var, w, wsum;

    *av = wsum = 0;
    for (i = 0; i < n; i++)
    {
        w      = wptr ? wptr[i] : 1.;
        (*av) += w*x[i];
        wsum  += w;
    }
    (*av) /= wsum;

    var = 0;
    for (i = 0; i < n; i++)
    {
        w    = wptr ? wptr[i] : 1.;
        tmp  = x[i] - (*av);
        var += w*gmx::square(tmp);
    }
    var /= wsum;
    *sigma = sqrt(var);
}

/* return the average of x^2 in double prec. */
void
average_x2_d(double *x, int n, double *av, double *wptr)
{
    int    i;
    double w, wsum;

    *av = wsum = 0;
    for (i = 0; i < n; i++)
    {
        w = wptr ? wptr[i] : 1.;
        (*av) += w*(x[i]*x[i]);
        wsum  += w;
    }
    (*av) /= wsum;
}

/* return the average of x*y in double prec. */
void
average_xy_d(double *x, double *y, int n, double *av, double *wptr)
{
    int    i;
    double w, wsum;

    *av = wsum = 0;
    for (i = 0; i < n; i++)
    {
        w      = wptr ? wptr[i] : 1.;
        (*av) += w*(x[i]*y[i]);
        wsum  += w;
    }
    (*av) /= wsum;
}

/* return the average of x*y in double prec. */
void
sum_squared_residual_d(double *x, double *y, int n, double *chi2, double *wptr)
{
    int    i;
    double w;

    *chi2 = 0;
    for (i = 0; i < n; i++)
    {
        w      = wptr ? wptr[i] : 1.;
        (*chi2) += w * gmx::square((x[i] - y[i]));
    }
}

double
pearson_d(int n, double *x, double *y, double avx,
          double avy, double sigx, double sigy, double *wptr)
{
    int    i;
    double cov, w, wsum;

    cov = wsum = 0;
    for (i = 0; i < n; i++)
    {
        w     = wptr ? wptr[i] : 1.;
        cov  += w*(x[i]-avx)*(y[i]-avy);
        wsum += w;
    }
    cov /= wsum;
    return cov/(sigx*sigy);
}

double
sum_array_d(int n, double *x)
{
    int    i;
    double sum = 0;

    for (i = 0; i < n; i++)
    {
        sum += x[i];
    }
    return sum;
}


void
nIndep_Shannon_Nyquist(t_waxsrec *wr, rvec x[], t_commrec *cr, gmx_bool bVerbose)
{
    real R;
    rvec cent;
    int t;
    t_waxsrecType *wt;

    if (MASTER(cr))
    {
        if (bVerbose)
        {
            printf("Number of independent points (Nyquist-Shannon) =");
        }
        for (t = 0; t < wr->nTypes; t++)
        {
            wt = &wr->wt[t];

            /* First get maximum diameter from the simplified bounding sphere */
            get_bounding_sphere_Ritter_COM(x, wr->indA_prot, wr->nindA_prot, cent, &R, FALSE);

            /* Nyquist-Shannon: N[indep] = (qmax -qmin) D / pi, where D is the maximum diameter of the solute */
            wt->nShannon = (wt->maxq - wt->minq) * 2 * R / M_PI;
            if (wt->nShannon > wt->nq)
            {
                wt->nShannon = wt->nq;
            }
            if (bVerbose)
            {
                printf(" %g", wt->nShannon);
            }
        }
        if (bVerbose)
        {
            printf("\n");
        }
    }
    if (PAR(cr))
    {
        for (t = 0; t < wr->nTypes; t++)
        {
            wt = &wr->wt[t];
            gmx_bcast(sizeof(real), &wt->nShannon, cr);
        }
    }
}

/* Returns a continuous array (size mtop->natoms) of number of electrons */
double*
make_nElecList(gmx_mtop_t *mtop)
{
    int       mb, isum=0, nmols, imol, natoms_mol, iatom, itot, *scattTypeId, i, stype;
    int       ngrps_solute, ngrps_solvent, cmtype;
    t_atoms  *atoms;
    double    nElec, *nElecList;
    gmx_bool  bSolvent, bSolute;
    /* Bookkeeping: natoms = Sum _i ^ nmolblock ( molblock[i]->nmol * molblock[i]->natoms_mol ) */
    /* moltype.atoms->nr = molblock[i]->natoms_mol */

    ngrps_solute  = get_actual_ngroups( &(mtop->groups), egcWAXSSolute );
    ngrps_solvent = get_actual_ngroups( &(mtop->groups), egcWAXSSolvent );

    snew(nElecList, mtop->natoms);
    for (i = 0; i < mtop->natoms; i++)
    {
        nElecList[i] = -1000;
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
            cmtype = atoms->atom[iatom].cmtype;
            if (cmtype >= mtop->scattTypes.ncm || (cmtype < 0 && cmtype != NOTSET))
            {
                gmx_fatal(FARGS, "Invalid cmtype found (%d) in make_nElecList()\n", cmtype);
            }

            /* Loop over molecules of this molecule type */
            for (imol = 0; imol < nmols; imol++)
            {
                itot = isum + imol*natoms_mol + iatom;
                bSolvent = ( ggrpnr(&(mtop->groups),egcWAXSSolvent, itot) < ngrps_solvent );
                bSolute  = ( ggrpnr(&(mtop->groups),egcWAXSSolute,  itot) < ngrps_solute  );

                if ((bSolute || bSolvent) && cmtype < 0)
                {
                    gmx_fatal(FARGS, "Atom %d is in solute or solvent group, but the Cromer Mann type is %d\n",
                              itot, cmtype);
                }

                /* cmtype <0 means that we did not read Cromer-Mann parameters for this atom
                   (because it is not in solute or solvent) */
                if (cmtype >= 0)
                {
                    nElec = CROMERMANN_2_NELEC(mtop->scattTypes.cm[cmtype]);
                }
                else
                {
                    nElec = -1;
                }

                nElecList[itot] = nElec;
            }
        }
        isum += nmols*natoms_mol;
    }
    /* For debugging */
    for (i = 0; i < mtop->natoms; i++)
    {
        if (nElecList[i] == -1000)
        {
            gmx_fatal(FARGS, "nelec[%d] was not set in make_nElecList()\n",i);
        }
    }

    return nElecList;
}

void turnDownPosresForceConst(t_commrec *cr, t_waxsrec *wr, double simtime, t_idef *idef)
{
    int i, type, m;
    double this_simtime;
    real fForceSwitch, fc;
    rvec minus_dx;
    t_iparams *pr;
    static gmx_bool bPosresFinished = FALSE, bPosresStartTurningDown = FALSE;

    this_simtime = simtime - wr->simtime0;
    if (this_simtime > 2.0*wr->tau)
    {
        fForceSwitch = 0;
    }
    else if (this_simtime < wr->tau)
    {
        fForceSwitch = 1;
    }
    else
    {
        fForceSwitch = 0.5*(cos(M_PI*(this_simtime-wr->tau)/(wr->tau)) + 1.);
    }
    fc = fForceSwitch * wr->posres_fconst;
    if ( (((int)(simtime*1000)) % 1000) == 0 && MASTER(cr))
    {
        printf("\nSimtime = %g, posres force const now = %g\n", simtime, fc);
    }

    if (wr->tau > 0)
    {
        /* This loop structure was taken from posres() in bondfree.c */
        for (i = 0; (i < idef->il[F_POSRES].nr); )
        {
            type = idef->il[F_POSRES].iatoms[i++];
            i++;
            pr   = &idef->iparams_posres[type];

            for (m = 0; (m < DIM); m++)
            {
                pr->posres.fcA[m] = fc;
                pr->posres.fcB[m] = fc;
            }
        }
    }

    if (MASTER(cr) && bPosresFinished == FALSE && fForceSwitch == 0)
    {
        fprintf(stderr, "\n\nWAXS-MD: Position restraints turned down to zero at simtime time %g (%g since restart)\n\n",
                simtime, this_simtime);
        bPosresFinished = TRUE;
    }
    if (MASTER(cr) && bPosresStartTurningDown == FALSE && fForceSwitch < 1)
    {
        fprintf(stderr, "\n\nWAXS-MD: Now starring to turn down the position restraints at simtime %g (%g since restart)\n\n",
                simtime, this_simtime);
        bPosresStartTurningDown = TRUE;
    }
}

/*
 * Functions for getting the calculation times
 */

struct waxsTiming
{
    int      *nMeasurements;
    double   *sumTimes, *tStartThisMeasurement, *lastTime;
    gmx_bool *bCurrentlyMeasuring;
    int       N;
};

/* These names must correpond to the enum in waxsmd_utils.h */
const char* waxsTimingNames[waxsTimeNr] = {
    "SWAXS step", "Solute preparation", "Solvent preparation", "Scattering amplitudes",
    "One scattering amplitude", "grad I(q)", "Solvent Fourier Tr", "Gibbs sampling",
    "Potential/forces", "Updates", "Compute I / grad I(q)", "Solvent density corr",
};

t_waxsTiming waxsTimeInit()
{
    int i;
    t_waxsTiming t;

    snew(t, 1);

    t->N = waxsTimeNr;
    snew(t->nMeasurements,         t->N);
    snew(t->sumTimes,              t->N);
    snew(t->bCurrentlyMeasuring,   t->N);
    snew(t->tStartThisMeasurement, t->N);
    snew(t->lastTime,              t->N);

    return t;
}

void
waxsTimingWriteStatus(t_waxsTiming t, FILE *fp)
{
    int type;

    fprintf(fp, "\nStatus of SWAXS Computing Time accouting:\n");
    for (type = 0; type < t->N; type++)
    {
        fprintf(fp, "%2d) %-30s   n = %6d   sum = %10g   currently measuring = %d\n",
                type, waxsTimingNames[type],  t->nMeasurements[type], t->sumTimes[type],
                t->bCurrentlyMeasuring[type]);
    }
    fprintf(fp, "\n");
}

void waxsTimingDo(t_waxsrec *wr, int type, int action, double toStore, t_commrec *cr)
{
    struct timespec now;
    double          now_sec, delta_t_sec;
    t_waxsTiming    t = wr->compTime;

    /* Timing only done on master, and after a few steps to reduce the bias from
       initial Fourier transforms
    */
    if ( (wr->waxsStep < WAXS_STEPS_RESET_TIME_AVERAGES) || !MASTER(cr) )
    {
        return;
    }

    /* Some consistency checks */
    if (type > waxsTimeNr-1 || type < 0)
    {
        gmx_fatal(FARGS, "Invalid type in waxsTimingDo(): %d\n", type);
    }
    if ((toStore != 0) && (action != waxsTimingAction_add))
    {
        gmx_fatal(FARGS, "Incompatible comibination in waxsTimingDo(): action = %d - toStore = %g\n",
                  action, toStore);
    }
    if (action < 0 || action > waxsTimingAction_add)
    {
        gmx_fatal(FARGS, "Invalid action in waxsTimingDo(): %d\n", action);
    }

    /* fprintf(stderr, "waxsTimingDo: %d %s act %d cur %d\n", type, waxsTimingNames[type], action, t->bCurrentlyMeasuring[type]); */

    switch (action)
    {
    case waxsTimingAction_start:
        if (t->bCurrentlyMeasuring[type])
        {
            gmx_fatal(FARGS, "Trying to start a time measurement of %s, but I am measuring already\n", waxsTimingNames[type]);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &now);
        now_sec                         = 1.0*now.tv_sec + now.tv_nsec*1.0e-9;
        t->tStartThisMeasurement [type] = now_sec;
        t->bCurrentlyMeasuring   [type] = TRUE;
        break;

    case waxsTimingAction_end:
        if (! t->bCurrentlyMeasuring[type])
        {
            gmx_fatal(FARGS, "Trying to finish a time measurement of %s, but I am currently not measuring it.\n", waxsTimingNames[type]);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &now);
        now_sec                        = 1.0*now.tv_sec + now.tv_nsec*1.0e-9;
        delta_t_sec                    = now_sec - t->tStartThisMeasurement[type];
        t->sumTimes            [type] += delta_t_sec;
        t->lastTime            [type]  = delta_t_sec;
        t->nMeasurements       [type] ++;
        t->bCurrentlyMeasuring [type]  = FALSE;
        break;

    case waxsTimingAction_add:
        if (toStore == 0)
        {
            gmx_fatal(FARGS, "Error while storing computing time, found time = %g\n", toStore);
        }
        if (t->bCurrentlyMeasuring[type])
        {
            gmx_fatal(FARGS, "Trying to add a time for measurement of %s, but I am measuring already.\n", waxsTimingNames[type]);
        }
        t->sumTimes     [type] += toStore;
        t->nMeasurements[type] ++;
        t->lastTime     [type]  = toStore;
        break;

    default:
        gmx_fatal(FARGS, "Invalid action in waxsTiming(): %d\n", action);
        break;
    }
}

double waxsTimingGetLast(t_waxsTiming t, int type)
{
    if (type > waxsTimeNr-1 || type < 0)
    {
        gmx_fatal(FARGS, "Invalid type in waxsTiming(): %d\n", type);
    }
    return t->lastTime[type];
}

void
waxsTimingWrite(t_waxsTiming t, FILE *fp)
{
    int  type;
    char buf[STRLEN];

    fprintf(fp, "\n=============== Computing time Statistics ===============\n");
    for (type = 0; type < t->N; type++)
    {
        /* Write average comput time in milliseconds */
        sprintf(buf, "%-30s (n = %4d) [ms]", waxsTimingNames[type], t->nMeasurements[type]);
        print2log(fp, buf, "%g", (t->sumTimes[type]/t->nMeasurements[type])*1000.0);
    }
    fprintf(fp, "=========================================================\n");
}

void
waxsTimingWriteLast(t_waxsTiming t, FILE *fp)
{
    int  type;
    char buf[STRLEN];

    fprintf(fp, "\nLast compute times:\n");
    for (type = 0; type < t->N; type++)
    {
        /* Write average comput time in milliseconds */
        fprintf(fp, "%-30s (last of %5d) [ms] = %10.3g\n", waxsTimingNames[type],
                t->nMeasurements[type], t->lastTime[type]*1000.0);
    }
    fprintf(fp, "\n");
}



/*
 * END of timing functions
 */

/* Formatted writing to log file */
void
print2log(FILE *fp, char const *s, char *fmt, ...)
{
    va_list ap;
    char buf[STRLEN], *p, *p1 = fmt;
    int length = 75;

    /* write Text and dots */
    if (*s == '\n')
    {
        length++;
    }
    p  = buf;
    sprintf(p, "%s ", s);
    p += strlen(s) + 1;
    for (; (p-buf) < length-1; p++)
    {
        sprintf(p, ".");
    }
    sprintf(p++, " ");

    /* write number */
    va_start(ap, fmt);
    while (!isalpha(*(++p1)));
    switch(*p1)
    {
    printf("B p1 = %s\n", p1);

    printf("Now p1 = %c\b", *p1);
    case 'd':
        sprintf(p, fmt, va_arg(ap, int));
        break;
    case 'g':
        sprintf(p, fmt, va_arg(ap, double));
        break;
    case 's':
        sprintf(p, fmt, va_arg(ap, char*));
        break;
    default:
        gmx_fatal(FARGS, "Unsupported format character %c in print2log()\n", *p1);
    }
    va_end(ap);

    fprintf(fp, "%s\n", buf);
}
