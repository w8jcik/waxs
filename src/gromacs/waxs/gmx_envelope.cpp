/* TEST
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

#include "gmx_envelope.h"
#include "sysstuff.h"
#include "smalloc.h"
#include "string2.h"
#include "futil.h"
// #include "maths.h"
// #include "gmx_fatal.h"
#include "vec.h"
#include "md5.h"
#include "network.h"
#include "gmx_omp.h"
#include "pbc.h"
#include "gmx_miniball.h"


/*
 *  This source file was written by Jochen Hub.
 */

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

/* Macro names for AVX/SSE instructions */
#ifdef GMX_X86_AVX_256
/* AVX256 */
#ifdef GMX_DOUBLE
#define REGISTER __m256d
#define REGISTER_SINCOS gmx_mm256_sincos_pd
#else
#define REGISTER __m256
#define REGISTER_SINCOS gmx_mm256_sincos_ps

#endif

#elif defined(GMX_X86_AVX_128_FMA) || defined(GMX_X86_SSE4_1) || defined(GMX_X86_SSE2)
/* AVX_128_FMA, SSE2, or SSE4.1 */
#ifdef GMX_DOUBLE
#define REGISTER __m128d
#define REGISTER_SINCOS gmx_mm_sincos_pd
#else
#define REGISTER __m128
#define REGISTER_SINCOS gmx_mm_sincos_ps
#endif

#else
/* No acceleration */
#define GMX_WAXS_NO_ACCELERATION
#define REGISTER double
#endif

#ifdef GMX_DOUBLE
#define COS cos
#define SIN sin
#else
#define COS cosf
#define SIN sinf
#endif


// #define WAXS_ENVELOPE_NR_POINTS 50

/* # of recusions done to build icosphere.
   ->  # of faces = 20 * 4^N
*/
#define WAXS_ENVELOPE_ICOSPHERE_NREC 4

#define WAXS_ENVELOPE_GRID_DENSITY_SPACING 0.1

// #define ENVELOPE_DEBUG

#if defined( ENVELOPE_DEBUG )
#define DEBUG_PRINTF(x) printf x
#else
#define DEBUG_PRINTF(x)
#endif


/* Array to average the electron density on a grid enclosing the envelope,
   such we can visualize the electron density insdide the envelope
 */
typedef struct {
    dvec min, max;    /* lower-left & upper-right corners of cuboid volume */
    ivec n;            /* number of volume bins in x/y/z */
    int  N;            /* total number of volume bins */
    double dx;         /* grid spacing */
    double *dens;      /* density (array of size n[0]*n[1]*n[2]) */
    double norm;       /* normalization for cumulative averaging */
    double fac1, fac2; /* factors for cumulative average */
    int nstep;
    gmx_bool bOpen;    /* The grid is open for adding atoms - toggled by nextFrame / closeFrame functions */
} grid_density_t;

typedef struct{
    ivec v;            /* the 3 vertices IDs    */
    dvec normal;       /* normal vector         */
    double sprodLim;   /* smallest scalar product between the normal with each of the vertices.
                           Useful to quickly check if a vector crosses this face. */
} t_icosphere_face;

typedef struct{
    int nrec;
    int nvertex;
    int nface;                      /* # of faces in lowest recusion level == rec_nface[nrec-1] */
    dvec *vertex;
    t_icosphere_face *face;         /* shortcut to the faces of lowest recusion level */

    /* Also store all levels of faces, to facilitate a search */
    int *rec_nface;                 /* # of faces in recusion levels */
    t_icosphere_face **rec_face;    /* faces in previsous recursion levels */
} t_icosphere;

typedef struct {
    double theta, phi;
} t_spherical;

typedef struct{
    int ind[3];        /* ray indices of the corners        */
    int iface;         /* Index of icosphere face           */
    gmx_bool bDefined; /* triangle defined, that is, inner/outer of all 3 rays is defined? */
    dvec center;       /* Geometric center of the triangle  */
    dvec normal;       /* normal vector pointint outwards   */
    double c;            /* constant c of the plane: vec{x}*normal = c */
    double area_r1;      /* area of the triangle at radius 1  */
    double spaceangle;   /* space angle covered by this triangle */
    double volPyramid;   /* Volume of the pyramide               */
    double rmin, rmax;   /* distance of corner with the shortest/largest distance from the origin */
} t_triangle;

enum{ eEnvOut2In, eEnvIn2Out };

struct gmx_envelope{
    int nrec;                /* # of recursion levels for icosphere. */
    int nrays;               /* # of directions of envelope */
    double d;                  /* Build envelope at this distance from atoms */

    t_spherical *s;          /* spherical coordinates of envelope rays (theta/phi) */
    dvec *r;                 /* vectors in direction of envelope rays */

    /* Ray params. Written in gmx_envelope_setStats */
    double *inner, *outer;     /* inner and outer surface of envelope along rays, size nrays */
    gmx_bool *isDefined;     /* are inner and outer defined?, size nrays */
    int        nDefined;     /* # of rays with defined surface */
    gmx_bool bHaveEnv;       /* Have envelope constructed? */
    double minInner, maxInner, minOuter, maxOuter;
    double vol;                /* Volume of the envelope */
    unsigned char chksum[16];
    char chksum_str[17];

    /* Written in gmx_envelope_setStats. Not needed for checkpointing. */
    /* surface elements (triangles of envelope) - each triangle refers to one face of the ikosphere  */
    int nSurfElems;             /* # of faces/surface elements of icosphere                          */
    gmx_bool bOriginInside;     /* Is the origin within the envelope? -> surfElemInner = NULL        */
    t_triangle *surfElemOuter;  /* The outer and inner surfaces of the envelope                      */
    t_triangle *surfElemInner;  /* size nSurfElems */

    /* Partly set in gmx_envelope_binVolumes, called from setStats. ft_re and ft_im calculated later */
    /* Binning the envelope into small volume elements */
    int ngrid;            /* radial grid number for volume elements */
    double **binVol;        /* volumes of ngrid volume bins for each surface element, ngrid*nSurfElem */
    rvec **xBinVol;       /* central coordinates of the volume elements (required for Fourier Transform) */

    real *ftunit_re, *ftunit_im;  /* Fourier transform of a unit density in the envelope.
                                   * Used for the pure solvent sphere
                                   * The q vectors are provided externally and not known here.
                                   * Thus, the FT(q) of the local qs are stored */
    gmx_bool bHaveFourierTrans; /* FT already computed? */

    /* solvent density inside the envelope, around the solute in WAXS. */
    double **solventNelec;        /* # electrons in solvent array of: [# surfElems][ngrid] (essentially density) */
    int nSolventStep;             /* # of frames over which we averaged the density   */
    real *ftdens_re, *ftdens_im;  /* Fourier Transform of the density, nq(qhomenr on each node)
                                   * Used for the A solvent. */
    gmx_bool bHaveSolventFT;      /* FT of density already already computed?          */
    double solventNelecNorm;      /* Norm for cumulative average of density           */
    double solventNelec_fac1, solventNelec_fac2; /* factors for cumulative average    */
    double solventNelecTotal;     /* Total # of electrons in the solvent              */

    t_icosphere *ico;      /* The icosphere */

    grid_density_t *grid_density;    /* The averaged electron density on a normal grid */

    /* Called in gmx_envelope_setStats. */
    /* bounding sphere of envelope */
    dvec bsphereCent;      /* Center */
    double bsphereR2;      /* Radius^2 */

    gmx_bool bVerbose;
};



// typedef struct gmx_envelope *gmx_envelope_t;

static void
copy_rdvec(const rvec in, dvec out)
{
    out[XX] = in[XX];
    out[YY] = in[YY];
    out[ZZ] = in[ZZ];
}

static void
copy_drvec(const dvec in, rvec out)
{
    out[XX] = in[XX];
    out[YY] = in[YY];
    out[ZZ] = in[ZZ];
}

static gmx_inline double ddet(dvec a[DIM])
{
    return ( a[XX][XX]*(a[YY][YY]*a[ZZ][ZZ]-a[ZZ][YY]*a[YY][ZZ])
             -a[YY][XX]*(a[XX][YY]*a[ZZ][ZZ]-a[ZZ][YY]*a[XX][ZZ])
             +a[ZZ][XX]*(a[XX][YY]*a[YY][ZZ]-a[YY][YY]*a[XX][ZZ]));
}

static grid_density_t*
grid_density_init(dvec min, dvec max, double spacing)
{
    grid_density_t *g;
    int d;

    snew(g, 1);

    for (d = 0; d < DIM; d++)
    {
        g->min[d] = min[d];
        g->n  [d] = ceil(max[d]-min[d])/spacing;
        g->max[d] = g->min[d] + g->n[d]*spacing;
    }

    g->N     = g->n[XX]*g->n[YY]*g->n[ZZ];
    g->dx    = spacing;
    g->norm  = 0;
    g->nstep = 0;
    g->fac1  = g->fac2 = 0;
    snew(g->dens, g->N);
    printf("Initiated grid density (required %g MB), spacing %g nm\n", 1.0*g->N*sizeof(double)/1024/1024, spacing);
    printf("\tDensity grid range: x (nm): %12g - %12g : %d bins\n",   min[0], max[0], g->n[0]);
    printf("\t                    y (nm): %12g - %12g : %d bins\n",   min[1], max[1], g->n[1]);
    printf("\t                    z (nm): %12g - %12g : %d bins\n\n", min[2], max[2], g->n[2]);

    return g;
}

static void
grid_density_destroy(grid_density_t** gptr)
{
    sfree((*gptr)->dens);
    sfree(*gptr);
}

static void grid_density_nextFrame(grid_density_t* g,  double scale)
{
    int j;

    if (g->bOpen)
    {
        gmx_fatal(FARGS, "Trying to prepare grid density for next frame, but the frame is already open - did you not call grid_density_closeFrame() ?\n");
    }
    if (!g)
    {
        gmx_fatal(FARGS, "Error, trying update grid-based electron density, but the grid was not yet initiated\n");
    }

    /* Norm and factors for cumulative averaging: D[n] = fac1*D[n-1] + fac2*d[n] */
    g->norm = 1. + scale * g->norm;
    g->fac1 = 1.0*(g->norm - 1.)/g->norm;
    g->fac2 = 1/g->norm;

    for (j = 0; j<g->N; j++)
    {
        g->dens[j] *= g->fac1;
    }

    g->nstep++;
    g->bOpen = TRUE;
}

static void grid_density_addAtom(grid_density_t* g, const rvec xin, double nelec)
{
    dvec x;
    ivec ix;
    gmx_bool bInside;
    int j, d;

    if (!g->bOpen)
    {
        gmx_fatal(FARGS, "The grid density is not open for adding atsom. Call function grid_density_nextFrame() first.\n");
    }

    copy_rdvec(xin, x);

    /* index of volume bin */
    bInside = TRUE;
    for (d = 0; d < DIM; d++)
    {
        ix[d] = floor((x[d] - g->min[d])/g->dx);
        bInside &= (0 <= ix[d] && ix[d] < g->n[d]);
    }

    if (bInside)
    {
        j = ix[XX]*g->n[YY]*g->n[ZZ] + ix[YY]*g->n[ZZ] + ix[ZZ];
        if (j < 0 || j >= g->N)
        {
            gmx_fatal(FARGS, "Invalid bin while adding atom to grid density (ix = %d / %d / %d)\n", ix[XX], ix[YY], ix[ZZ]);
        }
        g->dens[j] += g->fac2 * nelec;
    }
}

void grid_density_closeFrame(grid_density_t* g)
{
    if (!g->bOpen)
    {
        gmx_fatal(FARGS, "Trying to close the grid density for adding atoms, by the grid is not open.\n");
    }
    g->bOpen = FALSE;
}

void grid_density_write(grid_density_t* g, const char *fn)
{
    FILE *fp;
    double densNorm;
    int ix, iy, iz, j, count = 0;
    dvec x;
    char *fnCube;

    if (g->bOpen)
    {
        gmx_fatal(FARGS, "Cannot write grid density to file while the grid is still open. Did you not call grid_density_closeFrame() ?\n");
    }

    fp = ffopen(fn, "w");
    for (ix = 0; ix < g->n[XX]; ix++)
    {
        for (iy = 0; iy < g->n[YY]; iy++)
        {
            for (iz = 0; iz < g->n[ZZ]; iz++)
            {
                j = ix*g->n[YY]*g->n[ZZ] + iy*g->n[ZZ] + iz;

                x[XX] = g->min[XX] + (1.0*ix + 0.5) * g->dx;
                x[YY] = g->min[YY] + (1.0*iy + 0.5) * g->dx;
                x[ZZ] = g->min[ZZ] + (1.0*iz + 0.5) * g->dx;
                densNorm = g->dens[j]/(g->dx*g->dx*g->dx);
                fprintf(fp, "%8.3f %8.3f %8.3f  %12.5g\n", x[XX], x[YY], x[ZZ], densNorm);
            }
        }
    }
    ffclose(fp);

    /* Writing in cube format */
    snew(fnCube, strlen(fn) + 10);
    sprintf(fnCube, "%s.cube", fn);
    fp = ffopen(fnCube, "w");
    fprintf(fp, "CUBE FORMAT, DENSITY INSIDE ENVELOPE\n");
    fprintf(fp, "NOTE: FACTOR OF 2 IN VECTORS AN ORIGIN IS NEEDED FOR CORRECT VISUALIZAION IN PYMOL\n");
    fprintf(fp, "%5d  %12.6f %12.6f %12.6f\n", 1, g->min[XX]*20, g->min[YY]*20, g->min[ZZ]*20);
    fprintf(fp, "%5d  %12.6f %12.6f %12.6f\n", g->n[XX], g->dx*20,       0.,      0.);
    fprintf(fp, "%5d  %12.6f %12.6f %12.6f\n", g->n[YY],       0., g->dx*20,       0.);
    fprintf(fp, "%5d  %12.6f %12.6f %12.6f\n", g->n[ZZ],       0.,       0., g->dx*20);
    /* Add one atom in corner, otherwise PyMol crashes */
    fprintf(fp, "%5d  %12.6f %12.6f %12.6f\n", 1, g->min[XX]*20, g->min[YY]*20, g->min[ZZ]*20);
    /* Now write all denities */
    count = 0;
    for (ix = 0; ix < g->n[XX]; ix++)
    {
        for (iy = 0; iy < g->n[YY]; iy++)
        {
            for (iz = 0; iz < g->n[ZZ]; iz++)
            {
                j        = ix*g->n[YY]*g->n[ZZ] + iy*g->n[ZZ] + iz;
                densNorm = g->dens[j]/(g->dx*g->dx*g->dx);
                fprintf(fp, "%12.6f ", densNorm);
                if (((count++) % 6) == 0)
                {
                    fprintf(fp, "\n");
                }
            }
        }
    }
    ffclose(fp);

}


static t_icosphere*
icosphere_init(int nrec)
{
    t_icosphere *ico;

    snew(ico, 1);
    ico->nrec = nrec;
    ico->nvertex = 0;
    ico->nface = 0;
    ico->face = NULL;
    ico->rec_face = NULL;
    ico->rec_nface = NULL;
    ico->vertex = NULL;
    return ico;
}
static void
icosphere_destroy(t_icosphere **icoptr)
{
    sfree((*icoptr)->face);
    sfree((*icoptr)->vertex);
    sfree(*icoptr);
}
static int
icosphere_add_vertex(t_icosphere *ico, double x, double y, double z)
{
    double len;

    len = sqrt(x*x + y*y + z*z);
    srenew(ico->vertex, ico->nvertex+1);
    ico->vertex[ico->nvertex][XX] = x/len;
    ico->vertex[ico->nvertex][YY] = y/len;
    ico->vertex[ico->nvertex][ZZ] = z/len;

    ico->nvertex++;
    return ico->nvertex-1;
}
static void
icosphere_add_face(t_icosphere *ico, int irec, int a, int b, int c)
{
    t_icosphere_face *thisface;
    int i;
    dvec tmp, v1, v2;
    double fact, minSprod = 1000;

    /* Add one face to recusion level irec */
    ico->rec_nface[irec] += 1;
    srenew(ico->rec_face[irec], ico->rec_nface[irec]);

    thisface = &(ico->rec_face[irec][ico->rec_nface[irec]-1]);
    thisface->v[0] = a;
    thisface->v[1] = b;
    thisface->v[2] = c;

    /* Store normal. Important: We cannot just add up the 3 vertices, since this is not
       a "gleichschenkliges" triangle */
    dvec_sub(ico->vertex[thisface->v[1]], ico->vertex[thisface->v[0]], v1);
    dvec_sub(ico->vertex[thisface->v[2]], ico->vertex[thisface->v[0]], v2);
    dcprod(v1, v2, tmp);
    fact =  (diprod(tmp, ico->vertex[thisface->v[0]]) < 0. ? -1. : 1.);
    dsvmul(fact/sqrt(dnorm2(tmp)), tmp, thisface->normal);

    /* Store the smallest scalar product between the normal and each of the verices */
    for (i=0; i<3; i++)
    {
        fact = diprod(ico->vertex[thisface->v[i]], thisface->normal);
        minSprod = (fact < minSprod) ? fact : minSprod;
    }
    thisface->sprodLim = minSprod;
    if (thisface->sprodLim > 1. || thisface->sprodLim < 0.3)
    {
        gmx_fatal(FARGS, "sprodlim = %g.\n", thisface->sprodLim);
    }


    /*printf("Added face %d: %d %d %d, normal = %g %g %g, sprodlim = %g \n", ico->nface, a, b, c, face->normal[0],
      face->normal[1], face->normal[2], face->sprodLim); */
    ico->nface++;
}
static void
icosphere_copy_face(t_icosphere_face *source,  t_icosphere_face *dest)
{
    copy_ivec(source->v,      dest->v);
    copy_dvec(source->normal, dest->normal);
    dest->sprodLim = source->sprodLim;
}
static int
icosphere_getMiddlePoint(t_icosphere *ico, int a, int b)
{
    int i, nface;
    dvec middle, tmp;
    double difflim2;

    dvec_add(ico->vertex[a], ico->vertex[b], middle);
    dsvmul(0.5, middle, middle);

    /* Get typical square of distance between vertices */
    nface   = round(20 * pow(4.,ico->nrec));
    difflim2 = 4*M_PI/nface / 10;

    /* First check if we have this one already */
    for (i=0; i<ico->nvertex; i++)
    {
        dvec_sub(ico->vertex[i], middle, tmp);
        if (dnorm2(tmp) < difflim2)
        {
            return i;
        }
    }

    /* If we don't have this middle point yet, make it */
    return icosphere_add_vertex(ico, middle[XX],  middle[YY],  middle[ZZ]);
}

static void
icosphere_buildIcosahedron(t_icosphere *ico)
{
    const double t = (1.0 + sqrt(5.0)) / 2.0;

    icosphere_add_vertex(ico, -1,  t,   0);
    icosphere_add_vertex(ico,  1,  t,   0);
    icosphere_add_vertex(ico, -1, -t,   0);
    icosphere_add_vertex(ico,  1, -t,   0);

    icosphere_add_vertex(ico, 0, -1,   t);
    icosphere_add_vertex(ico, 0,  1,   t);
    icosphere_add_vertex(ico, 0, -1,  -t);
    icosphere_add_vertex(ico, 0,  1,  -t);

    icosphere_add_vertex(ico,  t,  0, -1);
    icosphere_add_vertex(ico,  t,  0,  1);
    icosphere_add_vertex(ico, -t,  0, -1);
    icosphere_add_vertex(ico, -t,  0,  1);

    // 5 faces around point 0
    icosphere_add_face(ico, 0, 0, 11, 5);
    icosphere_add_face(ico, 0, 0, 5, 1);
    icosphere_add_face(ico, 0, 0, 1, 7);
    icosphere_add_face(ico, 0, 0, 7, 10);
    icosphere_add_face(ico, 0, 0, 10, 11);
    // 5 adjacent faces
    icosphere_add_face(ico, 0, 1, 5, 9);
    icosphere_add_face(ico, 0, 5, 11, 4);
    icosphere_add_face(ico, 0, 11, 10, 2);
    icosphere_add_face(ico, 0, 10, 7, 6);
    icosphere_add_face(ico, 0, 7, 1, 8);
    // 5 faces around point 3
    icosphere_add_face(ico, 0, 3, 9, 4);
    icosphere_add_face(ico, 0, 3, 4, 2);
    icosphere_add_face(ico, 0, 3, 2, 6);
    icosphere_add_face(ico, 0, 3, 6, 8);
    icosphere_add_face(ico, 0, 3, 8, 9);
    // 5 adjacent faces
    icosphere_add_face(ico, 0, 4, 9, 5);
    icosphere_add_face(ico, 0, 2, 4, 11);
    icosphere_add_face(ico, 0, 6, 2, 10);
    icosphere_add_face(ico, 0, 8, 6, 7);
    icosphere_add_face(ico, 0, 9, 8, 1);
}
static t_icosphere*
icosphere_build(int nrec, gmx_bool bVerbose)
{
    t_icosphere *ico;
    int a, b, c, f, irec, nfacenow;

    ico = icosphere_init(nrec);

    snew(ico->rec_face,  nrec+1);
    snew(ico->rec_nface, nrec+1);

    icosphere_buildIcosahedron(ico);

    for (irec = 1; irec<=nrec; irec++)
    {
        /* backup current faces and delete ico->face */
        nfacenow = round(20 * pow(4.,irec-1));

        for (f = 0; f<nfacenow; f++)
        {
            a = icosphere_getMiddlePoint(ico, ico->rec_face[irec-1][f].v[0],  ico->rec_face[irec-1][f].v[1]);
            b = icosphere_getMiddlePoint(ico, ico->rec_face[irec-1][f].v[1],  ico->rec_face[irec-1][f].v[2]);
            c = icosphere_getMiddlePoint(ico, ico->rec_face[irec-1][f].v[2],  ico->rec_face[irec-1][f].v[0]);

            /* Create 4 faces in this recusion level (irec) from the previous recursion level (irec-1) */
            icosphere_add_face(ico, irec, ico->rec_face[irec-1][f].v[0], a, c);
            icosphere_add_face(ico, irec, ico->rec_face[irec-1][f].v[1], a, b);
            icosphere_add_face(ico, irec, ico->rec_face[irec-1][f].v[2], b, c);
            icosphere_add_face(ico, irec, a, b, c);
        }
        /* printf("After recusion %d, have %d faces\n", irec, ico->rec_nface[irec]); */
    }

    /* Store the lowest recusion level */
    ico->face  = ico->rec_face[nrec];
    ico->nface = ico->rec_nface[nrec];

    if (bVerbose)
    {
        printf("\nCreated icosphere with %d vertices and %d faces (%d recursions)\n\n", ico->nvertex, ico->nface, nrec);
    }
    return ico;
}

static int
icosphere_isInFace(t_icosphere* ico, t_icosphere_face *face, const dvec x, double tol, double *deviation)
{
    int i;
    double x_dot_n, a_dot_n, normx2, normx;
    gmx_bool bInside;
    dvec xcross;
    double invDenom, dot00, dot11, dot22, dot01, dot02, dot12, u, v;
    dvec v0, v1, v2;

    normx2 = dnorm2(x);
    if (normx2 == 0.)
    {
        /* The origin is in any face */
        DEBUG_PRINTF(("isInFace: accepted because |x| = 0\n"));
        return 1;
    }

    if (face->sprodLim < 0.2)
    {
        gmx_fatal(FARGS,"Invalid sprodlim %g\n",  face->sprodLim);
    }

    normx   = sqrt(normx2);
    x_dot_n = diprod(x, face->normal);
    if (x_dot_n/normx < face->sprodLim-tol)
    {
        /*The scalar product is too small, meaning that the angle between x and the
          normal is too large -> x is outside */
        DEBUG_PRINTF(("isInFace: rejected due to scalar product = %+22.15e\n", x_dot_n));
        return -1;
    }
    /* Now we can be sure that x is at least quite close to the triangle. */

    /* get xcross where the line along x crosses the triangle */
    a_dot_n = diprod(face->normal, ico->vertex[face->v[0]]);
    dsvmul(a_dot_n/x_dot_n, x, xcross);

    /* Now check if xcross is within the triangle. For for algorithm, see e.g.:
       http://www.blackpawn.com/texts/pointinpoly/
    */
    dvec_sub(ico->vertex[face->v[2]], ico->vertex[face->v[0]], v0);
    dvec_sub(ico->vertex[face->v[1]], ico->vertex[face->v[0]], v1);
    dvec_sub(xcross,                  ico->vertex[face->v[0]], v2);

    dot00 = diprod(v0, v0);
    dot01 = diprod(v0, v1);
    dot02 = diprod(v0, v2);
    dot11 = diprod(v1, v1);
    dot12 = diprod(v1, v2);

    // Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    /* First do exact check */
    bInside = ((u >= 0) && (v >= 0) && (u + v < 1));
    if (bInside)
    {
        DEBUG_PRINTF(("isInFace: accepted\n"));
        return 1;
    }

    /* Now do approximate check */
    bInside = ((u >= -tol) && (v >= -tol) && (u + v < (1+tol)));
    if (bInside)
    {
        /*fprintf(stderr,"isInFace: approximately accepted. v = %+22.15e, u = %+22.15e, u+v = %+22.15e\n", u, v, u+v );*/
        DEBUG_PRINTF(("isInFace: approximately accepted\n"));
        /* Get the largest deviation to area inside */
        if (deviation)
        {
            *deviation = ( (-u) > (-v) ) ? (-u) : (-v);
            *deviation = ( (u+v-1) > *deviation) ? (u+v-1) : *deviation;
        }
        return 2;
    }

    /* fprintf(stderr,"isInFace: rejected. v = %+22.15e, u = %+22.15e, u+v = %+22.15e\n", u, v, u+v); */
    DEBUG_PRINTF(("isInFace: rejected due to barycentric coordinates v = %+22.15e, u = %+22.15e, u+v = %+22.15e\n", u, v, u+v));
    return -2;
}

static void
icosphere_thisFace_to_CGO(FILE *fp, t_icosphere *ico, t_icosphere_face *face, double rcyl, double r, double g, double b, double fact)
{
    int i;
    dvec x1, x2;

    fact *= 10;
    for (i=0; i<3; i++)
    {
        copy_dvec(ico->vertex[face->v[ i     ]], x1);
        copy_dvec(ico->vertex[face->v[(i+1)%3]], x2);
        /* triangle on face */
        fprintf(fp, "CYLINDER, %+22.15e,%+22.15e,%+22.15e, %+22.15e,%+22.15e,%+22.15e, %g, %g,%g,%g,%g,%g,%g,\n",
                fact*x1[XX], fact*x1[YY], fact*x1[ZZ],
                fact*x2[XX], fact*x2[YY], fact*x2[ZZ],
                rcyl,
                r, g, b, r, g, b);
        /* origin to vertices */
        fprintf(fp, "CYLINDER, %+22.15e,%+22.15e,%+22.15e, %+22.15e,%+22.15e,%+22.15e, %g, %g,%g,%g,%g,%g,%g,\n",
                fact*x1[XX], fact*x1[YY], fact*x1[ZZ],
                0., 0., 0.,
                rcyl,
                r, g, b, 0., .6, 0.);
    }
    /* the normal vector */
    copy_dvec(face->normal, x1);
    fprintf(fp, "CYLINDER, %+22.15e,%+22.15e,%+22.15e, %+22.15e,%+22.15e,%+22.15e, %g, %g,%g,%g,%g,%g,%g,\n",
            fact*x1[XX], fact*x1[YY], fact*x1[ZZ],
            0., 0., 0.,
            rcyl,
            r, g, b, 0., .6, 0.);

}

static int
icosphere_x2faceID(t_icosphere *ico, const dvec xgiven)
{
    int irec, f = -1, f0, fapprox = -1, res[20], ntest;
    t_icosphere_face *facelist;
    gmx_bool bFound, bFoundApprox;
    dvec x;
    double smallestDeviation, devtmp;

    /* First normlize x */
    dsvmul(1./sqrt(dnorm2(xgiven)), xgiven, x);

    /* Now go throught the recusions levels */
    f = 0;
    for (irec = 0; irec <= ico->nrec; irec++)
    {
        facelist = ico->rec_face[irec];

        if (irec == 0)
        {
            /* In the zeroth recusion level (Ikosahedron) we test all 20 faces */
            ntest = 20;
        }
        else
        {
            /* In all lower recuions levels we need to test only 4 faces */
            ntest = 4;
        }


        f0                    = f*4;
        bFound = bFoundApprox = FALSE;
        smallestDeviation     = 1e20;
        DEBUG_PRINTF(("x2faceID recusion: irec = %d, ntest = %d -- f0 = %d\n", irec, ntest, f0));
        for (f = f0; f < (f0+ntest); f++)
        {
            res[f-f0] = icosphere_isInFace(ico, facelist+f, x, 1e-5, &devtmp);

            if (res[f-f0] == 1)
            {
                DEBUG_PRINTF(("\tbreaking loop at f = %d (f0 = %d)\n", f, f0));
                bFound = TRUE;
                break;
            }
            else if (res[f-f0] == 2)
            {
                if (devtmp < smallestDeviation)
                {
                    fapprox           = f;
                    smallestDeviation = devtmp;
                    bFoundApprox      = TRUE;
                }
                DEBUG_PRINTF(("\tNow fapprox = %d (smallest deviation = %g)\n", fapprox, smallestDeviation));
            }
        }

        if (!bFound && !bFoundApprox)
        {
            printf("Finally fapprox = %d\n", fapprox);

            FILE *fp;
            int k;
            fp = ffopen("error.py", "w");
            fprintf(fp, "from pymol.cgo import *\nfrom pymol import cmd\n");

            fprintf(fp, "obj=[]\n");
            fprintf(fp, "obj.extend([\n");

            fprintf(fp, "CYLINDER, %+22.15e,%+22.15e,%+22.15e %+22.15e,%+22.15e,%+22.15e %g, %g,%g,%g,%g,%g,%g,\n",
                    10*x[XX], 10*x[YY], 10*x[ZZ],
                    0., 0., 0.,
                    0.1,
                    .1, 1., .2,.1, 1., .1);

            for (k = f0; k < (f0+4); k++)
                icosphere_thisFace_to_CGO(fp, ico, facelist+k, k==(f0+3)?0.1:0.05, 1.-0.3*(k-f0), .1, .3*(k-f0), 1.);

            fprintf(fp, "])\n");
            /* IMPORTANT: CYLINDER statements should NOT be within a BEGIN/END block !!! */
            /* fprintf(fp, "obj.extend([END])\n\n"); */
            fprintf(fp, "cmd.load_cgo(obj,'ico_error')\n");
            ffclose(fp);

            /* Advanced debugging... */
            fprintf(stderr,"xgiven: %+22.15e %+22.15e %+22.15e \n", xgiven[XX], xgiven[YY], xgiven[ZZ] );

            gmx_fatal(FARGS, "Could not find a face in which is x = %+22.15e %+22.15e %+22.15e in recursion level %d\n"
                      "Rejection keys were %d %d %d %d",
                      x[XX], x[YY], x[ZZ], irec, res[0], res[1], res[2], res[3]);
        }
        else if (!bFound && bFoundApprox)
        {
            DEBUG_PRINTF(("Did not find a numerically exact face in rec lvl %d, but %d matches approximately\n",
                          irec, fapprox));
            /* The point was not exacly in one triangle, but nearly in (some) trianle(s) */
            f = fapprox;
        }

        DEBUG_PRINTF(("\tFound  f = %d\n", f));

        if (f<0 || f>=ico->nface)
        {
            gmx_fatal(FARGS, "Found invalid face ID %d in recursion %d\n", f, irec);
        }
    }

    return f;
}

gmx_envelope_t gmx_envelope_init(int nrec, gmx_bool bVerbose)
{
    int j, l, nlong, i, d, ndiv, k, nrays, f;
    double phi, theta, tmp, dtheta, len, phi0, phi1;
    dvec r1, r0, diff, rnew, rtmp, *vertex, t;

    gmx_envelope_t e;
    snew(e, 1);

    e->bVerbose = bVerbose;
    e->d = -1;
    e->bHaveEnv = FALSE;
    e->bOriginInside = FALSE;
    e->nDefined = 0;
    e->minInner = -1;
    e->maxInner = -1;
    e->minOuter = -1;
    e->maxOuter = -1;
    e->vol      = -1;
    e->ftunit_re    = NULL;
    e->ftunit_im    = NULL;
    e->bHaveFourierTrans = FALSE;
    e->ftdens_re = NULL;
    e->ftdens_im = NULL;
    e->bHaveSolventFT = FALSE;
    e->ngrid = 100;
    e->solventNelec = NULL;
    e->solventNelecTotal = -1;
    e->nSolventStep = 0;
    e->surfElemOuter = e->surfElemInner = NULL;
    clear_dvec(e->bsphereCent);
    e->bsphereR2 = -1;

    /* Make an icosohere mesh */
    e->nrec = nrec;
    e->ico  = icosphere_build(nrec, e->bVerbose);

    nrays = e->ico->nvertex;
    e->nSurfElems = e->ico->nface;
    e->nrays = nrays;
    snew(e->s,     nrays);
    snew(e->inner, nrays);
    for (j=0; j<nrays; j++)
    {
        e->inner[j] = 1e20;
    }
    snew(e->outer, nrays);
    snew(e->r,     nrays);
    snew(e->isDefined, nrays);

    for (j=0; j<nrays; j++)
    {
        copy_dvec(e->ico->vertex[j], e->r[j]);
        e->s[j].theta = acos(e->r[j][ZZ]);
        e->s[j].phi   = atan2(e->r[j][YY], e->r[j][XX]);
    }

    /* Init solvent density array */
    e->nSolventStep = 0;
    snew(e->solventNelec, e->nSurfElems);
    for (f = 0; f<e->nSurfElems; f++)
    {
        snew(e->solventNelec[f], e->ngrid);
    }
    e->solventNelecNorm = 0.;

    e->grid_density = NULL;

    return e;
}

void gmx_envelope_solvent_density_destroy(gmx_envelope_t e)
{
    int f;
    for (f = 0; f<e->nSurfElems; f++)
    {
        sfree(e->solventNelec[f]);
        e->solventNelec[f]=NULL;
    }
    sfree(e->solventNelec); e->solventNelec=NULL;
}

void gmx_envelope_ftunit_srenew(gmx_envelope_t e, int nq)
{
    srenew(e->ftunit_re, nq);
    srenew(e->ftunit_im, nq);
    /*if(!e->ftunit_re) snew(e->ftunit_re, nq);
      if(!e->ftunit_im) snew(e->ftunit_im, nq);*/
}

void gmx_envelope_ftdens_srenew(gmx_envelope_t e, int nq)
{
    srenew(e->ftdens_re, nq);
    srenew(e->ftdens_im, nq);
    /*if(!e->ftdens_re) snew(e->ftdens_re, nq);
      if(!e->ftdens_im) snew(e->ftdens_im, nq);*/
}

void gmx_envelope_ftunit_sfree(gmx_envelope_t e)
{
    sfree(e->ftunit_re);
    sfree(e->ftunit_im);
    /*sfree(e->ftunit_re); e->ftunit_re=NULL;
      sfree(e->ftunit_im); e->ftunit_re=NULL;*/
    e->bHaveFourierTrans = FALSE;
}

void gmx_envelope_ftdens_sfree(gmx_envelope_t e)
{
    sfree(e->ftdens_re);
    sfree(e->ftdens_im);
    /*sfree(e->ftdens_re); e->ftunit_re=NULL;
      sfree(e->ftdens_im); e->ftunit_re=NULL;*/
    e->bHaveSolventFT = FALSE;
}

void gmx_envelope_destroy(gmx_envelope_t e)
{

    //icosphere_destroy(&(e->ico));
    gmx_envelope_solvent_density_destroy(e);
    sfree(e->s);
    sfree(e->inner);
    sfree(e->outer);
    sfree(e->r);
    sfree(e->isDefined);
    gmx_envelope_ftunit_sfree(e);
    gmx_envelope_ftdens_sfree(e);

    sfree(e);
    e=NULL;
}

static void
gmx_envlope_triangle_to_CGO(gmx_envelope_t e, FILE *fp, int f, double rcyl, double r, double g, double b)
{
    int i, j, j1;
    dvec x, x1, y, y1;

    for (i=0; i<3; i++)
    {
        j   = e->surfElemOuter[f].ind[i];
        j1  = e->surfElemOuter[f].ind[(i+1)%3];

        dsvmul(e->outer[j ], e->r[j ], x );
        dsvmul(e->outer[j1], e->r[j1], x1);
        printf("\ne->bOriginInside = %d\n", e->bOriginInside);
        if (e->bOriginInside)
        {
            clear_dvec(y);
            clear_dvec(y1);
        }
        else
        {
            dsvmul(e->inner[j ], e->r[j ], y);
            dsvmul(e->inner[j1], e->r[j1], y1);
            printf("\ninner %d = %g  outer %g\n", j, e->inner[j], e->outer[j]);
        }
        printf("\n x = %g %g %g\n", x[XX], x[YY], x[ZZ]);
        printf("\n y = %g %g %g\n", y[XX], y[YY], y[ZZ]);

        fprintf(fp, "CYLINDER, %8g,%8g,%8g, %8g,%8g,%8g, %8g, %8g,%8g,%8g,%8g,%8g,%8g,\n",
                10*x[XX], 10*x[YY], 10*x[ZZ],
                10*y[XX], 10*y[YY], 10*y[ZZ],
                rcyl,
                r, g, b, r, g, b);
        fprintf(fp, "CYLINDER, %8g,%8g,%8g, %8g,%8g,%8g, %8g, %8g,%8g,%8g,%8g,%8g,%8g,\n",
                10*x [XX], 10*x [YY], 10*x [ZZ],
                10*x1[XX], 10*x1[YY], 10*x1[ZZ],
                rcyl,
                r, g, b, r, g, b);
        if (!e->bOriginInside)
        {
            fprintf(fp, "CYLINDER, %8g,%8g,%8g, %8g,%8g,%8g, %8g, %8g,%8g,%8g,%8g,%8g,%8g,\n",
                    10*y [XX], 10*y [YY], 10*y [ZZ],
                    10*y1[XX], 10*y1[YY], 10*y1[ZZ],
                    rcyl,
                    r, g, b, r, g, b);
        }
    }
}


void gmx_envelope_dump2pdb(gmx_envelope_t e, int nHighlight, int *high, gmx_bool bExtra, rvec x_extra,
                           double R, const char *fn)
{
    int i, j, res = 1;
    FILE *fp;
    gmx_bool bHigh;

    fp = fopen(fn, "w");

    for (j=0; j<e->nrays; j++)
    {
        bHigh = FALSE;
        for (i = 0; i<nHighlight; i++)
        {
            bHigh |= (j == high[i]);
        }

        if (!bHigh)
        {
            fprintf(fp, "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n",
                    j+1, "C", "MOL", "", res, R*e->r[j][XX], R*e->r[j][YY], R*e->r[j][ZZ], 1.0, 0.0);
        }
        else
        {
            res++;
            fprintf(fp, "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n",
                    j+1, "O", "HIG", "", res, R*e->r[j][XX], R*e->r[j][YY], R*e->r[j][ZZ], 1.0, 0.0);
            res++;
        }
    }
    if (bExtra)
    {
        res++;
        fprintf(fp, "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n",
                j+1, "FE", "FE", "", res, R*x_extra[XX], R*x_extra[YY], R*x_extra[ZZ], 1.0, 0.0);
    }
    fclose(fp);
    printf("\n ** Dumped envelope with error to %s **\n", fn);
}

static void
dvec_to_CGO(FILE *fp, const dvec x, double radius)
{
    fprintf(fp, "SPHERE, %g,%g,%g, %g,\n", 10*x[XX], 10*x[YY], 10*x[ZZ], radius);
}
static void
rvec_to_CGO(FILE *fp, const rvec x, double radius)
{
    fprintf(fp, "SPHERE, %g,%g,%g, %g,\n", 10*x[XX], 10*x[YY], 10*x[ZZ], radius);
}


static void
gmx_envlope_dump_triangle_x_to_CGO_file(gmx_envelope_t e, int f, const dvec x, const char *fn)
{
    FILE *fp;

    fp = fopen(fn, "w");
    fprintf(fp, "from pymol.cgo import *\nfrom pymol import cmd\n\n");
    fprintf(fp, "obj = [\n\nCOLOR, %g, %g, %g,\n\n", 0.8, 0.8, 0.8);
    gmx_envlope_triangle_to_CGO(e, fp, f, 0.05, 1.0, 0., 0.);
    dvec_to_CGO(fp, x, 0.1);
    fprintf(fp, "]\n\ncmd.load_cgo(obj, 'error')\n\n");
    printf("\n\nDumped surface element %d and x = %g %g %g to CGO file %s\n\n", f,
           x[0], x[1], x[2], fn);
    fclose(fp);
}

/* Given an arbitrary direction vec{x}, return where a*vec{x} crosses inner and outer surface */
static void
gmx_envelope_x2OuterInner(gmx_envelope_t e, const rvec xin, real *innerReturn, real *outerReturn)
{
    double x_dot_n, tol = 1e-2;
    dvec xcross, x;
    int f;

    copy_rdvec(xin, x);
    f = icosphere_x2faceID(e->ico, x);
    if (!  e->surfElemOuter[f].bDefined)
    {
        FILE *fp = fopen("env_error.py", "w");
        fprintf(fp, "from pymol.cgo import *\nfrom pymol import cmd\n\n");
        fprintf(fp, "obj = [\n\nCOLOR, %g, %g, %g,\n\n", 0.8, 0.8, 0.8);
        icosphere_thisFace_to_CGO(fp, e->ico, e->ico->face+f, 0.1, 1., 0.3, 0.3, 10.);
        dvec_to_CGO(fp, x, 0.5);
        fprintf(fp, "]\n\ncmd.load_cgo(obj, 'error')\n\n");
        fclose(fp);
        fprintf(stderr, "\n\nDumped icopshere face and atom to cgo file env_error.py\n\n");

        gmx_fatal(FARGS, "Trying to get distance of x = %g / %g / %g from the surface, but the surface element %d"
                  " is not defined\n", x[XX], x[YY], x[ZZ], f);
    }

    /* get xcross where the line along x crosses the surfElemOuter */
    x_dot_n = diprod(x, e->surfElemOuter[f].normal);
    dsvmul(e->surfElemOuter[f].c/x_dot_n, x, xcross);
    *outerReturn = sqrt(dnorm2(xcross));

    if ((*outerReturn > (e->outer[e->surfElemOuter[f].ind[0]] + tol)) &&
        (*outerReturn > (e->outer[e->surfElemOuter[f].ind[1]] + tol)) &&
        (*outerReturn > (e->outer[e->surfElemOuter[f].ind[2]] + tol)) )
    {
        gmx_fatal(FARGS, "Inconsistency error in gmx_envelope_x2OuterInner().\n");
    }

    if (e->bOriginInside)
    {
        *innerReturn = 0.;
    }
    else
    {
        x_dot_n = diprod(x, e->surfElemInner[f].normal);
        dsvmul(e->surfElemInner[f].c/x_dot_n, x, xcross);
        *innerReturn = sqrt(dnorm2(xcross));
    }
}

static gmx_bool
gmx_envelope_withinInnerOuter_thisSurfElem(gmx_envelope_t e, int f, const dvec x, double tol)
{
    dvec xmc;
    gmx_bool bBelowOuter, bAboveInner = TRUE;

    if (! e->surfElemOuter[f].bDefined)
    {
        return FALSE;
    }

    dvec_sub(x, e->surfElemOuter[f].center, xmc);
    bBelowOuter = (diprod(e->surfElemOuter[f].normal, xmc) <= tol);

    if (! e->bOriginInside)
    {
        dvec_sub(x, e->surfElemInner[f].center, xmc);
        bAboveInner = (diprod(e->surfElemInner[f].normal, xmc) >= -tol);
    }
    return (bBelowOuter && bAboveInner);
}

gmx_bool
gmx_envelope_isInside(gmx_envelope_t e, const rvec xin)
{
    real normx2;
    int f;
    dvec x;

    normx2 = norm2(xin);
    if (normx2 > dsqr(e->maxOuter))
    {
        return FALSE;
    }
    /* We cannot do the same with the e->minInner, since the surface trangle may be close to the origin
       than e->minInner */

    copy_rdvec(xin, x);
    f = icosphere_x2faceID(e->ico, x);
    return gmx_envelope_withinInnerOuter_thisSurfElem(e, f, x, 0.);
}

void
gmx_envelope_distToOuterInner(gmx_envelope_t e, const rvec x, real *distInner, real *distOuter)
{
    real inner, outer, normx;

    normx = sqrt(norm2(x));
    gmx_envelope_x2OuterInner(e, x, &inner, &outer);

    if (inner > 1e-5)
    {
        *distInner = normx - inner;
    }
    else
    {
        *distInner = 1e20;
    }
    *distOuter = outer - normx;
}

/* Return distance and index of the atom, which is closest to the envelope surface
   bMinDistToOuter indicates if the closest distance was found to an outer surface or to an inner surface */
void
gmx_envelope_minimumDistance(gmx_envelope_t e, const rvec x[], atom_id *index,
                             int isize, real *mindist, int *imin, gmx_bool *bMinDistToOuter)
{
    int i;
    real distInner, distOuter;

    *imin            = -1;
    *mindist         = 1e20;

    /* loop over atoms */
    for (i=0; i<isize; i++)
    {
        gmx_envelope_distToOuterInner(e, x[index[i]], &distInner, &distOuter);

        if (distOuter < -1e-5)
        {
            if (gmx_envelope_isInside(e, x[index[i]]))
            {
                gmx_fatal(FARGS, "Got distOuter = %g, but x is inside\n", distOuter);
            }
        }

        if (distInner < *mindist)
        {
            *mindist = distInner;
            *imin = i;
            *bMinDistToOuter = FALSE;
        }
        if (distOuter < *mindist)
        {
            *mindist = distOuter;
            *imin = i;
            *bMinDistToOuter = TRUE;
        }
    }
}

gmx_bool
gmx_envelope_bInsideCompactBox(gmx_envelope_t e, matrix Rinv, matrix box, rvec boxToXRef, t_pbc *pbc, gmx_bool bVerbose,
                               real tolerance)
{
    rvec boxcenter, vertex, dxpbc, dxdirect, rRay, rRayRot, boxToRay;
    dvec dRay;
    int d, j, nOutside = 0;
    real small = box[XX][XX]/10;
    FILE *fp = NULL;

    calc_box_center(ecenterTRIC, box, boxcenter);

    for (j=0; j<e->nrays; j++)
    {
        if (e->isDefined[j])
        {
            /* Get vertex relative to boxcenter */
            dsvmul(e->outer[j] + tolerance, e->r[j], dRay);
            copy_drvec(dRay,rRay);
            mvmul(Rinv, rRay, rRayRot);
            rvec_add(rRayRot, boxToXRef, boxToRay);

            /* Now check if the ray is inside the compact box */
            pbc_dx(pbc, boxToRay, boxcenter, dxpbc);
            rvec_sub(   boxToRay, boxcenter, dxdirect);
            if (fabs(dxpbc[XX] - dxdirect[XX]) > small ||
                fabs(dxpbc[YY] - dxdirect[YY]) > small ||
                fabs(dxpbc[ZZ] - dxdirect[ZZ]) > small )
            {
                if (bVerbose)
                {
                    if (fp == NULL)
                    {
                        fp = fopen("env_vertex_outside.py", "w");
                        fprintf(fp, "from pymol.cgo import *\nfrom pymol import cmd\n\n");
                        fprintf(fp, "obj = [\n\nCOLOR, %g, %g, %g,\n\n", 1., 0.0, 0.0);
                    }
                    dvec_to_CGO(fp, dRay, 0.4);
                    nOutside++;
                    printf("Wrote ray %d, r = %g R = %g %g %g\n", j, e->outer[j], e->r[j][0], e->r[j][1], e->r[j][2]);
                    printf("\t rRayRot   = %g %g %g\n", rRayRot[0],  rRayRot[1],  rRayRot[2]);
                    printf("\t boxToXRef = %g %g %g\n", boxToXRef[0],  boxToXRef[1],  boxToXRef[2]);
                    printf("\t boxToRay  = %g %g %g\n", boxToRay[0],  boxToRay[1],  boxToRay[2]);
                }
                else
                {
                    return FALSE;
                }
            }
        }
    }

    if (nOutside)
    {
        fprintf(fp, "]\n\ncmd.load_cgo(obj, 'vertex_outside')\n\n");
        fclose(fp);
        printf("Found %d vertices outside of the compact unitcell. Wrote Pymol cgo file env_vertex_outside.py\n\n",
               nOutside);
        return FALSE;
    }

    return TRUE;
}

void
gmx_envelope_bounding_sphere(gmx_envelope_t e, rvec cent, real *R2)
{
    if (!e->bHaveEnv)
    {
        gmx_fatal(FARGS, "Requested bounding sphere of the envelope, but the envelope is not yet constructed\n");
    }
    copy_drvec(e->bsphereCent, cent);
    *R2 = e->bsphereR2;
}
double
gmx_envelope_diameter(gmx_envelope_t e)
{
    if (!e->bHaveEnv)
    {
        gmx_fatal(FARGS, "Requested diameter of the envelope, but the envelope is not yet constructed\n");
    }
    return 2*sqrt(e->bsphereR2);
}

void
gmx_envelope_center_xyz(gmx_envelope_t e, matrix Rinv, rvec cent)
{
    int j, d;
    rvec rRay, rRayRot, min = {1e20, 1e20, 1e20}, max = {-1e20, -1e20, -1e20};
    dvec dRay;

    for (j=0; j<e->nrays; j++)
    {
        if (e->isDefined[j])
        {
            /* Get vertex relative to boxcenter */
            dsvmul(e->outer[j], e->r[j], dRay);
            copy_drvec(dRay, rRay);
            mvmul(Rinv, rRay, rRayRot);

            for (d=0; d<DIM; d++)
            {
                min[d] = (rRayRot[d] < min[d]) ? rRayRot[d] : min[d];
                max[d] = (rRayRot[d] > max[d]) ? rRayRot[d] : max[d];
            }
        }
    }
    for (d=0; d<DIM; d++)
    {
        cent[d] = (min[d] + max[d])/2;
    }
}


static void
gmx_envelope_build_bounding_sphere(gmx_envelope_t e)
{
    /* NT is the floating point variable in gmx_miniball.c. It may be either real or double */
    NT *centNT, **xenvNT, subopt, NTnull[DIM], relerr;
    int i,j,d,ndef;
    gmx_miniball_t mb;
    FILE *fp;
    dvec xenv;


    snew(xenvNT, e->nrays);

    for(j = 0; j < e->nrays; j++) {
        snew(xenvNT[j], DIM);
    }

    /* Need a linear NT* array with pointers to the envelope vertices */
    ndef = 0;
    for (j = 0; j<e->nrays; j++)
    {
        if (e->isDefined[j])
        {
            /* Get vertex vector from envelope reference point */
            dsvmul(e->outer[j], e->r[j], xenv);
            for (d = 0; d<DIM; d++)
            {
                xenvNT[ndef][d] = xenv[d];
            }
            ndef++;
        }
    }

    mb = gmx_miniball_init(DIM, xenvNT, ndef);

    if (!gmx_miniball_is_valid(mb, -1))
    {
        /* In very rare cases, Miniball does even in double precision not fully converge. Check for approximate convergence. */
        relerr = gmx_miniball_relative_error (mb, &subopt);
        fprintf(stderr,
                "\nMiniball:  Relative error = %g"
                "\n           Suboptimality  = %g"
                "\n           Radius         = %g\n",
                relerr, subopt, sqrt(gmx_miniball_squared_radius(mb)));

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
            for (i=0; i<e->nrays; i++)
            {
                fprintf(fp, "%.15f %.15f %.15f\n", xenvNT[i][XX], xenvNT[i][YY], xenvNT[i][ZZ]);
            }
            fclose(fp);
            fprintf(stderr, "Dumped %d coordinates that made an invalid Miniball to miniball_coords_error.dat\n", e->nrays);
            fprintf(stderr, "Relative error = %g, suboptimality = %g\n", gmx_miniball_relative_error (mb, &subopt),
                    subopt);

            gmx_fatal(FARGS, "Generating the bounding sphere with Miniball failed.\n");
        }
    }
    centNT = gmx_miniball_center(mb);
    for (d=0; d<DIM; d++)
    {
        e->bsphereCent[d] = centNT[d];
    }
    e->bsphereR2 = gmx_miniball_squared_radius(mb);


    for(j = 0; j < e->nrays; j++) {
        sfree(xenvNT[j]);
    }
    sfree(xenvNT);
    gmx_miniball_destroy(&mb);
}


#define GMX_ENVELOPE_SMOOTH_ALLOW_DECREASE_DIST 1
static void
gmx_envelope_smoothEnvelope(gmx_envelope_t e, double sigma)
{
    int i, j, k;
    double cos_2sigma, sigma2, *outer, *inner, osum, isum, wsumo, wsumi, w = 0., phi2, cosphi;

    if (e->bVerbose)
    {
        printf("Smoothing envelope with sigma = %g degree.\n", sigma*180./M_PI);
    }

    cos_2sigma = cos(2*sigma);
    sigma2     = sqr(sigma);

    snew(outer, e->nrays);
    snew(inner, e->nrays);

    for (j=0; j<e->nrays; j++)
    {
        wsumo = wsumi = osum = isum = 0.;

        if (e->outer[j] == 0.0)
        {
            outer[j] = e->outer[j];
            inner[j] = e->inner[j];
        }
        else
        {
            for (k=0; k<e->nrays; k++)
            {
                cosphi = diprod(e->r[j], e->r[k]);

                /* cut-off at ~2sigma */
                if (cosphi > cos_2sigma && e->outer[k] != 0.0)
                {
                    /* We smooth such that the outer surface is only moved outwards, while
                       the inner surface is only moved inwards */
                    if (e->outer[k] >= e->outer[j] || e->inner[k] <= e->inner[j])
                    {
                        /* approximate phi^2 by Taylor series, OK because sigma is small. */
                        phi2 = 2*(1-cosphi);
                        w    = exp(-phi2/(2*sigma2));
                    }
                    if (e->outer[k] >= e->outer[j])
                    {
                        wsumo += w;
                        osum  += w*e->outer[k];
                    }
                    if (e->inner[k] <= e->inner[j])
                    {
                        wsumi += w;
                        isum  += w*e->inner[k];
                    }
                }
            }
            outer[j] = osum/wsumo;
            inner[j] = isum/wsumi;
        }
    }

    /* Store the smoothed outer/inner */
    for (j=0; j<e->nrays; j++)
    {
        e->outer[j] = outer[j];
        e->inner[j] = inner[j];
    }

    sfree(outer);
    sfree(inner);
}

static void
gmx_envelope_build_md5(gmx_envelope_t e)
{
    md5_state_t state;
    char buf[10], buf1[20], buf2[20];
    // unsigned char *ptr;
    int j;

    md5_init(&state);
    for (j=0; j<e->nrays; j++)
    {
        sprintf(buf1, "%.3g", e->inner[j]);
        buf1[4] = '\0';
        sprintf(buf2, "%.3g", e->outer[j]);
        buf2[4] = '\0';
        sprintf(buf, "%s%s", buf1, buf2);
        md5_append(&state, (md5_byte_t*) buf, 8);
    }
    md5_finish(&state, e->chksum);

    /* make a null-terminated string to be used in printf and strcmp */
    strncpy(e->chksum_str, (char*) e->chksum, 16);
    e->chksum_str[16] = '\0';
}

static void
gmx_envelope_set_surfaceElements(gmx_envelope_t e, t_triangle *surfElem, double *R)
{
    int i, f, ind[3];
    dvec x[3], vec1, vec2, tmp;
    double length, mult, sum_iprods, tanOmegaHalf, det1, rmax, rmin, thisr;
    dvec m[DIM];

    // snew(e->surfElemOuter, e->ico->nface);
    for (f = 0; f<e->ico->nface; f++)
    {
        /* store corners and whether defined or not */
        copy_ivec(e->ico->face[f].v, surfElem[f].ind);
        copy_ivec(e->ico->face[f].v, ind);
        surfElem[f].iface = f;
        surfElem[f].bDefined = ( e->isDefined[ind[0]] && e->isDefined[ind[1]] && e->isDefined[ind[2]]);

        if (! surfElem[f].bDefined)
        {
            continue;
        }

        /* corners and get center */
        clear_dvec(tmp);
        for (i = 0; i<3; i++)
        {
            dsvmul(R[ind[i]], e->r[ind[i]], x[i]);
            dvec_inc(tmp, x[i]);
        }
        dsvmul(1./3, tmp, surfElem[f].center);

        /* Get rmax and rmin */
        rmax = 0.;
        rmin = 1e20;
        for (i = 0; i<3; i++)
        {
            /* important: rmin / rmax are not the from the distance of the corners, but instead from the
               distance AFTER projecting onto the normal */
            thisr = diprod(e->ico->face[f].normal, x[i]);
            rmax = (thisr > rmax) ? thisr : rmax;
            rmin = (thisr < rmin) ? thisr : rmin;
        }
        surfElem[f].rmax = rmax;
        surfElem[f].rmin = rmin;

        /* Get (normalized) normal vector and volume of pyramid */
        dvec_sub(x[1], x[0], vec1);
        dvec_sub(x[2], x[0], vec2);
        dcprod(vec1, vec2, tmp);
        length = sqrt(dnorm2(tmp));
        if (diprod(surfElem[f].center, tmp) < 0)
        {
            mult = -1./length;
        }
        else
        {
            mult = 1./length;
        }
        dsvmul(mult, tmp, tmp);
        copy_dvec(tmp, surfElem[f].normal);
        surfElem[f].volPyramid = (length/2)/3 * diprod(surfElem[f].normal, x[0]);

        /* plane constant */
        surfElem[f].c = diprod(surfElem[f].normal, x[0]);

        /* Get the area of the triangle projected to a radius of r = 1 */
        dvec_sub(e->r[ind[1]], e->r[ind[0]], vec1);
        dvec_sub(e->r[ind[2]], e->r[ind[0]], vec2);
        dcprod(vec1, vec2, tmp);
        /* area is 0.5 times the area of the parallelogram spaned by two triangle sides */
        surfElem[f].area_r1 = 0.5 * sqrt(dnorm2(tmp));

        /* Get the space angle using Oosterom-and-Strackee algorithm using |r1| = |r2| = |r3| = 1 */
        sum_iprods = 0.;
        for (i = 0; i<3; i++)
        {
            copy_dvec(e->r[ind[i]], m[i]);
            sum_iprods += diprod(e->r[ind[i]], e->r[ind[(i+1)%3]]);
        }
        det1 = ddet(m);
        tanOmegaHalf = (det1 < 0 ? -1 : 1 ) * det1 / (1 + sum_iprods);
        surfElem[f].spaceangle = 2 * atan(tanOmegaHalf);

        /*printf("Triangle %3d: center %g %g %g, normal %g %g %g, area %g vol %g, rmin/rmax = %g / %g\n",
          f,
          surfElem[f].center[XX],
          surfElem[f].center[YY],
          surfElem[f].center[ZZ],
          surfElem[f].normal[XX],
          surfElem[f].normal[YY],
          surfElem[f].normal[ZZ],
          surfElem[f].area_r1,
          surfElem[f].volPyramid,
          surfElem[f].rmin, surfElem[f].rmax); */
    }
}

/* static int */
/* compareDouble (const void * a, const void * b) */
/* { */
/*   return ( *(double*)a - *(double*)b ); */
/* } */


static double
gmx_envelope_triangle_2_area(dvec a, dvec b, dvec c)
{
    dvec vec1, vec2, normal;
    double A;

    dvec_sub(a, b, vec1);
    dvec_sub(a, c, vec2);
    dcprod(vec1, vec2, normal);
    A = 0.5 * sqrt(dnorm2(normal));
    if (A < 0)
    {
        A = -A;
    }
    return A;
}

static double
gmx_envelope_quadrangle_2_area_old(dvec a[])
{
    double area1, area2, area3;
    int i, iopp = -1, i1, i2, itol;
    dvec v1, v2, c1, c2;
    double tol[] = {0., 1e-7, 1e-6};

    area1 = gmx_envelope_triangle_2_area(a[0], a[1], a[2]);

    for (itol = 0; itol<sizeof(tol)/sizeof(tol[0]); itol++)
    {
        /* Which point of 0,1,2 is opposite of point 3 ? */
        for (i=0; i<3; i++)
        {
            i1 = (i+1)%3;
            i2 = (i+2)%3;

            dvec_sub(a[i1], a[i], v1);
            dvec_sub(a[i2], a[i], v2);
            dcprod(v1, v2, c1);

            dvec_sub(a[i1], a[3], v1);
            dvec_sub(a[i2], a[3], v2);
            dcprod(v1, v2, c2);

            /*printf("\n i = %d %d %d\n", i, i1, i2);
              printf(" i %d, dcprod1 = %g %g %g (area1 = %g)\n", i, c1[0], c1[1], c1[2], area1);
              printf(" i %d, dcprod2 = %g %g %g\n", i, c2[0], c2[1], c2[2]);
              printf(" i %d, iprod = %g\n", i, iprod(c1, c2)); */
            if (diprod(c1, c2) < tol[itol])
            {
                iopp = i;
                break;
            }
        }
        if (iopp != -1)
        {
            break;
        }
    }

    if (iopp == -1)
    {
        gmx_fatal(FARGS, "Could not find the point on the opposite side while compute area of quadrangle\n");
    }

    area2 = gmx_envelope_triangle_2_area(a[(iopp+1)%3], a[(iopp+2)%3], a[3]);

    /* Catch the case that two points are very close to each other. Then, we may sum up two areas
       which are nearly zero. */
    if (area1 < 1e-5 && area2 < 1e-5)
    {
        /* Take the 3rd corner of the previous trianngle 0-1-2 */
        area3 = gmx_envelope_triangle_2_area(a[(iopp)%3], a[(iopp+2)%3], a[3]);
        if (area3 > (area1+area2))
        {
            area2 = area3;
            printf("\nNOTE: Sum of two triangles of the quadrangle are nearly zero - replaced area by %g instead\n\n",
                   area3);
        }
    }

    return area1 + area2;
}

/* We know that the corners were added in clockwise (or counterclockwise) order, but
   a[0] and a[1] are for sure not opposite
   -> can use Area = 0.5 * |q x p|, where q and p are the two diagonal vectors */
static double
gmx_envelope_quadrangle_2_area(dvec a[])
{
    dvec q, p, tmp;

    dvec_sub(a[0], a[2], p);
    dvec_sub(a[1], a[3], q);
    dcprod(p, q, tmp);

    return 0.5*sqrt(dnorm2(tmp));
}


static void
linear_vec_comb2(dvec x0, double a, const dvec avec, double b, const dvec bvec, dvec res)
{
    dvec tmp;

    copy_dvec(x0, res);
    dsvmul(a, avec, tmp);
    dvec_inc(res, tmp);
    dsvmul(b, bvec, tmp);
    dvec_inc(res, tmp);
}

gmx_bool bNewCorner(dvec x, dvec corner[], int ncorn, double tol)
{
    int i;
    dvec v;
    double tol2 = sqr(tol);

    for (i = 0; i < ncorn; i++)
    {
        dvec_sub(x, corner[i], v);
        if (dnorm2(v) <= tol2)
        {
            return FALSE;
        }
    }
    return TRUE;
}

static void
gmx_envelope_binVolumes(gmx_envelope_t e)
{
    int f, i, j, jj, nIntersectInside, jcorn, nInside, ntriangles, k, nVolWarn = 0, ncorners, itol;
    ivec iray;
    dvec xouter[3], xinner[3], xIntersect, v, vtmp, xcentNorm, x[3], *xcomArea, u, du, dv, y;
    /*int  nbelowInner, naboveInner, nbelowOuter, naboveOuter;
      ivec ibelowInner, iaboveInner, ibelowOuter, iaboveOuter;*/
    double *area, area_r, area_tmp, vsum, vtotalsum = 0., relVolDiff, vexpect, a,
        n_dot_v, rmin, rmax, dr, r, router[3], rinner[3];
    double length, xu, xv, asum, tol, areaGrid, rray;
    gmx_bool bIsAboveOuter[3], bIsAboveInner[3], bIntersectInside[3], bInside;
    const int nAreaGrid = 50;
    dvec corner[9];
    double tols[2] = {1e-6, 1e-5};

    if (e->bVerbose && FALSE)
    {
        printf("Generating volume bins of envelope...");
    }
    fflush(stdout);

    snew(e->binVol,  e->nSurfElems);
    snew(e->xBinVol, e->nSurfElems);
    snew(area,       e->nSurfElems+1);
    snew(xcomArea,   e->nSurfElems+1);

    for (f = 0; f<e->nSurfElems; f++)
    {
        if (! e->surfElemOuter[f].bDefined)
        {
            continue;
        }

        if (! e->bOriginInside)
        {
            if (e->surfElemOuter[f].rmin < e->surfElemInner[f].rmax)
            {
                gmx_fatal(FARGS, "Volume of face %d of envelope is too skewed. (rminOuter = %g, rmaxInner = %g)\n",
                          f, e->surfElemOuter[f].rmin, e->surfElemInner[f].rmax);
            }
        }

        snew(e->binVol[f],  e->ngrid);
        snew(e->xBinVol[f], e->ngrid);
        rmin = e->bOriginInside ? 0. : e->surfElemInner[f].rmin;
        rmax = e->surfElemOuter[f].rmax;
        dr   = (rmax-rmin)/e->ngrid;

        for (j = 0; j<3; j++)
        {
            /* Corners of the volume (3 for the outer, and 3 for the inner end) */
            iray[j] = e->surfElemOuter[f].ind[j];
            dsvmul(e->outer[iray[j]], e->r[iray[j]], xouter[j]);
            router[j] = sqrt(dnorm2(xouter[j]));
            if (! e->bOriginInside)
            {
                dsvmul(e->inner[iray[j]], e->r[iray[j]], xinner[j]);
                rinner[j] = sqrt(dnorm2(xinner[j]));
            }
            else
            {
                clear_dvec(xinner[j]);
                rinner[j] = 0.;
            }
        }

        /* We have ngrid volumes and ngrid+1 areas (parallel to the faces of the Ikosphere) that cover the
           volumes above and below. This loop computes the areas */
        for (i = 0; i<e->ngrid+1; i++)
        {
            /* Take radii at the bottom (or top) of the volumina (not at the center) */
            r = rmin + i*dr;

            /* Intersections of the 3 rays with the plane (vec{n}*vec{x} = r) */
            for (j = 0; j<3; j++)
            {
                a = r / diprod(e->ico->face[f].normal, e->r[iray[j]]);
                dsvmul(a, e->r[iray[j]], x[j]);
            }

            /* First get area of the triangle of 3 intersections of rays with plane (vec{n}*vec{x} = r) */
            area_r = gmx_envelope_triangle_2_area(x[0], x[1], x[2]);

            /* normalized vector towards the midpoint of the triangle */
            length = sqrt(dnorm2(e->surfElemOuter[f].center));
            dsvmul(1./length, e->surfElemOuter[f].center, xcentNorm);

            if (r <= (e->surfElemOuter[f].rmin + 1e-6) && (e->bOriginInside || r >= (e->surfElemInner[f].rmax - 1e-6)))
            {
                /* The normal case. The normal plane simply crosses the 3 rays. */
                /*printf("Doing the normal case in bin %d, r = %g,\nOuter: rmin %g, rmax %g\n"
                  "Inner: rmin %g, rmax %g", i, r, e->surfElemOuter[f].rmin, e->surfElemOuter[f].rmax,
                  e->bOriginInside ? 0. : e->surfElemInner[f].rmin,
                  e->bOriginInside ? 0. : e->surfElemInner[f].rmax); */
                area[i] = area_r;
                dsvmul(r, xcentNorm, xcomArea[i]);
            }
            else
            {
                /* Do a simple grid search over the triangle to estimate the area. Anything else is too complicated.
                   Use barycentric coordinates and split each side of the triangle into nAreaGrid. This way,
                   you get nAreaGrid^2 little triangles.
                */
                /*printf("Doing the UNUSUAL case in bin %d, r = %g, rmin %g, rmax %g\n", i, r, e->surfElemOuter[f].rmin,
                  e->surfElemOuter[f].rmax); */


                dvec_sub(x[1], x[0], u);
                dvec_sub(x[2], x[0], v);
                dsvmul(1./nAreaGrid, u, du);
                dsvmul(1./nAreaGrid, v, dv);
                ntriangles = 0;
                nInside    = 0;
                clear_dvec(xcomArea[i]);
                tol = (i == 0 || i == e->ngrid) ? 1e-5 : 0.;

                for (j = 0; j<nAreaGrid; j++)
                {
                    for (k = 0; k<(nAreaGrid-j + nAreaGrid-j-1); k++)
                    {
                        if (k < (nAreaGrid-j))
                        {
                            /* first row - upright little triangles */
                            xv = (1./3 + j);
                            xu = (1./3 + k);
                        }
                        else
                        {
                            /* second row - upside down little trigles */
                            xv = (2./3 + j);
                            xu = (2./3 + k-(nAreaGrid-j));
                        }
                        /* y = x0 + xv*dv + xu*du */
                        linear_vec_comb2(x[0], xv, dv, xu, du, y);
                        if (gmx_envelope_withinInnerOuter_thisSurfElem(e, f, y, tol))
                        {
                            dvec_inc(xcomArea[i], y);
                            nInside ++;
                        }
                        ntriangles++;
                    }
                }

                if (ntriangles != nAreaGrid*nAreaGrid)
                {
                    gmx_fatal(FARGS, "Inconsistency while doing grid over triangle\n");
                }

                // printf(" face %d, bin %d (r = %g), nInside = %d of %d\n", f, i, r, nInside, ntriangles);

                if (nInside == 0)
                {
                    /* If none of the triangle grids was inside, we have to go over the corners.
                       This may happen at the very first or very last point */
                    for (j = 0; j<3; j++)
                    {
                        // printf("No area bins within triangle, trying corner %g %g %g\n", x[j][0], x[j][1], x[j][2]);
                        if (gmx_envelope_withinInnerOuter_thisSurfElem(e, f, x[j], 1e-6))
                        {
                            // printf("OK, this corner inside\n");
                            dvec_inc(xcomArea[i], x[j]);
                            nInside ++;
                        }
                    }
                }

                if (nInside == 0)
                {
                    FILE *fp;
                    fp = fopen("env_error.py", "w");
                    fprintf(fp, "from pymol.cgo import *\nfrom pymol import cmd\n\n");
                    fprintf(fp, "obj = [\n\nCOLOR, %g, %g, %g,\n\n", 0.8, 0.8, 0.8);
                    gmx_envlope_triangle_to_CGO(e, fp, f, 0.08, 1, 0, 0);
                    dvec_to_CGO(fp,  x[0], 0.2);
                    dvec_to_CGO(fp,  x[1], 0.2);
                    dvec_to_CGO(fp,  x[2], 0.2);
                    fprintf(fp, "]\n\ncmd.load_cgo(obj, 'error')\n\n");
                    fclose(fp);

                    gmx_fatal(FARGS, "Still nInside == 0. ibin = %d of %d, r = %g\n", i, e->ngrid, r);
                }

                dsvmul(1./nInside, xcomArea[i], xcomArea[i]);
                areaGrid = 1.0*nInside/ntriangles*area_r;


                /* The second way - should be better than the grid search */

                for (itol = 0; itol<sizeof(tols)/sizeof(tols[0]); itol++)
                {
                    tol = tols[itol];
                    ncorners = 0;
                    for (j = 0; j<3; j++)
                    {
                        rray = sqrt(dnorm2(x[j]));
                        if ( (rinner[j]-tol) <= rray && rray <=  (router[j]+tol) && bNewCorner(x[j], corner, ncorners, 2*tol))
                        {
                            copy_dvec(x[j], corner[ncorners++]);
                        }

                        jj = (j+1)%3;
                        /* Search intersection of the vec{n}*vec{x} = r plane with the edges of the triangle */
                        if (r >= e->surfElemOuter[f].rmin - tol)
                        {
                            dvec_sub(xouter[jj], xouter[j], v);
                            n_dot_v = diprod(e->ico->face[f].normal, v);
                            if (fabs(n_dot_v) < 1e-15)
                            {
                                a = 1e15;
                            }
                            else
                            {
                                a = (r - diprod(e->ico->face[f].normal, xouter[j])) / n_dot_v;
                            }
                            if (-tol <= a && a <= 1.+tol)
                            {
                                dsvmul(a, v, vtmp);
                                dvec_add(xouter[j], vtmp, xIntersect);
                                if (bNewCorner(xIntersect, corner, ncorners, 2*tol))
                                {
                                    copy_dvec(xIntersect, corner[ncorners++]);
                                }
                            }
                        }
                        else if (!e->bOriginInside && r <= e->surfElemInner[f].rmax+tol)
                        {
                            dvec_sub(xinner[jj], xinner[j], v);
                            n_dot_v = diprod(e->ico->face[f].normal, v);
                            if (fabs(n_dot_v) < 1e-15)
                            {
                                a = 1e15;
                            }
                            else
                            {
                                a = (r - diprod(e->ico->face[f].normal, xinner[j])) / n_dot_v;
                            }
                            if (-tol < a && a < 1.+tol)
                            {
                                dsvmul(a, v, vtmp);
                                dvec_add(xinner[j], vtmp, xIntersect);
                                if (bNewCorner(xIntersect, corner, ncorners, tol))
                                {
                                    copy_dvec(xIntersect, corner[ncorners++]);
                                }
                            }
                        }
                    }
                    // printf("\nr = %g, found %d corners\n", r, ncorners);
                    if (ncorners == 1 || ncorners == 2)
                    {
                        area[i] = 0.;
                        break;
                    }
                    else if (ncorners == 3)
                    {
                        area[i] = gmx_envelope_triangle_2_area(corner[0], corner[1], corner[2]);
                        break;
                    }
                    else if (ncorners == 4)
                    {
                        area[i] = gmx_envelope_quadrangle_2_area(corner);
                        break;
                    }
                    else
                    {
                        if (itol == sizeof(tols)/sizeof(tols[0])-1)
                        {
                            gmx_fatal(FARGS, "Found %d corners for face %d, r = %g, bin %d of %d\n"
                                      "In this face: e->surfElemOuter[f].rmax %g \n"
                                      "R of ray intersections with normal: %g %g %g\n"
                                      "R of outer corners: %g %g %g\n"
                                      "diff = %g %g %g"
                                      "diff(with tol) = %g %g %g",
                                      ncorners, f, r, i, e->ngrid, e->surfElemOuter[f].rmax,
                                      sqrt(dnorm2(x[0])), sqrt(dnorm2(x[1])), sqrt(dnorm2(x[2])),
                                      router[0], router[1], router[2],
                                      router[0]-sqrt(dnorm2(x[0])), router[1]-sqrt(dnorm2(x[1])), router[2]-sqrt(dnorm2(x[2])),
                                      router[0]+tol-sqrt(dnorm2(x[0])), router[1]+tol-sqrt(dnorm2(x[1])),
                                      router[2]+tol-sqrt(dnorm2(x[2])));
                        }
                        else
                        {
                            fprintf(stderr, "\nNOTE: While getting volume bins of envelope face %d, r = %g, bin %d of %d:\n"
                                    "\t%d corners found with tolerance = %g. Now try with %g\n\n",
                                    f, r, i, e->ngrid, ncorners, tol, tols[itol+1]);
                        }
                    }
                } /* loop itol over tolerances */

                if (fabs(area[i]-areaGrid) > 1e-2*(area[i]+areaGrid) && fabs(area[i]-areaGrid) > 1e-1)
                {
                    FILE *fp;
                    fp = fopen("env_error.py", "w");
                    fprintf(fp, "from pymol.cgo import *\nfrom pymol import cmd\n\n");
                    fprintf(fp, "obj = [\n\nCOLOR, %g, %g, %g,\n\n", 0.8, 0.8, 0.8);
                    gmx_envlope_triangle_to_CGO(e, fp, f, 0.08, 1, 0, 0);
                    dvec_to_CGO(fp,  x[0], 0.15);
                    dvec_to_CGO(fp,  x[1], 0.15);
                    dvec_to_CGO(fp,  x[2], 0.15);
                    fprintf(fp, "COLOR, 0., 1., 0.,\n");
                    for (j=0; j<ncorners; j++)
                    {
                        dvec_to_CGO(fp,  corner[j], 0.2);
                        printf("Added corner %g %g %g to env_error.py\n", corner[j][0], corner[j][1], corner[j][2]);
                    }
                    fprintf(fp, "]\n\ncmd.load_cgo(obj, 'error')\n\n");
                    fclose(fp);

                    fprintf(stderr, "Cross section area of volume bin %d of %d: area from grid and from intersections do not"
                            " agree : %g -- %g\n"
                            "r = %g, %d corners found\n"
                            "nInside = %d of %d", i, e->ngrid, areaGrid, area[i], r, ncorners, nInside, ntriangles);
                }

                /* printf("areas (%d o %d) = %g %g diff = %g (rediff = %g %%)\n",i, e->ngrid,
                   area[i], areaGrid, area[i]-areaGrid, (area[i]-areaGrid)/areaGrid*100); */
            }

            if (! gmx_envelope_withinInnerOuter_thisSurfElem(e, f, xcomArea[i], 1e-5))
            {
                gmx_envlope_dump_triangle_x_to_CGO_file(e, f, xcomArea[i], "env_error.py");
                gmx_fatal(FARGS, "Center of area bin (%d, r = %g) x = (%g / %g / %g, r = %g) is not between inner and "
                          "outer surface of face %d (rmin = %g, rmax = %g)\n",
                          i, r, xcomArea[i][XX], xcomArea[i][YY], xcomArea[i][ZZ], sqrt(dnorm2(xcomArea[i])), f,
                          rmin, rmax);
            }
            if (icosphere_isInFace(e->ico, &(e->ico->face[f]), xcomArea[i], 1e-5, NULL) <= 0)
            {
                gmx_fatal(FARGS, "Center of area bin (%d, r = %g) x = (%g / %g / %g, r = %g) is not inside face %d\n",
                          i, r, xcomArea[i][XX], xcomArea[i][YY], xcomArea[i][ZZ], sqrt(dnorm2(xcomArea[i])), f);
            }
        } /* end i loop over volume grid */

        /* Store the volumina */
        vsum = 0.;
        for (i = 0; i<e->ngrid; i++)
        {
            e->binVol[f][i] = 0.5*dr*(area[i]+area[i+1]);
            vsum += e->binVol[f][i];
            if (e->binVol[f][i] <= 0)
            {
                gmx_fatal(FARGS, "Strange, got a <= volueme (%g) for f = %d, i = %d \n", e->binVol[f][i], f, i);
            }
            dvec_add(xcomArea[i], xcomArea[i+1], v);
            /* double to real conversion */
            e->xBinVol[f][i][XX] = 0.5*v[XX];
            e->xBinVol[f][i][YY] = 0.5*v[YY];
            e->xBinVol[f][i][ZZ] = 0.5*v[ZZ];
            if (icosphere_isInFace(e->ico, &(e->ico->face[f]), v, 1e-3, NULL) <= 0)
            {
                gmx_fatal(FARGS, "Center of volume bin %d (face %d) is not within this face: x = %g / %g / %g\n", i, f,
                          e->xBinVol[f][i][XX], e->xBinVol[f][i][YY], e->xBinVol[f][i][ZZ]);
            }
        }
        vtotalsum += vsum;

        vexpect = e->surfElemOuter[f].volPyramid;
        if (!e->bOriginInside)
        {
            vexpect -= e->surfElemInner[f].volPyramid;
        }

        /* Check releative difference betwwen the two ways to compute the volumen, as compared to the
           average volume of the volume bins */
        relVolDiff = fabs(vsum-vexpect)/(e->vol/(e->nSurfElems*e->ngrid));
        if (relVolDiff > 1e-2)
        {
            nVolWarn++;
        }
        if (relVolDiff > 1e-2 && nVolWarn == 1 && e->bVerbose && FALSE)
        {
            fprintf(stderr, "\n\nWARNING --- In face %d:"
                    "\n\tSum of volumina of volume bins (%g) does not equal the "
                    "volume of the pyramides (%g) (rel. diff = %g %%)\n\tConsider using a deeper recusion level deeper than %d "
                    "for the envelope's ikosphere.\n\n",
                    f, vsum, vexpect, 100*fabs(vsum-vexpect)/vexpect, e->nrec);
        }

        if (fabs(vsum-vexpect)/vexpect > 1e-1)
        {
            gmx_fatal(FARGS, "Sum of volumina of volume bins (%g) does not equal the volume from pyramides (%g) (rel diff = %g %%)\n",
                      vsum, vexpect, 100*fabs(vsum-vexpect)/vexpect);
        }
    } /* End of loop over faces */

    if (nVolWarn > 0 &&  e->bVerbose && FALSE)
    {
        printf("\nWARNING - There were %d warnings in total because the sum of volume bins did not\n equal the result "
               "from the pyramides.\n\n", nVolWarn);
    }
    relVolDiff = e->vol/vtotalsum;
    if (e->bVerbose)
    {
        printf("Total volume of envelope: exact = %g, from bins = %g (%g %% difference)\n"
               "\t -> scaling volume bins by %g\n", e->vol, vtotalsum, 100*(e->vol-vtotalsum)/e->vol, relVolDiff);
    }
    for (f = 0; f<e->nSurfElems; f++)
    {
        if (e->surfElemOuter[f].bDefined)
        {
            for (i = 0; i<e->ngrid; i++)
            {

                e->binVol[f][i] *= relVolDiff;
            }
        }
    }

    sfree(area);
    sfree(xcomArea);
}


/* assign isDefined of 1 to a ray if inner and outer surface are defined,
   and set maxOuter and minInner */
static void
gmx_envelope_setStats(gmx_envelope_t e)
{
    int j, f;
    double vsum;

    e->minInner = 1e20;
    e->minOuter = 1e20;
    e->maxInner = 0.;
    e->maxOuter = 0.;
    e->nDefined = 0;

    for (j=0; j<e->nrays; j++)
    {
        e->minInner = (e->inner[j] < e->minInner) ? e->inner[j] : e->minInner;
        e->maxInner = (e->inner[j] > e->maxInner) ? e->inner[j] : e->maxInner;

        e->minOuter = (e->outer[j] < e->minOuter) ? e->outer[j] : e->minOuter;
        e->maxOuter = (e->outer[j] > e->maxOuter) ? e->outer[j] : e->maxOuter;

        if (e->outer[j] > 0.)
        {
            e->isDefined[j] = TRUE;
            e->nDefined++;
            if (e->inner[j] > e->outer[j])
            {
                gmx_fatal(FARGS, "Envelope error, ray %d: outer is %g, but inner is %g\n",
                          j, e->outer[j], e->inner[j]);
            }
        }
        else
        {
            e->isDefined[j] = FALSE;
        }
    }

    if (e->nDefined < 0.2*e->nrays && e->bVerbose)
    {
        fprintf(stderr,"\n\nWARNING -- Only %.2f %% of envelope rays (%d rays) are defined.\n"
                "\tMaybe your solute is far away from the origin?\n\n", 100.*e->nDefined/e->nrays, e->nDefined);
    }
    if (e->nDefined == 0)
    {
        gmx_fatal(FARGS, "No envelope rays are defined - something went wrong\n");
    }

    if (e->maxInner == 0.0)
    {
        /* set in case that that the envelope was not build on this node */
        e->bOriginInside = TRUE;
    }
    else
    {
        e->bOriginInside = FALSE;
    }

    gmx_envelope_build_md5(e);

    if (e->surfElemOuter == NULL)
    {
        snew(e->surfElemOuter, e->nSurfElems);
    }
    gmx_envelope_set_surfaceElements(e, e->surfElemOuter, e->outer);
    if (! e->bOriginInside)
    {
        if (e->surfElemInner == NULL)
        {
            snew(e->surfElemInner, e->nSurfElems);
        }
        gmx_envelope_set_surfaceElements(e, e->surfElemInner, e->inner);
    }

    /* Compute total volume */
    vsum = 0.;
    for (f = 0; f<e->nSurfElems; f++)
    {
        if (e->surfElemOuter[f].bDefined)
        {
            vsum += e->surfElemOuter[f].volPyramid;
        }
        if (! e->bOriginInside && e->surfElemInner[f].bDefined)
        {
            vsum -= e->surfElemInner[f].volPyramid;
        }
    }
    e->vol = vsum;
    if (e->bVerbose)
    {
        printf("Volume of envelope [nm3] = %12.8g\n", e->vol);
    }

    gmx_envelope_binVolumes(e);

    gmx_envelope_build_bounding_sphere(e);
}

void
gmx_envelope_bcast(gmx_envelope_t e, t_commrec *cr)
{
    int j;

    if (MASTER(cr) && e->bHaveEnv == FALSE)
    {
        gmx_fatal(FARGS, "Trying to bcast envelope to slaves, but the envelope is not defined on the Master\n");
    }

    for (j=0; j<e->nrays; j++)
    {
        gmx_bcast(e->nrays*sizeof(double), e->inner, cr);
        gmx_bcast(e->nrays*sizeof(double), e->outer, cr);
    }

    if (!e->bHaveEnv)
    {
        gmx_envelope_setStats(e);
    }

    e->bHaveEnv = TRUE;
}

gmx_envelope_t gmx_envelope_init_md(int nrecReq, t_commrec *cr, gmx_bool bVerbose)
{
    gmx_bool bEnvFile;
    gmx_envelope_t e = NULL;
    char *envfile = NULL;
    int nrec;

    if (MASTER(cr))
    {
        envfile = getenv("GMX_ENVELOPE_FILE");
        bEnvFile = ( envfile != NULL);
    }
    if (PAR(cr))
    {
        gmx_bcast(sizeof(gmx_bool), &bEnvFile, cr);
    }
    if (!bEnvFile)
    {
        /* Only init the envelope, don't have the envelope yet. */
        if (nrecReq < 0)
        {
            nrecReq = WAXS_ENVELOPE_ICOSPHERE_NREC;
        }
        e = gmx_envelope_init(nrecReq, bVerbose);
    }
    else
    {
        /* If we read the envelope from a file, we bcast it to all the nodes immediately */
        if (MASTER(cr))
        {
            printf("\nFound environment variable GMX_ENVELOPE_FILE -- reading envelope from file %s\n", envfile);
            e = gmx_envelope_readFromFile(envfile);
            nrec = e->nrec;
            if (nrecReq >= 0 && nrec != nrecReq)
            {
                fprintf(stderr, "\n\nWARNING -- requested to build an envelope with recusion level %d\n"
                        "\tFile %s contains %d recursion levels, however\n\n", nrecReq, envfile, nrec);
            }
        }
        if (PAR(cr))
        {
            gmx_bcast(sizeof(int), &nrec, cr);
        }

        if (!MASTER(cr))
        {
            e = gmx_envelope_init(nrec, bVerbose);
        }
        if (PAR(cr))
        {
            gmx_envelope_bcast(e, cr);
        }
        if (MASTER(cr) && PAR(cr))
        {
            printf("Broadcast envelope to all the nodes\n");
        }
    }

    return e;
}

/* void gmx_envelope_expandEnvelope(gmx_envelope_t e, rvec x[], atom_id *index, int isize) */
/* { */
/*     double d2 = sqr(e->d); */
/* } */


void gmx_envelope_superimposeEnvelope(gmx_envelope_t e_add, gmx_envelope_t e_base)
{
    int i, tot;

    /* Safety checks */
    if ( !e_base->bHaveEnv || !e_add->bHaveEnv )
    {
        gmx_fatal(FARGS,"The envelopes are not defined for at least one of the inputs!\n");
    }
    if ( e_base->nrec != e_add->nrec )
    {
        fprintf(stderr,"The recursion levels of the two envelopes are different!\n");
        gmx_fatal(FARGS,"Bailing out.\n");
    }
    /* Now should have the same number of rays in both. */
    tot=e_base->nrays;
    for (i=0; i<tot; i++)
    {
        e_base->outer[i] = (e_base->outer[i] < e_add->outer[i]) ? e_add->outer[i] : e_base->outer[i];
    }

    /* Also update inner if required. Otherwise always 0. */
    if ( !e_base->bOriginInside )
    {
        for (i=0; i<tot; i++)
        {
            e_base->inner[i] = (e_base->inner[i] > e_add->inner[i]) ? e_add->inner[i] : e_base->inner[i];
        }
    }

    gmx_envelope_setStats(e_base);

}

void
gmx_envelope_buildSphere(gmx_envelope_t e, real rad)
{
    int i;

    for (i=0; i<e->nrays; i++)
    {
        e->inner[i]=0.;
        e->outer[i]=rad;
    }
    e->bHaveEnv=TRUE;

    gmx_envelope_setStats(e);
}

void
gmx_envelope_buildEllipsoid(gmx_envelope_t e, rvec r)
{
    int i;

    gmx_fatal(FARGS, "Building an ellipsoindal envelope is currently not implemented\n");

    for (i=0; i<e->nrays; i++)
    {
        /* This seems wrong: */
        // dsvmul(e->inner[iray] * 10, e->r[iray], x[0]);
        e->inner[i]=0.;
        e->outer[i]=0;
    }
    e->bHaveEnv=TRUE;

    /* Remove warning for now */
    r[0] = 0;

    gmx_envelope_setStats(e);
}

void
gmx_envelope_buildEnvelope_inclShifted_omp(gmx_envelope_t e, rvec x[], atom_id *index,
                                           int isize, real dGiven, real phiSmooth, gmx_bool bSphere)
{
    int i, k, iVertex1, iVertex2, f;
    atom_id *index4;
    real normx, normdir, normShifted;
    rvec *xInclShifted, triangleCenter[3], dir, vertex1, vertex2, tmp;
    dvec xd;

    /* Add 3 extra atoms which are shifted (by dGiven) perpendicular to the side of the triangle,
       in which the atom is located. If we have a highly streched envelope in which faces are not
       defined, this way we avoid that atoms are near pyramides which are not defined
    */

    snew(xInclShifted, isize*4);
    snew(index4, isize*4);
    for (i = 0; i<isize; i++)
    {
        copy_rvec(x[index[i]], xInclShifted[i*4]);

        normx = sqrt(norm2(x[i]));

        /* get index of the face in which this x is located */
        copy_rdvec(x[index[i]], xd);
        f = icosphere_x2faceID(e->ico, xd);

        /* Loop over the three sides of the triangular face, in which this x is located */
        for (k = 0; k < 3; k++)
        {
            /* Get center of this side of the triangle, and normalize wrt. the length of x */
            iVertex1 = e->ico->face[f].v[k      ];
            iVertex2 = e->ico->face[f].v[(k+1)%3];
            copy_drvec(e->ico->vertex[iVertex1], vertex1);
            copy_drvec(e->ico->vertex[iVertex2], vertex2);
            rvec_add(vertex1, vertex2, triangleCenter[k]);
            svmul(0.5*normx, triangleCenter[k], triangleCenter[k]);

            /* Move x outside of its triangle by dGiven, in the direction of the center of the triangle side */
            rvec_sub(triangleCenter[k], x[index[i]], dir);
            normdir = sqrt(norm2(dir));
            svmul(dGiven/normdir, dir, dir);

            /* Make sure that the shifted atom has the same distance from origin, and store in xInclShifted */
            rvec_add(x[i], dir, tmp);
            normShifted = sqrt(norm2(tmp));
            svmul(normx/normShifted, tmp, xInclShifted[i*4 + 1 + k]);
        }
    }

    /* get a simple index group 0,1,2,...,isize*4-1 */
    for (i = 0; i<isize*4; i++)
    {
        index4[i] = i;
    }

    FILE *fp = fopen("test.pdb", "w");
    for (i = 0; i<isize*4; i++)
    {
        fprintf(fp, "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n", i+1, "O", "ELE", "",
                1, 10*xInclShifted[i][XX], 10*xInclShifted[i][YY], 10*xInclShifted[i][ZZ], 1.0, 1.0);
    }
    fclose(fp);

    gmx_envelope_buildEnvelope_omp(e, xInclShifted, index4,
                                   4*isize, dGiven, phiSmooth, bSphere);
    sfree(index4);
    sfree(xInclShifted);
}

void
gmx_envelope_buildEnvelope_omp(gmx_envelope_t e, rvec x[], atom_id *index,
                               int isize, real dGiven, real phiSmooth, gmx_bool bSphere)
{
    int i, j, f;
    double theta, d2, maxR, innerMax, outerMin;
    dvec *r = e->r;

    e->d = dGiven;
    d2 = sqr(e->d);

    /* first check if the origin is within the envelope */
    if (e->bOriginInside == FALSE)
    {
        for (i=0; i<isize; i++)
        {
            if (norm2(x[index[i]]) < d2)
            {
                e->bOriginInside = TRUE;
                break;
            }
        }
    }
    if (!e->bOriginInside)
    {
        printf("\nNote: The origin is not within the envelope\n\n");
    }
    if (e->bVerbose)
    {
        printf("\nBuilding envelope around %d atoms\n", isize);
    }

    /* loop over atoms */

    /* #pragma omp parallel                                                                                         \ */
    /*         shared(x,index,e,r,d2)                                                                                         \ */
    /*         private(outer,inner,xd,xnorm2,xnorm,xdNorm,minOuter,inv_xnorm,j,r_dot_x,cosGamma,l1,l2,discr,sqrtDiscr) */

    #pragma omp parallel shared(x,index,e,r,d2) private(j)
    {
        /* Varialbes private to each thread: */
        double discr, sqrtDiscr, xnorm2, xnorm, inv_xnorm, l1, l2, cosGamma, r_dot_x, *outer, *inner, minOuter = 0, tmp;
        dvec xd, xdNorm;
        int threadID = gmx_omp_get_thread_num();
        int nThreads = gmx_omp_get_max_threads();

        snew(outer, e->nrays);
        snew(inner, e->nrays);
        for (j = 0; j < e->nrays; j++)
        {
            outer[j] = 0.;
            inner[j] = 1e20;
        }

        #pragma omp for
        for (i = 0; i < isize; i++)
        {
            copy_rdvec(x[index[i]], xd);
            xnorm2    = dnorm2(xd);
            xnorm     = sqrt(xnorm2);
            dsvmul(1./xnorm, xd, xdNorm);

            /* Check if we can skip this atom */
            if (e->bOriginInside && (xnorm+e->d < minOuter))
            {
                continue;
            }

            inv_xnorm = 1.0/xnorm;
            // printf("Envelope: x = %g %g %g\n", x[index[i]][XX], x[index[i]][YY], x[index[i]][ZZ]);

            if (e->bVerbose && (((i+1)%10000) == 0 || (i == isize-1)) && threadID == 0)
            {
                printf("\r        %8.1f %% done", 100.*(i+1)/(isize/nThreads));
                fflush(stdout);
            }

            if ( ((i+1) % 5000) == 0)
            {
                /* Every 5000th atom, get the minimum outer surface - used to skip atoms
                   with r < minOuter-d */
                minOuter = 1e20;
                for (j = 0; j < e->nrays; j++)
                {
                    minOuter = (outer[j] < minOuter) ? outer[j] : minOuter;
                }
            }

            /* loop over envelope rays */
            for (j = 0; j < e->nrays; j++)
            {

                /* Angle between x and r */
                r_dot_x  = diprod(r[j], xd);
                cosGamma = r_dot_x*inv_xnorm;
                /* Intersection of envelope ray with the sphere of radius d round x.
                   Use law of cosine:

                   l[1,2] = x.cos(gamma) +- sqrt[ (x.cos(gamma))^2 - (x^2-d^2) ]
                */
                discr = sqr(r_dot_x) - (xnorm2-d2);

                /* The origin is NOT within the sphere of radius d round x
                   -> may bet 2 solutions */
                if (xnorm2 >= d2)
                {
                    /* The origin is NOT within the sphere of radius d round x
                       -> ray crosses the sphere only if gamma<90degree, that is cosGamma > 0
                       -> have 2 solutions if discr > 0 (discr == 0 case is ignored) */

                    if (discr > 0. && cosGamma > 0.)
                    {
                        sqrtDiscr = sqrt(discr);
                        l1        = r_dot_x - sqrtDiscr;
                        l2        = r_dot_x + sqrtDiscr;
#ifdef ENVELOPE_DEBUG
                        if (l1 < 0. || l2 < 0.)
                        {
                            gmx_fatal(FARGS, "Error while building envelope, found negative distance from"
                                      "origin:\n l1 = %g, l2 = %g  (|x|*cos(gamma) = %g, sqrt(discr) = %g\n"
                                      "cosGamma = %g",
                                      l1, l2, xnorm*cosGamma, sqrtDiscr, cosGamma);
                        }
                        if (l1 > l2)
                        {
                            gmx_fatal(FARGS, "Error while constructing envelope: l1 > l2: %g > %g\n",
                                      l1, l2);
                        }
#endif
                        inner[j] = (l1 < inner[j]) ? l1 : inner[j];
                        outer[j] = (l2 > outer[j]) ? l2 : outer[j];
                    }
                }
                else
                {
                    /* The origin is WITHIN the sphere of radius d around x
                       -> want only the positive solution of the law of cosine */
                    l2 = r_dot_x + sqrt(discr);

#ifdef ENVELOPE_DEBUG
                    if (l2 < 0.)
                    {
                        gmx_fatal(FARGS, "Error while building envelope, found negative distance from"
                                  "origin:\n l2 = %g (|x|*cos(gamma) = %g, sqrt(discr) = %g\n"
                                  "cosGamma = %g",
                                  l2, xnorm*cosGamma, sqrt(discr), cosGamma);
                    }
#endif

                    /* everything between 0 and l2 is withing protein */
                    inner[j] = 0.;
                    outer[j] = (l2 > outer[j]) ? l2 : outer[j];
                }
            }
        }

        /* Now collect maximum outer[] / minimal inner[] from the threads */
        for (j = 0; j < e->nrays; j++)
        {
            tmp = outer[j];
            #pragma omp flush(tmp)
            if (tmp > e->outer[j])
            {
                #pragma omp critical
                {
                    if (tmp > e->outer[j])
                    {
                        e->outer[j] = tmp;
                    }
                }
            }
            tmp = inner[j];
            #pragma omp flush(tmp)
            if (tmp < e->inner[j])
            {
                #pragma omp critical
                {
                    if (tmp < e->inner[j])
                    {
                        e->inner[j] = tmp;
                    }
                }
            }

        }
        sfree(inner);
        sfree(outer);
    } /* end omp pagma */


    /* For all faces, the radii for ALL three inner corners must be smaller than the
     * radii of all three outer corners. Otherwise, our method to construct the volume bins
     * fails. Check for this:
     */
    for (f = 0; f<e->ico->nface; f++)
    {
        innerMax = 0;
        outerMin = 1e20;
        for (i = 0; i<3; i++)
        {
            j = e->ico->face[f].v[i];
            innerMax = (e->inner[j] > innerMax) ? e->inner[j] : innerMax;
            outerMin = (e->outer[j] < outerMin) ? e->outer[j] : outerMin;
        }
        if (outerMin < innerMax && innerMax < 0.9e20 && outerMin > 0)
        {
            printf("\nThe pyramide of face %d is highly skewed (innerMax = %g - outerMin = %g), "
                   "\treducing the inner radii to %g\n", f, innerMax, outerMin, outerMin);
            printf("Radii of inner = ");
            for (i = 0; i<3; i++) printf(" %10g", e->inner[e->ico->face[f].v[i]]);
            printf("\n");
            printf("Radii of outer = ");
            for (i = 0; i<3; i++) printf(" %10g", e->outer[e->ico->face[f].v[i]]);
            printf("\n");
            for (i = 0; i<3; i++)
            {
                j = e->ico->face[f].v[i];
                if (e->inner[j] > outerMin)
                {
                    e->inner[j] = outerMin - GMX_FLOAT_EPS;
                }
            }
        }
    }

    if (bSphere)
    {
        maxR = 0.;
        for (j=0; j<e->nrays; j++)
        {
            e->inner[j] = 0.;
            maxR = (e->outer[j] > maxR) ? e->outer[j] : maxR;
        }
        for (j=0; j<e->nrays; j++)
        {
            e->outer[j] = maxR;
        }
    }

    if (isize > 50000 && e->bVerbose)
    {
        printf("\n");
    }

    if (e->bOriginInside)
    {
        for (j=0; j<e->nrays; j++)
        {
            /* In the main loop above, we may have skipped a few atoms that are far inside of the envelope,
               because we knew already that bOriginInside = TRUE. Consequently, inner[j] may not be zero here,
               and we must fix this (example is PDB code 1RJW with g_genenv -d 0.7): */
            e->inner[j] = 0.;
        }
    }

    if (! e->bOriginInside)
    {
        for (j=0; j<e->nrays; j++)
        {
            if (e->outer[j] > 0. && e->inner[j] == 0.)
            {
                gmx_fatal(FARGS, "ray %d: outer = %g, inner = %g, this is not possible when the origin is not within "
                          "the envelope\n", j, e->outer[j], e->inner[j]);
            }
        }
    }

    if (phiSmooth > 0.)
    {
        gmx_envelope_smoothEnvelope(e, phiSmooth);
    }

    if (e->bVerbose)
    {
        printf("\n");
    }
    gmx_envelope_setStats(e);

    printf("\nConstructed envelope around %d atoms.\n\tInner surface spanning radii from %g to %g.\n"
           "\tOuter surface spanning radii from %g to %g\n"
           "\t%d of %d directions defined.\n\n",
           isize, e->minInner, e->maxInner,  e->minOuter, e->maxOuter, e->nDefined, e->nrays);

    e->bHaveEnv = TRUE;
}


void
gmx_envelope_buildEnvelope(gmx_envelope_t e, rvec x[], atom_id *index,
                           int isize, real dGiven, real phiSmooth)
{
    int i, j;
    double theta, phi, d2, discr, sqrtDiscr, xnorm2, xnorm, inv_xnorm, l1, l2, cosGamma, r_dot_x;
    double minOuter = 0, maxInner;
    dvec *r = e->r, xd, xdNorm;
    int ncont = 0;

    e->d = dGiven;
    d2 = sqr(e->d);

    /* first check if the origin is within the envelope */
    if (e->bOriginInside == FALSE)
    {
        for (i=0; i<isize; i++)
        {
            if (norm2(x[index[i]]) < d2)
            {
                e->bOriginInside = TRUE;
                break;
            }
        }
    }
    if (!e->bOriginInside)
    {
        printf("\nNote: The origin is not within the envelope\n\n");
    }
    if (e->bVerbose)
    {
        printf("\nBuilding envelope around %d atoms\n", isize);
    }

    for (i = 0; i < isize; i++)
    {
        copy_rdvec(x[index[i]], xd);
        xnorm2    = dnorm2(xd);
        xnorm     = sqrt(xnorm2);
        dsvmul(1./xnorm, xd, xdNorm);

        /* Check if we can skip this atom */
        if (e->bOriginInside && (xnorm+e->d < minOuter))
        {
            ncont++;
            continue;
        }

        inv_xnorm = 1.0/xnorm;
        // printf("Envelope: x = %g %g %g\n", x[index[i]][XX], x[index[i]][YY], x[index[i]][ZZ]);

        if (e->bVerbose && (((i+1)%50000) == 0 || (i == isize-1)))
        {
            printf("\r        %8.1f %% done", 100.*(i+1)/isize);
            fflush(stdout);
        }

        if ( ((i+1) % 5000) == 0)
        {
            /* Every 5000th atom, get the minimum outer surface - used to skip atoms
               with r < minOuter-d */
            minOuter = 1e20;
            for (j = 0; j < e->nrays; j++)
            {
                if (e->outer[j] < minOuter)
                {
                    minOuter = e->outer[j];
                }
            }
        }

        /* loop over envelope rays */
        for (j = 0; j < e->nrays; j++)
        {

            /* Angle between x and r */
            r_dot_x  = diprod(r[j], xd);
            cosGamma = r_dot_x*inv_xnorm;
            /* Intersection of envelope ray with the sphere of radius d round x.
               Use law of cosine:

               l[1,2] = x.cos(gamma) +- sqrt[ (x.cos(gamma))^2 - (x^2-d^2) ]
            */
            discr = sqr(r_dot_x) - (xnorm2-d2);

            /* The origin is NOT within the sphere of radius d round x
               -> may bet 2 solutions */
            if (xnorm2 >= d2)
            {
                /* The origin is NOT within the sphere of radius d round x
                   -> ray crosses the sphere only if gamma<90degree, that is cosGamma > 0
                   -> have 2 solutions if discr > 0 (discr == 0 case is ignored) */

                if (discr > 0. && cosGamma > 0.)
                {
                    sqrtDiscr = sqrt(discr);
                    l1        = r_dot_x - sqrtDiscr;
                    l2        = r_dot_x + sqrtDiscr;
#ifdef ENVELOPE_DEBUG
                    if (l1 < 0. || l2 < 0.)
                    {
                        gmx_fatal(FARGS, "Error while building envelope, found negative distance from"
                                  "origin:\n l1 = %g, l2 = %g  (|x|*cos(gamma) = %g, sqrt(discr) = %g\n"
                                  "cosGamma = %g",
                                  l1, l2, xnorm*cosGamma, sqrtDiscr, cosGamma);
                    }
                    if (l1 > l2)
                    {
                        gmx_fatal(FARGS, "Error while constructing envelope: l1 > l2: %g > %g\n",
                                  l1, l2);
                    }
#endif
                    e->inner[j] = (l1 < e->inner[j]) ? l1 : e->inner[j];
                    e->outer[j] = (l2 > e->outer[j]) ? l2 : e->outer[j];
                }
            }
            else
            {
                /* The origin is WITHIN the sphere of radius d around x
                   -> want only the positive solution of the law of cosine */
                l2 = r_dot_x + sqrt(discr);

#ifdef ENVELOPE_DEBUG
                if (l2 < 0.)
                {
                    gmx_fatal(FARGS, "Error while building envelope, found negative distance from"
                              "origin:\n l2 = %g (|x|*cos(gamma) = %g, sqrt(discr) = %g\n"
                              "cosGamma = %g",
                              l2, xnorm*cosGamma, sqrt(discr), cosGamma);
                }
#endif

                /* everything between 0 and l2 is withing protein */
                e->inner[j] = 0.;
                e->outer[j] = (l2 > e->outer[j]) ? l2 : e->outer[j];
            }
        }
    }

    if (isize > 50000 && e->bVerbose)
    {
        printf("\n");
    }

    if (e->bOriginInside)
    {
        for (j=0; j<e->nrays; j++)
        {
            if (fabs(e->inner[j]) > 0.)
            {
                gmx_fatal(FARGS, "Something is strange. Origin is inside envelope, but inner[%d] = %g\n", j, e->inner[j]);
            }
            e->inner[j] = 0.;
        }
    }

    if (! e->bOriginInside)
    {
        for (j=0; j<e->nrays; j++)
        {
            if (e->outer[j] > 0. && e->inner[j] == 0.)
            {
                gmx_fatal(FARGS, "ray %d: outer = %g, inner = %g, this is not possible when the origin is not within "
                          "the envelope\n", j, e->outer[j], e->inner[j]);
            }
        }
    }

    if (phiSmooth > 0.)
    {
        gmx_envelope_smoothEnvelope(e, phiSmooth);
    }

    gmx_envelope_setStats(e);

    printf("\nConstructed envelope around %d atoms.\n\tInner surface spanning radii from %g to %g.\n"
           "\tOuter surface spanning radii from %g to %g\n"
           "\t%d of %d directions defined.\n\n",
           isize, e->minInner, e->maxInner,  e->minOuter, e->maxOuter, e->nDefined, e->nrays);

    e->bHaveEnv = TRUE;

    printf("\nCould skip %d of %d (%g %%)  -  minOuter %g\n",
           ncont, isize, 100.*ncont/isize, minOuter);
}

void
gmx_envelope_clearSurface(gmx_envelope_t e)
{
    int j;

    /* loop over envelope rays */
    for (j=0; j<e->nrays; j++)
    {
        e->inner[j]  = 1e20;
        e->outer[j]  = 0.;
        e->isDefined[j] = FALSE;
    }
    e->bHaveEnv = FALSE;
    e->minInner = -1;
    e->maxOuter = -1;
}

gmx_bool gmx_envelope_bHaveSurf(gmx_envelope_t e)
{
    return e->bHaveEnv;
}

gmx_bool gmx_envelope_getNrec(gmx_envelope_t e)
{
    return e->nrec;
}

double gmx_envelope_getVolume(gmx_envelope_t e)
{
    if (! e->bHaveEnv)
    {
        gmx_fatal(FARGS, "gmx_envelope_getVolume: envelope not defined.\n");
    }

    return e->vol;
}

double gmx_envelope_maxR(gmx_envelope_t e)
{
    if (! e->bHaveEnv)
    {
        gmx_fatal(FARGS, "gmx_envelope_maxR: envelope not defined.\n");
    }

    return e->maxOuter;
}

char *gmx_envelope_chksum_str(gmx_envelope_t e)
{
    return e->chksum_str;
}

void  gmx_envelope_writeVMDCGO(gmx_envelope_t e, const char * fn, rvec rgb, real alpha)
{
    FILE *fp;
    int i, j, jj, iray;
    double r, g, b, rad = 1.0;
    dvec diff1, diff2, normal, x[3], ccyl;
    const double rcyl = 0.06 ;
    const int res = 10 ;
    gmx_bool bDrawCyl = FALSE ;

    ccyl[0]=0.8*256;
    ccyl[1]=0.8*256;
    ccyl[2]=0.8*256;

    if (e->nSurfElems == 0)
    {
        gmx_fatal(FARGS, "Trying to write envelope triangles to file, but no triangles were defined\n");
    }

    if (rgb == NULL)
    {
        /* magenta */
        r = 212/256.0;
        g =  31/256.0;
        b = 123/256.0;
    }
    else
    {
        r = rgb[0];
        g = rgb[1];
        b = rgb[2];
    }

    fp = ffopen(fn, "w");

    /* Prepare molecule material.*/
    fprintf(fp, "proc draw-envelope {} {\n");
    /* fprintf(fp, "mol new\n"); */
    /* fprintf(fp, "mol rename top %s\n", name); */
    fprintf(fp, "graphics top color magenta\n");
    if ( alpha < 1.0 && alpha > 0.0 )
    {
        /* fprintf(fp, "material add copy Glossy\n");
           fprintf(fp, "material rename Material22 GlossyGlass\n");
           fprintf(fp, "material change opacity GlossyGlass %g\n", alpha);
           fprintf(fp, "graphics top material GlossyGlass\n"); */
        fprintf(fp, "material change opacity Glossy %g\n", alpha);
        fprintf(fp, "graphics top material Glossy\n");
    }
    else
    {
        fprintf(fp, "graphics top material Glossy\n");
    }

    /* Write origin */
    fprintf(fp, "graphics top sphere {0 0 0} radius %g\n", rad);

    /* Set colour and material */
    fprintf(fp, "color change rgb 32 %g %g %g\n", r, g, b);
    fprintf(fp, "graphics top color 32\n");

    /* Write triangles */
    for (i = 0; i<e->nSurfElems; i++)
    {
        if (e->surfElemOuter[i].bDefined)
        {
            fprintf(fp, "graphics top triangle ");
            for (j = 0; j<3; j++)
            {
                iray = e->surfElemOuter[i].ind[j];
                dsvmul(e->outer[iray] * 10, e->r[iray], x[0]);
                fprintf(fp, "{%g %g %g} ", x[0][XX], x[0][YY], x[0][ZZ]);
            }
            fprintf(fp, "\n");
        }
        if (e->bOriginInside == FALSE && e->surfElemInner[i].bDefined)
        {
            fprintf(fp, "graphics top triangle ");
            for (j = 0; j<3; j++)
            {
                iray = e->surfElemInner[i].ind[j];
                dsvmul(e->inner[iray] * 10, e->r[iray], x[0]);
                fprintf(fp, "{%g %g %g} ", x[0][XX], x[0][YY], x[0][ZZ]);
            }
            fprintf(fp, "\n");
        }
    }

    /* Now print cylinders on the edges of the surface elements */
    if ( bDrawCyl )
    {
        fprintf(fp, "color change rgb 31 %g %g %g\n", ccyl[0], ccyl[1], ccyl[2]);
        fprintf(fp, "graphics top color 31\n");
        for (i = 0; i<e->nSurfElems; i++)
        {
            if (e->surfElemOuter[i].bDefined)
            {
                for (j = 0; j<3; j++)
                {
                    iray =  e->surfElemOuter[i].ind[j];
                    dsvmul(e->outer[iray] * 10, e->r[iray], x[j]);
                }
                for (j = 0; j<3; j++)
                {
                    jj = (j+1) % 3;
                    fprintf(fp, "graphics top cylinder {%g %g %g} {%g %g %g} radius %g resolution %i filled no\n",
                            x[ j][XX], x[ j][YY], x[ j][ZZ],
                            x[jj][XX], x[jj][YY], x[jj][ZZ],
                            rcyl, res);
                }
            }
            if (e->bOriginInside == FALSE && e->surfElemInner[i].bDefined)
            {
                for (j = 0; j<3; j++)
                {
                    iray =  e->surfElemInner[i].ind[j];
                    dsvmul(e->inner[iray] * 10, e->r[iray], x[j]);
                }
                for (j = 0; j<3; j++)
                {
                    jj = (j+1) % 3;
                    fprintf(fp, "graphics top cylinder {%g %g %g} {%g %g %g} radius %g resolution %i filled no\n",
                            x[ j][XX], x[ j][YY], x[ j][ZZ],
                            x[jj][XX], x[jj][YY], x[jj][ZZ],
                            rcyl, res);
                }
            }

            /* Draw the lines in radial direction to connect inner and outer surface elements.
               If the surface element is not defined but the ray on a corner is, then draw a cylinder */
            if (!e->bOriginInside && !e->surfElemOuter[i].bDefined)
            {
                for (j = 0; j<3; j++)
                {
                    /* Check if the ray is defined */
                    iray =  e->surfElemInner[i].ind[j];
                    if (e->isDefined[iray])
                    {
                        dsvmul(e->inner[iray] * 10, e->r[iray], x[0]);
                        dsvmul(e->outer[iray] * 10, e->r[iray], x[1]);
                        fprintf(fp, "graphics top cylinder {%g %g %g} {%g %g %g} radius %g resolution %i filled no\n",
                                x[0][XX], x[0][YY], x[0][ZZ],
                                x[1][XX], x[1][YY], x[1][ZZ],
                                rcyl, res);
                    }
                }
            }
        }
    }
    fprintf(fp, "\nputs \"Done drawing.\"\n}\ndraw-envelope\n");
    ffclose(fp);
    printf("Wrote envelope as VMD-tcl into %s\n", fn);
}

void gmx_envelope_writePymolCGO(gmx_envelope_t e, const char * fn, const char *name, rvec rgb, rvec rgb_inside, real alpha)
{

    FILE *fp;
    gmx_bool bTrianglesInside;
    int i, j, jj, iray, nDefined, iDefined[3];
    double r, g, b, a;
    dvec v1, v2, normal, x[4], faceCent;
    const double rcyl = 0.06;
    const double mvInnerTriangles = 0.01;  /* Distance in Angstroem of the inner surface triangles */

    if (e->nSurfElems == 0)
    {
        gmx_fatal(FARGS, "Trying to write envelope triangles to file, but no triangles were defined\n");
    }

    bTrianglesInside = (rgb_inside != NULL);
    if (rgb == NULL)
    {
        /* magenta */
        r = 202./256;
        g =  31./256;
        b = 123./256;
    }
    else
    {
        r = rgb[0];
        g = rgb[1];
        b = rgb[2];
    }

    if (alpha < 0)
    {
        alpha = 1;
    }

    fp = ffopen(fn, "w");
    /* Write origin */
    fprintf(fp, "from pymol.cgo import *\nfrom pymol import cmd\n\n");
    fprintf(fp, "obj3 = [\nCOLOR, %g, %g, %g,\n", 0.9, 0.0, 0.9);
    fprintf(fp, "SPHERE, %g,%g,%g, %g,\n", 0., 0., 0., 0.5);
    fprintf(fp, "]\n\n");
    fprintf(fp, "cmd.load_cgo(obj3, '%s_origin')\n\n", name);

    /* Write triangles */
    /* fprintf(fp, "obj= [\n\nBEGIN, TRIANGLES,\n\nALPHA, %g,\n", alpha, r, g, b); Alpha ignored anyway by pymol */
    fprintf(fp, "obj= [\n\nBEGIN, TRIANGLES,\n\n\n");
    for (i = 0; i<e->nSurfElems; i++)
    {
        if (e->surfElemOuter[i].bDefined)
        {
            fprintf(fp, "COLOR, %g, %g, %g,\n", r, g, b);
            fprintf(fp, "NORMAL, %g, %g, %g,\n", e->surfElemOuter[i].normal[XX],
                    e->surfElemOuter[i].normal[YY], e->surfElemOuter[i].normal[ZZ]);

            for (j = 0; j<3; j++)
            {
                iray = e->surfElemOuter[i].ind[j];
                dsvmul(e->outer[iray] * 10, e->r[iray], x[0]);
                fprintf(fp, "VERTEX, %g, %g, %g,\n", x[0][XX], x[0][YY], x[0][ZZ]);
            }
            if (bTrianglesInside)
            {
                /* Add triangle for the inner surface of the envelope - otherwise it's just black in Pymol */
                fprintf(fp, "COLOR, %g, %g, %g,\n", rgb_inside[XX], rgb_inside[YY], rgb_inside[ZZ]);
                fprintf(fp, "NORMAL, %g, %g, %g,\n", -e->surfElemOuter[i].normal[XX],
                        -e->surfElemOuter[i].normal[YY], -e->surfElemOuter[i].normal[ZZ]);

                for (j = 0; j<3; j++)
                {
                    iray = e->surfElemOuter[i].ind[j];
                    /* Move vertices slightly inside, by 0.05 Angstroem */
                    dsvmul(e->outer[iray] * 10 - mvInnerTriangles, e->r[iray], x[0]);
                    fprintf(fp, "VERTEX, %g, %g, %g,\n", x[0][XX], x[0][YY], x[0][ZZ]);
                }

            }
        }
        if (e->bOriginInside == FALSE && e->surfElemInner[i].bDefined)
        {
            dsvmul(-1., e->surfElemInner[i].normal, normal);
            fprintf(fp, "COLOR, %g, %g, %g,\n", r, g, b);
            fprintf(fp, "NORMAL, %g, %g, %g,\n", normal[XX], normal[YY], normal[ZZ]);

            for (j = 0; j<3; j++)
            {
                iray = e->surfElemInner[i].ind[j];
                dsvmul(e->inner[iray] * 10, e->r[iray], x[0]);
                fprintf(fp, "VERTEX, %g, %g, %g,\n", x[0][XX], x[0][YY], x[0][ZZ]);
            }
            if (bTrianglesInside)
            {
                /* Add triangle for the inner surface of the envelope - otherwise it's just black in Pymol */
                fprintf(fp, "COLOR, %g, %g, %g,\n", rgb_inside[XX], rgb_inside[YY], rgb_inside[ZZ]);
                fprintf(fp, "NORMAL, %g, %g, %g,\n", -normal[XX], -normal[YY], -normal[ZZ]);

                for (j = 0; j<3; j++)
                {
                    iray = e->surfElemInner[i].ind[j];
                    /* Move vertices slightly outside, by 0.05 Angstroem */
                    dsvmul(e->inner[iray] * 10 + mvInnerTriangles, e->r[iray], x[0]);
                    fprintf(fp, "VERTEX, %g, %g, %g,\n", x[0][XX], x[0][YY], x[0][ZZ]);
                }
            }
        }
        /* Draw quadrilaterals on the surfaces that conect the outer with the inner surface.
           If this triangle is not defined, but two of its corerns are defined, then draw the quadrilateral */
        if (!e->bOriginInside && !e->surfElemOuter[i].bDefined)
        {
            nDefined = 0;
            for (j = 0; j<3; j++)
            {
                /* Check if the ray is defined */
                iray =  e->ico->face[i].v[j];
                if (e->isDefined[iray])
                {
                    iDefined[nDefined++] = iray;
                }
            }
            if (nDefined == 2)
            {
                dsvmul(e->inner[iDefined[0]] * 10, e->r[iDefined[0]], x[0]);
                dsvmul(e->outer[iDefined[0]] * 10, e->r[iDefined[0]], x[1]);
                dsvmul(e->inner[iDefined[1]] * 10, e->r[iDefined[1]], x[2]);
                dsvmul(e->outer[iDefined[1]] * 10, e->r[iDefined[1]], x[3]);
                dvec_sub(x[1], x[0], v1);
                dvec_sub(x[2], x[0], v2);
                dcprod(v1, v2, normal);
                /* Get direction of normal right - is normal pointing the direction of the center of this face? */
                clear_dvec(faceCent);
                for (j = 0; j<3; j++)
                {
                    iray = e->ico->face[i].v[j];
                    dvec_inc(faceCent, e->ico->vertex[iray]);
                }
                dsvmul(1./3, faceCent, faceCent);
                dvec_sub(faceCent, x[0], v1);
                if (diprod(normal, v1) < 0.)
                {
                    dsvmul(-1, normal, normal);
                }
                fprintf(fp, "COLOR, %g, %g, %g,\n", r, g, b);
                fprintf(fp, "NORMAL, %g, %g, %g,\n", normal[XX], normal[YY], normal[ZZ]);
                fprintf(fp, "VERTEX, %g, %g, %g,\n", x[0][XX], x[0][YY], x[0][ZZ]);
                fprintf(fp, "VERTEX, %g, %g, %g,\n", x[1][XX], x[1][YY], x[1][ZZ]);
                fprintf(fp, "VERTEX, %g, %g, %g,\n", x[2][XX], x[2][YY], x[2][ZZ]);
                fprintf(fp, "NORMAL, %g, %g, %g,\n", normal[XX], normal[YY], normal[ZZ]);
                fprintf(fp, "VERTEX, %g, %g, %g,\n", x[1][XX], x[1][YY], x[1][ZZ]);
                fprintf(fp, "VERTEX, %g, %g, %g,\n", x[2][XX], x[2][YY], x[2][ZZ]);
                fprintf(fp, "VERTEX, %g, %g, %g,\n", x[3][XX], x[3][YY], x[3][ZZ]);
                if (bTrianglesInside)
                {
                    /* Add triangles for the inner surface */
                    a = mvInnerTriangles;
                    fprintf(fp, "COLOR, %g, %g, %g,\n", r, g, b);
                    fprintf(fp, "NORMAL, %g, %g, %g,\n", -normal[XX], -normal[YY], -normal[ZZ]);
                    fprintf(fp, "VERTEX, %g, %g, %g,\n", x[0][XX]-normal[XX]*a, x[0][YY]-normal[YY]*a, x[0][ZZ]-normal[ZZ]*a);
                    fprintf(fp, "VERTEX, %g, %g, %g,\n", x[1][XX]-normal[XX]*a, x[1][YY]-normal[YY]*a, x[1][ZZ]-normal[ZZ]*a);
                    fprintf(fp, "VERTEX, %g, %g, %g,\n", x[2][XX]-normal[XX]*a, x[2][YY]-normal[YY]*a, x[2][ZZ]-normal[ZZ]*a);
                    fprintf(fp, "NORMAL, %g, %g, %g,\n", -normal[XX], -normal[YY], -normal[ZZ]);
                    fprintf(fp, "VERTEX, %g, %g, %g,\n", x[1][XX]-normal[XX]*a, x[1][YY]-normal[YY]*a, x[1][ZZ]-normal[ZZ]*a);
                    fprintf(fp, "VERTEX, %g, %g, %g,\n", x[2][XX]-normal[XX]*a, x[2][YY]-normal[YY]*a, x[2][ZZ]-normal[ZZ]*a);
                    fprintf(fp, "VERTEX, %g, %g, %g,\n", x[3][XX]-normal[XX]*a, x[3][YY]-normal[YY]*a, x[3][ZZ]-normal[ZZ]*a);
                }
            }
        }
    }

    fprintf(fp, "\nEND ]\n\n");
    fprintf(fp, "cmd.load_cgo(obj,'%s')\n\n", name);

    /* Now print cylinders on the edges of the surface elements */
    fprintf(fp, "obj2 = [\n\nCOLOR, %g, %g, %g,\n\n", 0.8, 0.8, 0.8);
    for (i = 0; i<e->nSurfElems; i++)
    {
        if (e->surfElemOuter[i].bDefined)
        {
            for (j = 0; j<3; j++)
            {
                iray =  e->surfElemOuter[i].ind[j];
                dsvmul(e->outer[iray] * 10, e->r[iray], x[j]);
            }
            for (j = 0; j<3; j++)
            {
                jj = (j+1) % 3;
                fprintf(fp, "CYLINDER, %g,%g,%g, %g,%g,%g, %g, %g,%g,%g,%g,%g,%g,\n",
                        x[j ][XX], x[j ][YY], x[j ][ZZ],
                        x[jj][XX], x[jj][YY], x[jj][ZZ],
                        rcyl,
                        .1, .7, .7, .1, .7, .7);
            }
        }
        if (e->bOriginInside == FALSE && e->surfElemInner[i].bDefined)
        {
            for (j = 0; j<3; j++)
            {
                iray =  e->surfElemInner[i].ind[j];
                dsvmul(e->inner[iray] * 10, e->r[iray], x[j]);
            }
            for (j = 0; j<3; j++)
            {
                jj = (j+1) % 3;
                fprintf(fp, "CYLINDER, %g,%g,%g, %g,%g,%g, %g, %g,%g,%g,%g,%g,%g,\n",
                        x[j ][XX], x[j ][YY], x[j ][ZZ],
                        x[jj][XX], x[jj][YY], x[jj][ZZ],
                        rcyl,
                        .1, .7, .7, .1, .7, .7);
            }
        }

        /* Draw the lines in radial direction to connect inner and outer surface elements.
           If the surface element is not defined but the ray on a corner is, then draw a cylinder */
        if (!e->bOriginInside && !e->surfElemOuter[i].bDefined)
        {
            for (j = 0; j<3; j++)
            {
                /* Check if the ray is defined */
                iray =  e->surfElemInner[i].ind[j];
                if (e->isDefined[iray])
                {
                    dsvmul(e->inner[iray] * 10, e->r[iray], x[0]);
                    dsvmul(e->outer[iray] * 10, e->r[iray], x[1]);
                    fprintf(fp, "CYLINDER, %g,%g,%g, %g,%g,%g, %g, %g,%g,%g,%g,%g,%g,\n",
                            x[0][XX], x[0][YY], x[0][ZZ],
                            x[1][XX], x[1][YY], x[1][ZZ],
                            rcyl,
                            .1, .7, .7, .1, .7, .7);
                }
            }
        }
    }
    fprintf(fp, "]\n\n");
    fprintf(fp, "cmd.load_cgo(obj2, '%s_mesh')\n\n", name);

    ffclose(fp);
    printf("Wrote envelope as pymol CGO into %s\n", fn);
}


void gmx_envelope_writeToFile(gmx_envelope_t e, const char * fn)
{
    int j;
    FILE *fp;

    fp = ffopen(fn, "w");
    fprintf(fp, "%d\n", e->nrec);
    for (j = 0; j < e->nrays; j++)
    {
        fprintf(fp, "%.12g %.12g\n", e->inner[j], e->outer[j]);
    }
    ffclose(fp);
}

gmx_envelope_t gmx_envelope_readFromFile(const char * fn)
{
    int j = 0, nrec, nrays;
    FILE *fp;
    char line[200];
    gmx_envelope_t e;
    char fm[10] = "%lf %lf";

    fp = ffopen(fn, "r");
    if (fgets(line, 200, fp) == NULL)
    {
        gmx_fatal(FARGS, "Error while reading the first line of file %s\n", fn);
    }
    if (sscanf(line, "%d", &nrec) != 1)
    {
        gmx_fatal(FARGS, "Expected exactly one number in the first line of %s\n", fn);
    }

    e = gmx_envelope_init(nrec, TRUE);
    nrays = e->nrays;
    printf("Initiating envelope from file %s with %d rays (recusion level %d)\n", fn, nrays, nrec);

    while(fgets(line, 200, fp) != NULL)
    {
        if (j>=nrays)
        {
            gmx_fatal(FARGS, "Found (at least) %d inner / outer surfaces in %s. Expected only %d\n",
                      j+1, fn, nrays);
        }
        if (sscanf (line, fm, &e->inner[j], &e->outer[j]) != 2)
        {
            gmx_fatal(FARGS, "Error, could not read 2 numers (innter/outer surface) from %s (j=%d). Line was:\n%s\n",
                      fn, j, line);
        }
        j++;
    }
    if (j != nrays)
    {
        gmx_fatal(FARGS, "Expected %d inner / outer surfaces in envelope file %s. Found %d\n", nrays, j);
    }
    ffclose(fp);
    printf("Read surfaces for %d envelope rays from file %s\n", j, fn);

    gmx_envelope_setStats(e);

    printf("\tInner surface spanning radii from %g to %g nm.\n"
           "\tOuter surface spanning radii from %g to %g nm\n"
           "\t%d of %d directions defined.\n\n",
           e->minInner, e->maxInner,  e->minOuter, e->maxOuter, e->nDefined, e->nrays);

    e->bHaveEnv = TRUE;
    return e;
}

static void
gmx_envelope_FourierTransform_low(gmx_envelope_t e, rvec *q, int qhomenr, real *ft_re, real *ft_im,
                                  gmx_bool bSolventDens)
{
    int jj, nq2, nReal;

    #pragma omp parallel for
    for (jj = 0; jj < qhomenr; jj++)
    {
        ft_re[jj] = 0.;
        ft_im[jj] = 0.;
    }

    nReal = sizeof(REGISTER)/sizeof(real);
    nq2   = ((qhomenr-1)/nReal+1)*nReal;

    #pragma omp parallel shared(ft_re, ft_im )
    {
        int j, i, f, k;
        REGISTER *m_qdotx,*mRe,*mIm;
        real *p_qdotx, *re, *im, fact, *ft_re_loc, *ft_im_loc;

        snew_aligned(p_qdotx,   nq2, 32);
        snew_aligned(re,        nq2, 32);
        snew_aligned(im,        nq2, 32);
        snew(ft_re_loc, qhomenr);
        snew(ft_im_loc, qhomenr);


        #pragma omp for
        for (f = 0; f<e->nSurfElems; f++)
        {
            if (! e->surfElemOuter[f].bDefined)
            {
                continue;
            }

            if (e->bVerbose && ((f % 20) == 0))
            {
                printf("\r%7.2f %% done", (f+1.)*100./e->nSurfElems);
                fflush(stdout);
            }

//            fprintf(stderr, "Number surface elements %d , OpenMP Thread Id %d\n ", f, gmx_omp_get_thread_num() );
            {
                for (i = 0; i < e->ngrid; i++)
                {
                    // fprintf(stderr, "start f = %d, i = %d\n", f, i);
                    if (bSolventDens)
                    {
                        /* When doing FT of the solvent, multiply here by the # of electrons in this bin */
                        fact = e->solventNelec[f][i];
                    }
                    else
                    {
                        /* For FT of a unit density, multiply by the volume of the bin */
                        fact = e->binVol[f][i];
                    }

                    if (fact == 0.0)
                    {
                        continue;
                    }

                    /* Write q*x into array */
                    for (j = 0; j < qhomenr; j++)
                    {
                        //                    fprintf(stderr, "q_qdotx %f \n ", p_qdotx[j] );
                        p_qdotx[j] = iprod(q[j], e->xBinVol[f][i]);

                    }

                    m_qdotx = (REGISTER*) p_qdotx;
                    mRe     = (REGISTER*) re;
                    mIm     = (REGISTER*) im;

                    /* Compute all sines and cosines AVX/FMA instructions */
                    for (k = 0; k < qhomenr; k += nReal)
                    {
                        //#define GMX_WAXS_NO_ACCELERATION
#ifndef GMX_WAXS_NO_ACCELERATION
                        REGISTER_SINCOS(*m_qdotx, mIm, mRe);
#else
                        *mIm = SIN(*m_qdotx);
                        *mRe = COS(*m_qdotx);
#endif
                        m_qdotx++;
                        mRe++;
                        mIm++;
                    }

                    /* And write result into ft_re/ft_im arrays */
                    for (j = 0; j < qhomenr; j++)
                    {
                        ft_re_loc[j] += fact*re[j];
                        ft_im_loc[j] += fact*im[j];
                    }
                }
            }
        }
        #pragma omp critical
        {
            for(j = 0; j< qhomenr; j++)
            {
                ft_im[j] += ft_im_loc[j];
                ft_re[j] += ft_re_loc[j];
            }
        }
        sfree_aligned(p_qdotx);
        sfree_aligned(re);
        sfree_aligned(im);
        sfree(ft_re_loc);
        sfree(ft_im_loc);
    }

    for (jj = 0; jj<qhomenr; jj++)
    {
        if (isnan(ft_re[jj]) || isinf(ft_re[jj]) || isnan(ft_im[jj]) || isinf(ft_im[jj]))
        {
            gmx_fatal(FARGS, "Nan/Inf error after fourier transform: j = %d, ft_re / ft_im = %g / %g\n",
                      jj, ft_re[jj], ft_im[jj]);
        }
    }


    if (e->bVerbose)
    {
        printf("...done\n\n");
    }
}


void
gmx_envelope_unitFourierTransform(gmx_envelope_t e, rvec *q, int nq, real **ft_re, real **ft_im)
{
    int i, j;
    FILE *fp;
    char buf[1024];

    if (! e->bHaveEnv)
    {
        gmx_fatal(FARGS, "Cannot compute FT of envelope since the envelope is not yet constructed.\n");
    }

    if (e->bHaveFourierTrans == FALSE)
    {
        if (!e->ftunit_re) snew(e->ftunit_re, nq);
        if (!e->ftunit_im) snew(e->ftunit_im, nq);

        if (e->bVerbose)
        {
            printf("\nDoing Fourier transform of envelope... (%d surface elements, %d grid, nq %d -> %.2e sin/cos)\n",
                   e->nSurfElems, e->ngrid, nq, 1.0*e->nSurfElems*e->ngrid*nq);
            fflush(stdout);
        }

        gmx_envelope_FourierTransform_low(e, q, nq, e->ftunit_re, e->ftunit_im, FALSE);
        e->bHaveFourierTrans = TRUE;
    }


    /* return pointers to the FT (if != NULL) */
    if (ft_re)
    {
        *ft_re = e->ftunit_re;
    }
    if (ft_im)
    {
        *ft_im = e->ftunit_im;
    }
}

gmx_bool
gmx_envelope_bHaveUnitFT(gmx_envelope_t e)
{
    return e->bHaveFourierTrans;
}

gmx_bool
gmx_envelope_bHaveSolventFT(gmx_envelope_t e)
{
    return e->bHaveSolventFT;
}

void
gmx_envelope_solventFourierTransform(gmx_envelope_t e, rvec *q, int nq, gmx_bool bRecalcFT,
                                     real **ft_re, real **ft_im)
{
    gmx_bool bCalcFT = bRecalcFT;

    if (e->nSolventStep == 0)
    {
        gmx_fatal(FARGS, "Envelope: cannot compute FT of density, since the density has not been computed yet\n");
    }

    if (e->ftdens_re == NULL)
    {
        snew(e->ftdens_re, nq);
        snew(e->ftdens_im, nq);
        bCalcFT = TRUE;
    }

    if (bCalcFT)
    {
        if (e->bVerbose)
        {
            printf("\nDoing Fourier transform of solvent density... (%d surface elements, %d grid, nq %d)\n",
                   e->nSurfElems, e->ngrid, nq);
            fflush(stdout);
        }

        gmx_envelope_FourierTransform_low(e, q, nq, e->ftdens_re, e->ftdens_im, TRUE);

        e->bHaveSolventFT = TRUE;
    }

    /* return pointers to the FT (if != NULL) */
    if (ft_re)
    {
        *ft_re = e->ftdens_re;
    }
    if (ft_im)
    {
        *ft_im = e->ftdens_im;
    }
}


/* cumulative average of solvent density
   tau specifies the exponentially decaying weight (tau = 0 means non-weighted average)

   scale = exp(-dt/tau) gives the decay of weights for cumulative average. scale == 1 means non-weighted average
*/
void gmx_envelope_solvent_density_nextFrame(gmx_envelope_t e,  double scale)
{
    int f, ibin;

    if (!e->bHaveEnv)
    {
        gmx_fatal(FARGS, "Error, trying update solvent density, but "
                  "envelope is not defined\n");
    }

    /* Norm and factors for cumulative averaging: D[n] = fac1*D[n-1] + fac2*d[n] */
    e->solventNelecNorm  = 1.0 + scale * e->solventNelecNorm;
    e->solventNelec_fac1 = 1.0*(e->solventNelecNorm - 1.)/e->solventNelecNorm;
    e->solventNelec_fac2 = 1.0/e->solventNelecNorm;

    for (f = 0; f<e->nSurfElems; f++)
    {
        for (ibin = 0; ibin < e->ngrid; ibin++)
        {
            e->solventNelec[f][ibin] *= e->solventNelec_fac1;
        }
    }

    e->nSolventStep++;
}

void gmx_envelope_solvent_density_addAtom(gmx_envelope_t e, const rvec xin, double nelec)
{
    int ibin, f;
    double rmin, rmax, dr, r;
    dvec x;

    copy_rdvec(xin, x);
    f = icosphere_x2faceID(e->ico, x);

    /* Important: r is here not simply the distance from the origin, since the volume bins
       have a flat surface. Instead, our r is the distance AFTER projecting on normal vector
       of the face */
    r = diprod(e->ico->face[f].normal, x);

    if (! e->surfElemOuter[f].bDefined)
    {
        gmx_fatal(FARGS, "Error while updatig solvent density inside envelope\n"
                  "Atom (x = %g / %g / %g) is in a non-defined face of the envelope\n",
                  x[XX], x[YY], x[ZZ]);
    }

    rmin = e->bOriginInside ? 0.0 : e->surfElemInner[f].rmin;
    rmax = e->surfElemOuter[f].rmax;
    if (r>rmax || r<rmin)
    {
        gmx_fatal(FARGS, "Error while updatig solvent density inside envelope\n"
                  "Atom (r = %g ) is outside of the envelope (rmin = %g, rmax = %g)\n",
                  r, rmin, rmax);
    }

    dr                       = (rmax-rmin)/e->ngrid;
    ibin                     = floor((r-rmin)/dr);
    if (ibin == e->ngrid && r < (rmax+1e-5))
    {
        /* catch numerical inaccuracy */
        ibin = e->ngrid-1;
    }

    if (ibin < 0 || ibin >= e->ngrid)
    {
        gmx_fatal(FARGS, "Incorrect ibin = %d found (ngrid = %d)\n", ibin, e->ngrid);
    }

    /* Add density for the present frame with appriate weight fac2 */
    e->solventNelec[f][ibin] += e->solventNelec_fac2 * nelec;

    if (isinf(e->solventNelec[f][ibin]) || isnan(e->solventNelec[f][ibin]))
    {
        fprintf(stderr, "Inf/ Nan for f/ ibin = %d / %d. fac2 = %g, nelec = %g\n", f, ibin, e->solventNelec_fac2,
                nelec);
    }
    // printf("e->solventNelec[f][ibin] = %d %d : %g  (nelec = %g)\n", f, ibin, e->solventNelec[f][ibin], nelec);
}

void gmx_envelope_solvent_density_bcast(gmx_envelope_t e, t_commrec *cr)
{
    int ibin, f;
    double sumNelec = 0.;

    if (cr && PAR(cr))
    {
        // fprintf(stderr, "Bcasting solvent density\n");
        for (f = 0; f<e->nSurfElems; f++)
        {
            gmx_bcast(e->ngrid*sizeof(double), e->solventNelec[f], cr);
            for (ibin = 0; ibin < e->ngrid; ibin++)
            {
                sumNelec += e->solventNelec[f][ibin];
                if (isnan(e->solventNelec[f][ibin]) || isinf(e->solventNelec[f][ibin]))
                {
                    gmx_fatal(FARGS, "Found NaN or Inf for solvent density of envelope: solventNelec[%d][%d] = %g\n",
                              f, ibin, e->solventNelec[f][ibin]);
                }
            }
        }
    }

    e->solventNelecTotal = 0.;
    for (f = 0; f<e->nSurfElems; f++)
    {
        for (ibin = 0; ibin < e->ngrid; ibin++)
        {
            e->solventNelecTotal += e->solventNelec[f][ibin];
        }
    }

    if (e->bVerbose && FALSE)
    {
        printf("\tNode %2d: Total # of electrons in solvation shell = %g\n", cr->nodeid, sumNelec);
    }
}

double
gmx_envelope_solvent_density_getNelecTotal(gmx_envelope_t e)
{
    if (e->nSolventStep == 0)
    {
        gmx_fatal(FARGS, "Cannot report number of electrons of solvent in envelope - not computed yet\n");
    }
    return e->solventNelecTotal;
}

void gmx_envelope_solvent_density_2pdb(gmx_envelope_t e, const char *fn)
{
    int f, ibin, iat = 1;
    double max = -1.;
    FILE *fp;
    const double bfacmax = 50.;
    dvec x, xcentNorm;
    double r, length, dr, rmax, rmin, bfac, tmp;


    for (f = 0; f<e->nSurfElems; f++)
    {
        if (e->surfElemOuter[f].bDefined)
        {
            for (ibin = 0; ibin < e->ngrid; ibin++)
            {
                tmp = e->solventNelec[f][ibin]/e->binVol[f][ibin];
                max = tmp > max ? tmp : max;
            }
        }
    }

    fp = fopen(fn, "w");
    printf("File %s open (max = %g)\n", fn, max);

    for (f = 0; f < e->nSurfElems; f++)
    {
        if (e->surfElemOuter[f].bDefined)
        {
            rmin = e->bOriginInside ? 0.0 : e->surfElemInner[f].rmin;
            rmax = e->surfElemOuter[f].rmax;
            dr   = (rmax-rmin)/e->ngrid;

            length = sqrt(dnorm2(e->surfElemOuter[f].center));
            dsvmul(1./length, e->surfElemOuter[f].center, xcentNorm);

            for (ibin = 0; ibin < e->ngrid; ibin++)
            {
                if (e->solventNelec[f][ibin] > 0.)
                {
                    bfac = bfacmax/max * e->solventNelec[f][ibin]/e->binVol[f][ibin];
                    if (ibin == 0) printf("bfac = %g\n", bfac);
                    r    = rmin + (ibin + 0.5)*dr;
                    dsvmul(r, xcentNorm, x);
                    fprintf(fp, "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n", iat++, "O", "ELE", "",
                            1, 10*x[XX], 10*x[YY], 10*x[ZZ], 1.0, bfac);
                }
            }
        }
    }
    fclose(fp);
    printf("Wrote solvent density to pdb file %s\n", fn);
}



int gmx_envelope_getStepSolventDens(gmx_envelope_t e)
{
    return e->nSolventStep;
}

/* Just wrapper functions visible from outside to use the grid density */
void gmx_envelope_griddensity_nextFrame(gmx_envelope_t e, double scale)
{
    /* Init density grid if not done yet */
    if (!e->grid_density)
    {
        dvec xmin = {1e20, 1e20, 1e20}, xmax = {-1e20, -1e20, -1e20}, x;
        int j, d;
        double spacing = WAXS_ENVELOPE_GRID_DENSITY_SPACING;

        if (getenv("WAXS_ENVELOPE_GRID_DENSITY_SPACING") != NULL)
        {
            spacing = atof(getenv("WAXS_ENVELOPE_GRID_DENSITY_SPACING"));
            printf ("Found environment variable WAXS_ENVELOPE_GRID_DENSITY_SPACING. Using grid spacing of %g for density in evelope.\n", spacing);
        }

        /* First get largest and smallest corners enclosing the envelope */
        for (j = 0; j < e->nrays; j++)
        {
            if (e->isDefined[j])
            {
                dsvmul(e->outer[j], e->r[j], x);
                for (d = 0; d < DIM; d++)
                {
                    xmax[d] = (x[d] > xmax[d]) ? x[d] : xmax[d];
                    xmin[d] = (x[d] < xmin[d]) ? x[d] : xmin[d];
                }
            }
        }
        e->grid_density = grid_density_init(xmin, xmax, spacing);
    }

    /* Now open the grid for adding atoms */
    grid_density_nextFrame(e->grid_density, scale);
}
void gmx_envelope_griddensity_closeFrame(gmx_envelope_t e)
{
    if (!e->grid_density)
    {
        gmx_fatal(FARGS, "Inconsitency, trying to use grid_density but grid_density has not been initialized\n");
    }
    grid_density_closeFrame(e->grid_density);
}
void gmx_envelope_griddensity_addAtom(gmx_envelope_t e, const rvec x, const double nElec)
{
    if (!e->grid_density)
    {
        gmx_fatal(FARGS, "Inconsitency, trying to use grid_density but grid_density has not been initialized\n");
    }
    if (gmx_envelope_isInside(e, x))
    {
        grid_density_addAtom(e->grid_density, x, nElec);
    }
}
void gmx_envelope_griddensity_write(gmx_envelope_t e, const char *fn)
{
    if (!e->grid_density)
    {
        gmx_fatal(FARGS, "Inconsitency, trying to use grid_density but grid_density has not been initialized\n");
    }
    grid_density_write(e->grid_density, fn);
}

/* Computing the volume of the envelope using Monte-Carlo Integration. The function creates a
   cuboid box that encloses the envelope, gerates nTry_d random points in this box, and uses
   gmx_envelope_isInside() to test if the point is inside.
   Result: Seems like we get excellent agreement with the analytic result in
   gmx_envelope_getVolume(), within the statistical uncertainty.
*/
void
gmx_envelope_volume_montecarlo_omp(gmx_envelope_t e, double nTry_d, int seed0)
{
    int              i, j, iray, d, iround;
    gmx_large_int_t  nTry = nTry_d, *nInside_round, nTryPerThread;
    dvec             x, L, min = {1e20, 1e20, 1e20}, max = {-1e20, -1e20, -1e20};
    double           vol, volerr, *vol_round, tmp = 0;
    const int        NROUNDS = 50;
    char             fmt[1024];
    int              envelope_nseed, nThreads = gmx_omp_get_max_threads();

    if (!e->bHaveEnv)
    {
        gmx_fatal(FARGS, "Error, trying to compute envelope volume using Monte-Carlo, but "
                  "envelope is not defined\n");
    }

    /* Use more seeds for RNG in case of many insertions - gives higher entropy in the RNG */
    if (nTry_d < 1e9)
    {
        envelope_nseed = 2;
    }
    else
    {
        envelope_nseed = 4;
    }

    for (i = 0; i<e->nSurfElems; i++)
    {
        if (e->surfElemOuter[i].bDefined)
        {
            for (j = 0; j<3; j++)
            {
                iray =  e->surfElemOuter[i].ind[j];
                dsvmul(e->outer[iray], e->r[iray], x);

                for (d = 0; d < DIM; d++)
                {
                    if (x[d] > max[d])
                    {
                        max[d] = x[d];
                    }
                    if (x[d] < min[d])
                    {
                        min[d] = x[d];
                    }
                }
            }
        }
    }

    for (d = 0; d < DIM; d++)
    {
        L[d] = max[d] - min[d];
    }

    snew(vol_round,     NROUNDS);
    snew(nInside_round, NROUNDS);

    /* Make sure nTry can be divided by the number of threads */
    nTry = nTry_d/NROUNDS;
    nTry = (nTry/nThreads)*nThreads;
    nTryPerThread = nTry/nThreads;

    printf("\nComputing envelope volume from Monte Carlo using %g insertion trials, %d rounds:\n", nTry_d, NROUNDS);
    printf("\tmin coordinates of verticies = %12g %12g %12g\n", min[XX], min[YY], min[ZZ]);
    printf("\tmax coordinates of verticies = %12g %12g %12g\n", max[XX], max[YY], max[ZZ]);

    #pragma omp parallel shared(L,min) private(iround)
    {
        /* Varialbes private to each thread: */
        gmx_rng_t        rng = NULL;
        gmx_large_int_t  imc, nInside_loc[NROUNDS];
        rvec             R;
        unsigned int     dl, is, seed[envelope_nseed];
        char             fmt_loc[1024];
        int              threadID = gmx_omp_get_thread_num();

        /* Init the random number generator with different seeds in every thread */
        if (seed0 < 0)
        {
            for (is = 0; is < envelope_nseed; is++)
            {
                seed[is] = gmx_rng_make_seed();
            }
        }
        else
        {
            for (is = 0; is < envelope_nseed; is++)
            {
                seed[is] = seed0 + is + threadID*envelope_nseed;
            }
        }
        rng = gmx_rng_init_array(seed, envelope_nseed);

        if (threadID == 0)
        {
            printf("\tRandom numbers initiated\n");
            fflush(stdout);
        }

        for (iround = 0; iround < NROUNDS; iround++)
        {
            nInside_loc[iround] = 0;

            if (threadID == 0)
            {
                printf("\r\t%5.0f %% done", 100.0*(iround+1)/NROUNDS);
                fflush(stdout);
            }

            /* The long loop over the Monte Carlo trials */
            for (imc = 0; imc < nTryPerThread; imc++)
            {
                for (dl = 0; dl < DIM; dl++)
                {
                    R[dl] = gmx_rng_uniform_real(rng) * L[dl] + min[dl];
                }

                if (gmx_envelope_isInside(e, R))
                {
                    nInside_loc[iround]++;
                }
            }
        }

        #pragma omp critical
        for (iround = 0; iround < NROUNDS; iround++)
        {
            nInside_round[iround] += nInside_loc[iround];
        }
    } /* end omp pagma */
    printf("\n");

    for (iround = 0; iround < NROUNDS; iround++)
    {
        vol_round[iround] = 1.0*nInside_round[iround]/nTry*L[XX]*L[YY]*L[ZZ];
    }

    vol    = 0;
    volerr = 0;
    for (iround = 0; iround < NROUNDS; iround++)
    {
        vol += vol_round[iround];
    }
    vol /= NROUNDS;

    for (iround = 0; iround < NROUNDS; iround++)
    {
        tmp += dsqr(vol_round[iround]-vol);
    }
    volerr = sqrt(tmp/NROUNDS) / sqrt(1.0*NROUNDS);

    printf("Volume = %12.8g +- %12g\n", vol, volerr);

    sfree(vol_round);
}
