
#include "visibility.h"
#include "typedefs.h"
#include "sysstuff.h"
#include "types/commrec.h"
#include "types/pbc.h"

#ifndef GMX_ENVELOPE_H_
#define GMX_ENVELOPE_H_

#ifndef _gmx_enveloope_h
#define _gmx_enveloope_h

#ifdef __cplusplus
extern "C" {
#endif
#if 0
} /* fixes auto-indentation problems */
#endif

/*
   Provide an abastract data type to use the envelope functionality.
   This way, all envelope-internal variables are hidden.
   The communication with the envelope occurs throught the functions listed below.
*/
typedef struct gmx_envelope *gmx_envelope_t;

/* Width of Gaussian used to smooth the evelope over the space angle (in rad) */
// #define GMX_ENVELOPE_SMOOTH_SIGMA (15./180.*M_PI)
/* Our current smoothing sometimes leads to strange peaks in the envelope at present - use not smoothing as
   default until we fixed this */
#define GMX_ENVELOPE_SMOOTH_SIGMA (0./180.*M_PI)

/* With automatic determination of the number of recursions for envelope, use at least: */
#define WAXS_ENVELOPE_NREC_MIN 4

/* Setting up and destroying envelope *************************************** */
GMX_LIBGMX_EXPORT
gmx_envelope_t gmx_envelope_init(int n, gmx_bool bVerbose);

GMX_LIBGMX_EXPORT
void gmx_envelope_superimposeEnvelope(gmx_envelope_t e_add, gmx_envelope_t e_base);

GMX_LIBGMX_EXPORT
void gmx_envelope_buildSphere(gmx_envelope_t e, real rad);

GMX_LIBGMX_EXPORT
void gmx_envelope_buildEllipsoid(gmx_envelope_t e, rvec r);

GMX_LIBGMX_EXPORT
void gmx_envelope_buildEnvelope(gmx_envelope_t e, rvec x[], atom_id *index,
                                int isize, real d, real phiSmooth);
void gmx_envelope_buildEnvelope_omp(gmx_envelope_t env, rvec x[], atom_id *index,
                                int isize, real d, real phiSmooth, gmx_bool bSphere);
void
gmx_envelope_buildEnvelope_inclShifted_omp(gmx_envelope_t e, rvec x[], atom_id *index,
					   int isize, real dGiven, real phiSmooth, gmx_bool bSphere);
GMX_LIBGMX_EXPORT
void gmx_envelope_clearSurface(gmx_envelope_t e);

GMX_LIBGMX_EXPORT
void gmx_envelope_destroy(gmx_envelope_t e);

GMX_LIBGMX_EXPORT
gmx_envelope_t gmx_envelope_init_md(int nreq, t_commrec *cr, gmx_bool bVerbose);

GMX_LIBGMX_EXPORT
void gmx_envelope_bcast(gmx_envelope_t e, t_commrec *cr);

GMX_LIBGMX_EXPORT
void gmx_envelope_alloc_ftglob(gmx_envelope_t e, int nq);

GMX_LIBGMX_EXPORT
void gmx_envelope_free_ftglob(gmx_envelope_t e);

GMX_LIBGMX_EXPORT
void gmx_envelope_ftunit_srenew(gmx_envelope_t e, int nq);

GMX_LIBGMX_EXPORT
void gmx_envelope_ftdens_srenew(gmx_envelope_t e, int nq);

GMX_LIBGMX_EXPORT
void gmx_envelope_ftunit_sfree(gmx_envelope_t e);

GMX_LIBGMX_EXPORT
void gmx_envelope_ftdens_sfree(gmx_envelope_t e);

/* Write a VMD tcl script into envelope.tcl */
GMX_LIBGMX_EXPORT
void gmx_envelope_writeVMDCGO(gmx_envelope_t e, const char *fn, rvec rgb, real alpha);

/* Write a PyMOL CGO file into envelope.py */
GMX_LIBGMX_EXPORT
void gmx_envelope_writePymolCGO(gmx_envelope_t e, const char *fn, const char *name, rvec rgb, rvec rgb_inside, real alpha);
/* ************************************************************************ */


/* Using the envelope ****************************************************** */
GMX_LIBGMX_EXPORT
gmx_bool gmx_envelope_isInside(gmx_envelope_t e, const rvec x);

GMX_LIBGMX_EXPORT
void gmx_envelope_distToOuterInner(gmx_envelope_t e, const rvec x, real *distInner, real *distOuter);

GMX_LIBGMX_EXPORT
void gmx_envelope_minimumDistance(gmx_envelope_t e, const rvec x[], atom_id *index,
                                  int isize, real *mindist, int *imin, gmx_bool *bMinDistToOuter);

GMX_LIBGMX_EXPORT
gmx_bool gmx_envelope_bInsideCompactBox(gmx_envelope_t e, matrix Rinv, matrix box, rvec boxToXRef, t_pbc *pbc, gmx_bool bVerbose,
    real tolerance);

GMX_LIBGMX_EXPORT
void gmx_envelope_bounding_sphere(gmx_envelope_t e, rvec cent, real *R2);

GMX_LIBGMX_EXPORT
double gmx_envelope_diameter(gmx_envelope_t e);

GMX_LIBGMX_EXPORT
void gmx_envelope_center_xyz(gmx_envelope_t e, matrix Rinv, rvec cent);
/* ********************************************************************************** */


/* Compute Fourier Transform of the envlope ***************************************** */
GMX_LIBGMX_EXPORT
void gmx_envelope_unitFourierTransform(gmx_envelope_t e, rvec *q, int nq,
                                       real **ft_re, real **ft_im);

GMX_LIBGMX_EXPORT
gmx_bool gmx_envelope_bHaveUnitFT(gmx_envelope_t env);
/* ********************************************************************************** */


/* ********************************************************************************** */
/* Averaging density of solvation shell inside envelope and compute Fourier Transform */
GMX_LIBGMX_EXPORT
void gmx_envelope_solvent_density_nextFrame(gmx_envelope_t e,  double scale);

GMX_LIBGMX_EXPORT
void gmx_envelope_solvent_density_addAtom(gmx_envelope_t e, const rvec x, double nelec);

GMX_LIBGMX_EXPORT
void gmx_envelope_solvent_density_bcast(gmx_envelope_t e, t_commrec *cr);

GMX_LIBGMX_EXPORT
void gmx_envelope_solventFourierTransform(gmx_envelope_t e, rvec *q, int nq, gmx_bool bRecalcFT,
                                          real **ft_re, real **ft_im);

GMX_LIBGMX_EXPORT
gmx_bool gmx_envelope_bHaveSolventFT(gmx_envelope_t e);

GMX_LIBGMX_EXPORT
int gmx_envelope_getStepSolventDens(gmx_envelope_t e);

GMX_LIBGMX_EXPORT
void gmx_envelope_solvent_density_2pdb(gmx_envelope_t e, const char *fn);

GMX_LIBGMX_EXPORT
double gmx_envelope_solvent_density_getNelecTotal(gmx_envelope_t e);
/* ********************************************************************************** */


/* ************************************************************************ */

/* Envelope currently constructed? */
GMX_LIBGMX_EXPORT
gmx_bool gmx_envelope_bHaveSurf(gmx_envelope_t e);

/* Maximum radius of envelope */
GMX_LIBGMX_EXPORT
double gmx_envelope_maxR(gmx_envelope_t e);

/* Get volume of envelope */
GMX_LIBGMX_EXPORT
double gmx_envelope_getVolume(gmx_envelope_t e);

/* Slow method for computing the volume, introduced to test if volume is correct */
GMX_LIBGMX_EXPORT
void gmx_envelope_volume_montecarlo_omp(gmx_envelope_t e, double nTry_d, int seed0);

/* Return pointer to checksum string (is null-terminated) */
GMX_LIBGMX_EXPORT
char *gmx_envelope_chksum_str(gmx_envelope_t e);


/* ************************************************************************ */
/* Writing and reading evnelope file ************************************** */
GMX_LIBGMX_EXPORT
gmx_envelope_t gmx_envelope_readFromFile(const char * fn);

GMX_LIBGMX_EXPORT
void gmx_envelope_writeToFile(gmx_envelope_t e, const char * fn);
/* ************************************************************************ */


/* Functions to use the electron density on a regular grid (which can be visualized in VMD or so) */
void gmx_envelope_griddensity_nextFrame(gmx_envelope_t e, double scale);
void gmx_envelope_griddensity_addAtom(gmx_envelope_t e, const rvec x, const double nElec);
void gmx_envelope_griddensity_closeFrame(gmx_envelope_t e);

/* Write density as x/y/z density to a file */
void gmx_envelope_griddensity_write(gmx_envelope_t e, const char *fn);


#ifdef __cplusplus
}
#endif

#endif

#endif
