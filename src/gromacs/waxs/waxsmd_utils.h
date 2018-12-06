#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/futil.h"
#include "gromacs/math/vec.h"
#include "gromacs/waxs/waxsmd.h"
#include "gromacs/topology/idef.h"


#ifndef GMX_WAXSMD_UTILS_H
#define GMX_WAXSMD_UTILS_H

/* Return an estimate for the center and the radius of the bounding sphere
   around atoms x[]. Uses Ritter's bounding sphere (see Wikipedia) */
void
get_bounding_sphere(rvec x[], atom_id *index, int n, rvec cent, real *R, gmx_bool bVerbose);

void
mv_boxcenter_to_rvec(int natoms, rvec x[], rvec cent, matrix box);

void
mv_cog_to_rvec(int natoms, rvec x[], int *index, int nsolute, rvec target, rvec cog);

/* Get minimum distance of the solute to the box boundary.
   If mindistReturn != NULL, put the minimum distance into *mindistReturn
   Works with wr == NULL
*/
void
check_prot_box_distance(rvec x[], atom_id *index, int isize, matrix box,
                        t_waxsrec *wr, gmx_bool bPrintWarn, real *mindistReturn);

/* Do all the steps required on the solute frame before getting the solvation shell */
void
waxs_prepareSoluteFrame(t_waxsrec *wr, gmx_mtop_t *mtop, rvec x[], matrix box, int ePBC,
                        FILE *fpLog, matrix Rinv);

/* Move the pure-solvent frame onto the envelope, so we can use the pure-solvent frame
   to get the excluded solvent. On exit, ws->xPrepared is on the envelope. */
void
preparePureSolventFrame(t_waxs_solvent ws, int waxsStep, gmx_envelope_t env, int debugLvl);

/*Read an STX file name, and then copy only the relevant sections to xref.*/
void
read_fit_reference(const char* fn, rvec xref[], int nsys, atom_id* isol, int nsol, atom_id* ifit, int nfit);

/* Compute RDFs between the nTypes different atom types (nRDFs RDFs in total) */
void
atom_types_rdfs(int nFramesLocal, rvec *x[], matrix *box, int nTypes, int *isize, int *index[], double **rdf,
		int **irdf, int nR, real rmax, gmx_bool bVerbose);

/* Compute the sine transform of the RDF */
void
rdf_sine_transform(double *rdf, int nR, real rmax, real qmax, int nq, double *sinetransf);

void
interpolate_solvent_intensity(double *Igiven, real qmax, int nq, t_waxsrecType *wt);

void
read_pure_solvent_intensity_file(const char *fn, t_waxsrecType *wt);

void
do_pure_solvent_intensity(t_waxsrec *wr, t_commrec *cr, gmx_mtop_t *mtop, rvec **xSolv, int natoms, matrix *boxSolv, int nFramesRead,
                          int *sftypes, int nTypes, double qmax, int nq, double *intensitySum, char *fnIntensity, int t);

double
CMSF_q(t_cromer_mann cmsf, real q);

real
soluteRadiusOfGyration(t_waxsrec *wr, rvec x[]);

double
guinierFit(t_waxsrecType *wt, double *I, double *varI, double RgSolute);

/* Weighted average and stddev of x. Weights may be NULL */
void
average_stddev_d(double *x, int n, double *av, double *sigma, double *weights);

void
average_x2_d(double *x, int n, double *av, double *wptr);

void
average_xy_d(double *x, double *y, int n, double *av, double *wptr);

void
sum_squared_residual_d(double *x, double *y, int n, double *av, double *wptr);

/* Pearson correlation coeff. Note: avx, avy, sigx and sigy must be computed before. Weights may be NULL. */
double
pearson_d(int n, double *x, double *y, double avx, double avy, double sigx, double sigy, double *weights);

double
sum_array_d(int n, double *x);

void
nIndep_Shannon_Nyquist(t_waxsrec *wr, rvec x[], t_commrec *cr, gmx_bool bVerbose);

double*
make_nElecList(gmx_mtop_t *mtop);

void
turnDownPosresForceConst(t_commrec *cr, t_waxsrec *wr, double simtime, t_idef *idef);

/* Formatted writing to log file */
void
print2log(FILE *fp, const char *s, char *fmt, ...);

/*
 * Stuff for measuring computing time
 */

/* Start timing averaging after these steps */
#define WAXS_STEPS_RESET_TIME_AVERAGES 30

enum{
    /* These values must correspond to the names in const char* waxsTimingNames[waxsTimeNr]
       defined in waxsmd_utils.c */
    waxsTimeStep, waxsTimePrepareSolute, waxsTimePrepareSolvent, waxsTimeScattAmplitude,
    waxsTimeOneScattAmplitude, waxsTimedkI, waxsTimeFourier, waxsTimeGibbs, waxsTimePotForces, waxsTimeScattUpdates,
    waxsTimeComputeIdkI, waxsTimeSolvDensCorr,
    waxsTimeNr
};
extern const char* waxsTimingNames[waxsTimeNr];
enum{
    waxsTimingAction_start, waxsTimingAction_end, waxsTimingAction_add,
};

/* Init compute time measurements */
t_waxsTiming
waxsTimeInit();

/* Do a measurement: start, finish, or just add a time measured somewhere else (in seconds) */
void
waxsTimingDo(t_waxsrec *wr, int type, int action, double toStore, t_commrec *cr);

/* Get the last stored measurement, can be used to sum up different contributions */
double
waxsTimingGetLast(t_waxsTiming t, int type);

/* Write timing summary */
void
waxsTimingWrite(t_waxsTiming t, FILE *fp);

void
waxsTimingWriteStatus(t_waxsTiming t, FILE *fp);

void
waxsTimingWriteLast(t_waxsTiming t, FILE *fp);

#endif
