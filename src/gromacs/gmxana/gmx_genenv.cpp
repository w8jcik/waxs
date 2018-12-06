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
#include "gmxpre.h"

#include <cmath>
#include <cstring>

#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/fileio/gmxfio.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/commandline/pargs.h"
#include "gromacs/commandline/filenm.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/utility/futil.h"
#include "gromacs/fileio/pdbio.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/filetypes.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/topology/index.h"
#include "gromacs/math/vec.h"
#include "gromacs/fileio/xtcio.h"
#include "gromacs/math/do_fit.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/fileio/xdrf.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/waxs/gmx_envelope.h" //SWAX stuff
#include "gromacs/waxs/waxsmd.h" //SWAX stuff
#include "gromacs/waxs/waxsmd_utils.h" //SWAX stuff
#include "gromacs/utility/gmxomp.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/utility/cstringutil.h"


#define TIME_EXPLICIT 0
#define TIME_CONTINUE 1
#define TIME_LAST     2
#ifndef FLT_MAX
#define FLT_MAX 1e36
#endif
#define FLAGS (TRX_READ_X | TRX_READ_V | TRX_READ_F)


void
mv_bounding_sphere_to_origin(rvec x[], int nat, int nsolute, int *index)
{
    int i;
    rvec cent = {0, 0, 0};
    real R = 0;
    static int bFirst = TRUE;

    get_bounding_sphere(x, index, nsolute, cent, &R, bFirst);

    /* shift geomtric center to orgin */
    for (i = 0; i < nat; i++)
    {
        rvec_dec(x[i], cent);
    }
    bFirst = FALSE;
}

#define WAXS_WARN_BOX_DIST 0.5
static void solvent_density_scan(int nd, real dmin, real dmax,
                                 rvec *x_solute_allFrames, atom_id *index_solute_allFrames, int isize_solute_allFrames,
                                 atom_id *index_solute, int nsolute,
                                 rvec *xref, real *w_rls, int nfit, atom_id *ifit, gmx_bool bFit, gmx_bool bCentBoundingSphere,
                                 int nAtomsTotal,
                                 real sigma_smooth, t_trxstatus *status, output_env_t oenv, t_topology *top,
                                 const char *ndxfile, int nrec, const char *tprfile, const char *solvdensfile,
                                 int ePBC)
{
    int                    iEnv, *isolvent, nsolvent, j, natoms_tpr, nframes = 0, nWarn = 0, nlayers;
    gmx_envelope_t        *envelopes = 0;
    real                   dd, t, d, boxdist;
    rvec                  *xread = 0, xbefore, xdiff, origin, boxcenter;
    char                  *solventname = 0;
    const char            *warn;
    gmx_mtop_t             mtop;
    rvec                  *top_x;
    matrix                 box;
    t_tpxheader            tpx;
    int                    version, generation, iThread;
    double                *nElecList = 0, *nElecInLayer = 0, *nElecInLayer_sqr = 0, *nElec_loc = 0;
    double                 densLayer, densLayerErr, volInnerEnv, volOuterEnv, volLayer, nElecReduce;
    FILE                  *fp;
    const int              nWarnMax = 20;
    int                    nThreads = gmx_omp_get_max_threads();

    if (nd < 2)
    {
        gmx_fatal(FARGS, "Need at least 2 distnaces (option -nd)\n");
    }
    dd = (dmax-dmin)/(nd-1);
    nlayers = nd-1;
    snew(envelopes,        nd);
    snew(nElecInLayer,     nlayers);
    snew(nElecInLayer_sqr, nlayers);
    snew(nElec_loc,        nlayers*nThreads);

    printf("\nComputing solvent density around solute, building %d envelopes...\n", nd);

    printf("\nChoose a group for the solvent\n");
    get_index(&top->atoms, ndxfile, 1, &nsolvent, &isolvent, &solventname);

    for (iEnv = 0; iEnv<nd; iEnv++)
    {
        d = dmin + dd*iEnv;
        printf("CREATING ENVELOPE %d of %d -- d = %g\n", iEnv, nd, d);
        envelopes[iEnv] = gmx_envelope_init(nrec, TRUE);
        gmx_envelope_buildEnvelope_omp(envelopes[iEnv], x_solute_allFrames, index_solute_allFrames, isize_solute_allFrames,
                                       d, sigma_smooth, FALSE);
    }


    /* Read mtop from tpr file and make list of number of electrons for all atoms */
    read_tpxheader(tprfile, &tpx, TRUE, &version, &generation);
    snew(top_x, tpx.natoms);
    if (tpx.natoms != nAtomsTotal)
    {
        gmx_incons("Atom number mismatch in solvent_density_scan()\n");
    }
    read_tpx(tprfile, NULL, box, &natoms_tpr, top_x, NULL, NULL, &mtop);
    nElecList = make_nElecList(&mtop);

    /* Read first frame of the xtc file */
    snew(xread, nAtomsTotal);
    read_next_x(oenv, status, &t, nAtomsTotal, xread, box);

    clear_rvec(origin);

    do{
        /*
         * First of all, get the solvent around the protein.
         * So move the protein to the center of the box and put the solvent into the compact unitcell
         */

        /* 1) get box center */
        calc_box_center(ecenterTRIC, box, boxcenter);

        /* 2) move protein to box center */
        mv_cog_to_rvec(nAtomsTotal, xread, index_solute, nsolute, boxcenter, NULL);

        /* 3) put all atoms into the compact unit cell */
        if ( (warn = put_atoms_in_compact_unitcell(ePBC, ecenterTRIC, box, nAtomsTotal, xread)) != NULL)
        {
            gmx_fatal(FARGS, "Could not put all atoms into a compact unit cell representation."
                      "Warning was:\n%s\n", warn);
        }

        /* 4) Check if the protein has a reasonable distance from the box surface */
        check_prot_box_distance(xread, index_solute, nsolute, box, NULL, FALSE, &boxdist);
        if (boxdist < WAXS_WARN_BOX_DIST && nWarn < nWarnMax)
        {
            printf("\nWARNING, solute getting close to the box surface (%g nm). Your solute is probably not whole. Fix this with trjconv,\n"
                   "e.g. with options -pbc mol or -pbc nojump or -pbc cluster before calling g_genenv.\n", boxdist);
            nWarn++;
        }

        if (bFit)
        {
            copy_rvec(xread[0], xbefore);
            /* shift to zero (required for do_fit) */
            reset_x(nfit, ifit, nAtomsTotal, NULL, xread, w_rls);
            rvec_sub(xbefore, xread[0], xdiff);
            do_fit(nAtomsTotal, w_rls, xref, xread);
            for (j = 0; j < nAtomsTotal; j++)
            {
                /* shift back */
                rvec_inc(xread[j], xdiff);
            }
        }

        if (bCentBoundingSphere)
        {
            /* This option is deprecated - we now always shift the COG to the origin */
            mv_bounding_sphere_to_origin(xread, nAtomsTotal, nsolute, index_solute);
        }
        else
        {
            mv_cog_to_rvec(nAtomsTotal, xread, index_solute, nsolute, origin, NULL);
        }

#pragma omp parallel shared(xread, isolvent, envelopes, nElec_loc, nElecList, nd) private(iEnv)
        {
            int threadID = gmx_omp_get_thread_num();
            int iEnvMax, iEnvMin, iEnvTest;

            /* Clear nElec_loc elements used in this thread */
            for (iEnv = 0; iEnv < (nd-1); iEnv++)
            {
                nElec_loc[threadID*nlayers + iEnv] = 0;
            }

#pragma omp for
            for (j = 0; j<nsolvent; j++)
            {
                /* First check if atom is really between the largest and the smallest envelope */
                if (!gmx_envelope_isInside(envelopes[nd-1], xread[isolvent[j]]) ||
                    gmx_envelope_isInside(envelopes[0   ], xread[isolvent[j]]) )
                {
                    continue;
                }

                iEnvMax = nd-1;
                iEnvMin = 0;

                while (iEnvMax-iEnvMin > 1)
                {
                    /* Test if atom is inside the envelpe in the middle of iEnvMax and iEnvMin */
                    iEnvTest = (iEnvMax+iEnvMin)/2;
                    if (gmx_envelope_isInside(envelopes[iEnvTest], xread[isolvent[j]]))
                    {
                        iEnvMax = iEnvTest;
                    }
                    else
                    {
                        iEnvMin = iEnvTest;
                    }
                }

                /* Make sure that every thread writes into different elements */
                nElec_loc[threadID*nlayers + iEnvMin] += nElecList[isolvent[j]];
            }
        }

        for (iEnv = 0; iEnv < (nd-1); iEnv++)
        {
            /* Sum over the threads (reduce array) */
            nElecReduce = 0;
            for (iThread = 0; iThread < nThreads; iThread++)
            {
                nElecReduce += nElec_loc[iThread*nlayers + iEnv];
            }

            /* Keep sum and sum^2 to compute errors below */
            nElecInLayer    [iEnv] += nElecReduce;
            nElecInLayer_sqr[iEnv] += dsqr(nElecReduce);
        }

        nframes++;
    } while (read_next_x(oenv, status, &t, nAtomsTotal, xread, box));

    fp = xvgropen(solvdensfile, "Solvent density", "R (nm)", "density (e/nm\\S3\\N)", oenv);
    fprintf(fp, "@TYPE xydy\n");
    for (iEnv = 0; iEnv < (nd-1); iEnv++)
    {
        nElecInLayer    [iEnv] /= nframes;
        nElecInLayer_sqr[iEnv] /= nframes;

        volInnerEnv = gmx_envelope_getVolume(envelopes[iEnv  ]);
        volOuterEnv = gmx_envelope_getVolume(envelopes[iEnv+1]);
        volLayer    = volOuterEnv - volInnerEnv;

        densLayer    = nElecInLayer[iEnv]/volLayer;
        densLayerErr = sqrt(nElecInLayer_sqr[iEnv]-dsqr(nElecInLayer[iEnv]))/sqrt(nframes)/volLayer;

        fprintf(fp, "%10g %10g %10g\n", dmin + dd*(iEnv+0.5), densLayer, densLayerErr);
    }
    ffclose(fp);

    printf("\nComputed solvent density from %d frames\n", nframes);
    printf("\nWrote solvent density to %s\n", solvdensfile);
}

static void
write_viewable_envelope(gmx_envelope_t e, const char *name, rvec rgb, rvec rgb_inside, real alpha, gmx_bool bVMD)
{
    char *outfn, *fn_root;
    int filelen;

    filelen = strlen(name);
    snew(outfn,   filelen + 10);
    snew(fn_root, filelen + 10);

    strncpy(fn_root, name, filelen-4);
    fn_root[filelen-4] = '\0';

    if ( bVMD )
    {
        sprintf(outfn, "%s.tcl", fn_root);
        gmx_envelope_writeVMDCGO(e, outfn, rgb, alpha);
        /*printf("Wrote VMD-tcl file of envelope to %s\n", outfn);*/
    }
    else
    {
        sprintf(outfn, "%s.py", fn_root);
        gmx_envelope_writePymolCGO(e, outfn, name, rgb, rgb_inside, alpha);
        /*printf("Wrote PyMol CGO file of envelope to %s\n", outfn);*/
    }

    sfree(outfn);
    sfree(fn_root);
}


void
get_good_pbc_atom(rvec x[], int nsolute, int *index, int *iPBCsolute, real *rpbc, real *Rbsphere)
{
    int i, imin = -1;
    rvec diff, cent = {0, 0, 0};
    real R, r2, rmin2 = 1e20;

    get_bounding_sphere(x, index, nsolute, cent, &R, FALSE);

    /* Get a good PBC atom (solute-internal numbering)
       Note that this solute-internal numbering may differ from the global atom number if, e.g.,
       virtual sites are present */
    for (i = 0; i < nsolute; i++)
    {
        rvec_sub(x[index[i]], cent, diff);
        r2 = norm2(diff);
        if (r2 < rmin2)
        {
            rmin2 = r2;
            imin  = i;
        }
    }
    *iPBCsolute = imin;
    *rpbc       = sqrt(rmin2);
    *Rbsphere   = R;
}

static void
env_writeFourier(t_spherical_map *qvecs, real *ft_re, real *ft_im,
                 const char *ftfile, const char *ftabsfile, const char *intfile, output_env_t oenv)
{
    int i, j;
    real av_re, av_im;
    FILE *out;

    if (ftfile)
    {
        out = ffopen(ftfile, "w");
        fprintf(out, "#      %-10s %-12s %-12s   %-12s %-12s\n", "qx", "qy", "qz", "Re(FT)", "Im(FT)");
        for (i = 0; i < qvecs->n; i++)
        {
            fprintf(out, "%12g %12g %12g   %12g %12g\n", qvecs->q[i][XX], qvecs->q[i][YY], qvecs->q[i][ZZ],
                    ft_re[i], ft_im[i]);
        }
        ffclose(out);
        printf("Wrote Fourier transform of envelope to %s\n", ftfile);
    }
    if (ftabsfile)
    {
        out = xvgropen(ftabsfile, "FT of envelope (rotationally averaged)",
                       "q (1/nm)", "FT ", oenv);
        for (i = 0; i < qvecs->nabs; i++)
        {
            av_re = 0.;
            av_im = 0.;
            for (j = qvecs->ind[i]; j < qvecs->ind[i+1]; j++)
            {
                av_re += ft_re[j];
                av_im += ft_im[j];
            }
            av_re /= qvecs->ind[i+1]-qvecs->ind[i];
            av_im /= qvecs->ind[i+1]-qvecs->ind[i];
            fprintf(out, "%12g %12g %12g\n", qvecs->abs[i], av_re, av_im);
        }
        printf("Wrotes spherically averaged Fourier transform of envelope to %s\n", ftabsfile);
    }
    if (intfile)
    {
        out = xvgropen(intfile, "|FT|\\S2\\N (rotaionally averaged)",
                       "q (1/nm)", "Intensity", oenv);
        for (i = 0; i < qvecs->nabs; i++)
        {
            av_re = 0.;
            for (j = qvecs->ind[i]; j < qvecs->ind[i+1]; j++)
            {
                av_re += sqr(ft_re[j]) + sqr(ft_im[j]);
            }
            av_re /= qvecs->ind[i+1]-qvecs->ind[i];
            fprintf(out, "%12g %12g\n", qvecs->abs[i], av_re);
        }
        printf("Wrotes spherically averaged scattering intensity of envelope to %s\n", intfile);
    }
}


int gmx_genenv(int argc, char *argv[])
{
    const char
        *desc[] =
        {
            "[TT]g_genenv[tt] writes an envelope file containing the radii",
            "of points on an icosphere, surrounding all atoms at all points",
            "in the given trajectory.[PAR]",
            "This tool by default fits the given trajectory around the first frame,",
            "and the resulting envelope around the origin. Options [TT]-fittpr[tt]",
            "can be used to alter this to fit to the tpr file. This requires that ",
            "WAXSMD runs and reruns to also use those coordinates as a reference fit.",
            "[PAR]Two main methods are offered to create a common envelope to cover",
            "multiple simulation systems are may be difficult to combine:",
            "[PAR]- the argument [TT]-sphere[tt] replaces all rays with the maximum value",
            "determined by the trajectory and [TT]-d[tt]. Adding [TT]-d_sphere[tt]",
            "will skip envelope calculations and simply print the sphere.",
            "[PAR]- loading an additional envelope in [TT]-e1[tt] will superimpose this",
            "onto the constructed one. Giving two envelope files in [TT]-e1[tt] and [TT]-e2[tt]",
            "will skip envelope calculations and simply combine them."
        };

    static real d = 0.7, d_sphere = -1., dmin = 0.3, dmax = 2.5;
    static gmx_bool bFit = TRUE, bFitTPR = FALSE, bOrig = TRUE, bPBC = TRUE, bCentBoundingSphere = FALSE;
    static gmx_bool bCGO = TRUE, bVMDout = FALSE, bSphere = FALSE, bCheckEnvInBox = FALSE;
    static int nrec = 4, J = 200, nq = 50, nthreads = 1;
    static real sigma_smooth = GMX_ENVELOPE_SMOOTH_SIGMA, qmax = 30., alpha = 0.5, nTry_vol_mc = 0;
    static rvec rgb        = {0.0, 0.644531, 1.0};   /* marine blue */
    static rvec rgb_inside = {1.0, 0.746094, 0.0,};  /* orange      */
    static rvec r_elipsoid = {0, 0, 0};
    static int seed = -1, nd_scan = 40, nfrEnvMax = -1;

    t_pargs
        pa[] =
        {
            { "-nt", FALSE, etINT, {&nthreads},
              "Number of threads used by g_genenv"},
            { "-d", FALSE, etREAL,
              { &d }, "Distance to surface" },
            { "-nrec", FALSE, etINT,
              { &nrec }, "# of rerusions for icosphere (20 * 4^N faces). <=0 means automatic determination." },
            { "-sig", FALSE, etREAL,
              { &sigma_smooth }, "Angular sigma to smooth the envelope (rad)" },
            { "-fit", FALSE, etBOOL,
              { &bFit }, "Fit frames, by default to the first frame read (not to tpr/pdb)." },
            { "-fittpr", FALSE, etBOOL,
              { &bFitTPR }, "Fit frames to the TPR file instead." },
            { "-orig", FALSE, etBOOL,
              { &bOrig }, "Move COG of solute atoms to origin before building envelope" },
            { "-pbc", FALSE, etBOOL,
              { &bPBC }, "PBC check" },
            { "-nq", FALSE, etINT,
              { &nq }, "for Fourier tranform: # of abolute |q|" },
            { "-qmax", FALSE, etREAL,
              { &qmax }, "for Fourier tranform: maximum |q| (1/nm)" },
            { "-J", FALSE, etINT,
              { &J }, "for Fourier tranform: # of q vectors per |q|" },
            { "-centbs", FALSE, etBOOL,
              { &bCentBoundingSphere }, "Move bounding sphere to origin (instead of center of geometry)" },
            { "-sphere", FALSE, etBOOL,
              { &bSphere }, "Make spherical envelope" },
            { "-d_sphere", FALSE, etREAL,
              { &d_sphere }, "Write a sphere of this size (nm), and skip all calculations."},
            { "-elipsoid", FALSE, etRVEC,
              { &r_elipsoid }, "HIDDENBuild elipsoid with those half-axes."},
            { "-cgo", FALSE, etBOOL,
              { &bCGO }, "Also write a CGO file for visualisation." },
            { "-vmdout", FALSE, etBOOL,
              { &bVMDout }, "write visualisation as a VMD-tcl file instead of python" },
            { "-rgb", FALSE, etRVEC,
              { &rgb }, "RGB color of envelope" },
            { "-alpha", FALSE, etREAL,
              { &alpha }, "Transparency of envelope" },
            { "-inbox", FALSE, etBOOL,
              { &bCheckEnvInBox }, "Write warnings if the envelope does not fit into any simulation frame boxes" },
            { "-vol-mc", FALSE, etREAL,
              { &nTry_vol_mc }, "Write volume of envelope, computed by Monte Carlo."},
            { "-seed", FALSE, etINT,
              { &seed }, "Monte Carlo seed (-1 = generate seed)."},
            { "-dmin", FALSE, etREAL,
              { &dmin }, "Minimum distance for solvent density scan."},
            { "-dmax", FALSE, etREAL,
              { &dmax }, "Maximum distance for solvent density scan."},
            { "-nd", FALSE, etINT,
              { &nd_scan }, "Number of distance for solvent density scan."},
            { "-nfrEnvMax", FALSE, etINT,
              { &nfrEnvMax }, "Maximum number of frames used for envelope (useful with -od)."},
        };
#define npargs asize(pa)

    int             nfit, i, nat, *index_all = 0, nframes, j, ePBC, nsolute, filelen, iPBCsolute;
    atom_id        *index = 0, *ifit = 0;
    rvec *x = NULL, *xref = NULL, xbefore, xdiff, *xtps = 0, *xread = 0;
    t_atoms *atoms, useatoms;
    real *w_rls = NULL, t, av_re, av_im, rPBCatom, *ft_re = 0, *ft_im = 0, Rbsphere = 0, area1, l1;
    gmx_bool bTop;
    gmx_envelope_t envelope=NULL, e1=NULL, e2=NULL;
    char buf[STRLEN], envfile_cgo[STRLEN], title[256], reftitle[256];
    output_env_t    oenv;
    const char     *envfile, *outreffile, *trxfile, *ndxfile, *allfrfile, *ftfile, *ftabsfile, *intfile;
    const char     *inenvfile1, *inenvfile2, *inreffile, *solvdensfile;
    t_trxstatus    *status;
    t_topology      top;
    matrix box;
    char *fitname, *solutename;
    gmx_conect       gc    = NULL;
    FILE            *out    = NULL;
    gmx_rmpbc_t  gpbc = NULL;
    t_spherical_map *qvecs;
    rvec origin;

    t_filenm fnm[] = {
        { efTPS, "-s", NULL,         ffREAD },
        { efTRX, "-f", NULL,         ffREAD },
        { efNDX, "-n", "index",      ffOPTRD },
        { efSTO, "-r", "fit",        ffOPTRD },
        { efDAT, "-o", "envelope",   ffWRITE },
        { efSTO, "-or", "envelope-ref", ffWRITE },
        { efPDB, "-of", "allframes", ffOPTWR },
        { efDAT, "-e1", "envelope1", ffOPTRD },
        { efDAT, "-e2", "envelope2", ffOPTRD },
        { efDAT, "-ft",  "fourier",   ffOPTWR },
        { efDAT, "-fta", "fourier_qabs",   ffOPTWR },
        { efDAT, "-int", "intensity",   ffOPTWR },
        { efXVG, "-od", "solvdens",   ffOPTWR },
    };

#define NFILE asize(fnm)

    CopyRight(stderr, argv[0]);
    parse_common_args(&argc, argv, PCA_BE_NICE | PCA_CAN_TIME, NFILE, fnm,
                      asize(pa), pa, asize(desc), desc, 0, NULL, &oenv);

    ndxfile    = ftp2fn_null(efNDX, NFILE, fnm);
    trxfile    = ftp2fn(efTRX, NFILE, fnm);
    envfile    = opt2fn("-o", NFILE, fnm);
    inreffile  = opt2fn_null("-r", NFILE, fnm);
    outreffile = opt2fn("-or", NFILE, fnm);
    allfrfile  = ftp2fn_null(efPDB, NFILE, fnm);
    inenvfile1 = opt2fn_null("-e1", NFILE, fnm);
    inenvfile2 = opt2fn_null("-e2", NFILE, fnm);
    ftfile     = opt2fn_null("-ft", NFILE, fnm);
    ftabsfile  = opt2fn_null("-fta", NFILE, fnm);
    intfile    = opt2fn_null("-int", NFILE, fnm);
    solvdensfile = opt2fn_null("-od", NFILE, fnm);

    /*Check if we are just computing envelopes without using any coordinate information.
     * A premature return is needed to not use any TPR file information?  */

    /* Combining two envelopes */
    if ( inenvfile1 )
    {
        e1 = gmx_envelope_readFromFile(inenvfile1);
    }

    if ( inenvfile1 && inenvfile2 )
    {
        e2 = gmx_envelope_readFromFile(inenvfile2);

        gmx_envelope_superimposeEnvelope(e1,e2);

        /* Write envelope */
        gmx_envelope_writeToFile(e2, envfile);
        printf("Wrote envelope to %s\n", envfile);

        if ( bCGO )
        {
            write_viewable_envelope( e2, envfile, rgb, rgb_inside, alpha, bVMDout);
        }
        if (ftfile || ftabsfile || intfile)
        {
            qvecs = gen_qvecs_map( 0., qmax, nq, J, FALSE, NULL, ewaxsanisoNO, 0., NULL, 0, 0., FALSE);
            gmx_envelope_unitFourierTransform(e2, qvecs->q, qvecs->n, &ft_re, &ft_im);
            env_writeFourier(qvecs, ft_re, ft_im, ftfile, ftabsfile, intfile, oenv);
        }

        thanx(stderr);
        return 0;
    }

    /* Hack a sphere of a specific dimension. */
    if ( bSphere && d_sphere > 0)
    {
        if (nrec <= 0)
        {
            gmx_fatal(FARGS, "Automatic determination of number of recursions (-nrec) not available for spherical envelope\n");
        }
        envelope = gmx_envelope_init(nrec, TRUE);
        gmx_envelope_buildSphere(envelope, d_sphere);
        /* Write envelope */
        gmx_envelope_writeToFile(envelope, envfile);
        printf("Wrote envelope to %s\n", envfile);

        if ( bCGO )
        {
            write_viewable_envelope( envelope, envfile, rgb, rgb_inside, alpha, bVMDout);
        }
        if (ftfile || ftabsfile || intfile)
        {
            qvecs = gen_qvecs_map( 0., qmax, nq, J, FALSE, NULL, ewaxsanisoNO, 0., NULL, 0, 0., FALSE);
            gmx_envelope_unitFourierTransform(envelope, qvecs->q, qvecs->n, &ft_re, &ft_im);
            env_writeFourier(qvecs, ft_re, ft_im, ftfile, ftabsfile, intfile, oenv);
        }

        thanx(stderr);
        return 0;
    }

    if (r_elipsoid[XX] > 0)
    {
        if (nrec < 0)
        {
            gmx_fatal(FARGS, "Automatic determination of number of recursions (-nrec) not available for spherical envelope\n");
        }
        envelope = gmx_envelope_init(nrec, TRUE);
        gmx_envelope_buildEllipsoid(envelope, r_elipsoid);
        gmx_envelope_writeToFile(envelope, envfile);
        //gmx_fatal(FARGS, "gmx_envelope_buildElipsoid() mising\n");

        if (ftfile || ftabsfile || intfile)
        {
            qvecs = gen_qvecs_map( 0., qmax, nq, J, FALSE, NULL, ewaxsanisoNO, 0., NULL, 0, 0., FALSE);
            gmx_envelope_unitFourierTransform(envelope, qvecs->q, qvecs->n, &ft_re, &ft_im);
            env_writeFourier(qvecs, ft_re, ft_im, ftfile, ftabsfile, intfile, oenv);
        }
        if ( bCGO )
        {
            write_viewable_envelope( envelope, envfile, rgb, rgb_inside, alpha, bVMDout);
        }

        thanx(stderr);
        return 0;
    }

    bTop = read_tps_conf(ftp2fn(efTPS, NFILE, fnm), buf, &top, &ePBC, &xtps,
                         NULL, box, TRUE);
    atoms = &top.atoms;

    if ( (bFitTPR || inreffile) && !bFit)
    {
        bFit = TRUE;
    }
    if (bFitTPR && inreffile)
    {
        gmx_fatal(FARGS,"Not possible to fit to the TPR and a given reference file at the same time!\n");
    }


    if (bFit)
    {
        printf("\nChoose a group for the least squares fit\n");
        get_index(atoms, ndxfile, 1, &nfit, &ifit, &fitname);
        if (nfit < 3)
        {
            gmx_fatal(FARGS, "Need >= 3 points to fit!\n");
        }

        /* Read TPR coordinates instead of first-frame coordinates. */
        if (bFitTPR)
        {
            snew(xref, atoms->nr);
            for (i=0; i<atoms->nr; i++)
            {
                copy_rvec(xtps[i], xref[i]);
            }
        }

        /* Doing a non-weighted fit */
        snew(w_rls, atoms->nr);
        for (i = 0; (i < nfit); i++)
        {
            w_rls[ifit[i]] = 1.;
        }
    }
    else
    {
        nfit = 0;
    }

    printf("\nChoose the solute group\n");
    get_index(atoms, ndxfile, 1, &nsolute, &index, &solutename);

    nat = read_first_x(oenv, &status, trxfile, &t, &xread, box);
    if (nat != top.atoms.nr)
    {
        gmx_fatal(FARGS, "\nTopology/PDB has %d atoms, whereas trajectory has %d\n",
                  top.atoms.nr, nat);
    }

    if (bPBC)
    {
        gpbc = gmx_rmpbc_init(&top.idef, ePBC, top.atoms.nr, box);
        gmx_rmpbc(gpbc, top.atoms.nr, box, xread);
    }

    if (bFit)
    {
        snew(xref, nat);
        for (i=0; i<nat; i++)
        {
            clear_rvec(xref[i]);
        }

        /* Read TPR coordinates instead of first-frame coordinates. */
        if (bFitTPR)
        {
            for (i=0; i<atoms->nr; i++)
            {
                copy_rvec(xtps[i], xref[i]);
            }
        }
        /* Take from the input reference file. */
        else if (inreffile)
        {
            read_fit_reference(inreffile, xref, nat, index, nsolute, ifit, nfit);
        }
        /* Take from the first frame. */
        else
        {
            for (i=0; i<nat; i++)
            {
                copy_rvec(xread[i], xref[i]);
            }
        }
        reset_x(nfit, ifit, nat, NULL, xref, w_rls);
    }

    if (allfrfile)
    {
        init_t_atoms(&useatoms, atoms->nr, FALSE);
        sfree(useatoms.resinfo);
        useatoms.resinfo = atoms->resinfo;
        for (i = 0; (i < nsolute); i++)
        {
            useatoms.atomname[i] = atoms->atomname[index[i]];
            useatoms.atom[i]     = atoms->atom[index[i]];
            useatoms.nres        = max(useatoms.nres, useatoms.atom[i].resind+1);
        }
        useatoms.nr = nsolute;

        out = ffopen(allfrfile, "w");
    }

    j = 0;
    clear_rvec(origin);
    do
    {
        if (bPBC)
        {
            gmx_rmpbc(gpbc, top.atoms.nr, box, xread);
        }

        if (bFit)
        {
            copy_rvec(xread[0], xbefore);
            /* shift to zero (required for do_fit) */
            reset_x(nfit, ifit, nat, NULL, xread, w_rls);
            rvec_sub(xbefore, xread[0], xdiff);
            do_fit(nat, w_rls, xref, xread);
            for (i=0; i<nat; i++)
            {
                /* shift back */
                rvec_inc(xread[i], xdiff);
            }
        }

        if (j == 0)
        {
            /* In the first frame, get a good PBC atom that is close to the center
               of the bounding sphere*/
            get_good_pbc_atom(xread, nsolute, index, &iPBCsolute, &rPBCatom, &Rbsphere);
            printf("\n############ G O O D   P B C   A T O M ################################\n"
                   "N.B.: Solute atom number %d is near the center of the bounding sphere -\n"
                   "      it would make a good waxs-pbc atom (distance  = %g)\n"
                   "      Global atom number = %d (name %s, residue %s-%d)\n"
                   "#######################################################################\n\n",
                   iPBCsolute+1, rPBCatom, index[iPBCsolute]+1, *(atoms->atomname[index[iPBCsolute]]),
                   *(atoms->resinfo[atoms->atom[index[iPBCsolute]].resind].name),
                   atoms->resinfo[atoms->atom[index[iPBCsolute]].resind].nr);
        }

        if (bOrig)
        {
            if (bCentBoundingSphere)
            {
                /* This option is deprecated - we now always shift the COG to the origin */
                mv_bounding_sphere_to_origin(xread, nat, nsolute, index);
            }
            else
            {
                mv_cog_to_rvec(nat, xread, index, nsolute, origin, NULL);
            }
        }

        /* Now store solute coordinates in a single long array */
        srenew(x, (j+1)*nsolute);
        for (i=0; i<nsolute; i++)
        {
            copy_rvec(xread[index[i]], x[j*nsolute+i]);
        }

        if (allfrfile)
        {
            fprintf(out, "REMARK    GENERATED BY G_GENENV\n");
            sprintf(title, "frame %d", j+1);
            write_pdbfile(out, title, &useatoms, x + j*nsolute,
                          ePBC, box, ' ', j+1, gc, TRUE);
        }

        j++;
    }
    while (read_next_x(oenv, status, &t, nat, xread, box) && (nfrEnvMax<0 || j<nfrEnvMax));
    nframes = j;

    if (out)
    {
        ffclose(out);
    }

    printf("Read %d frames from %s\n", j, trxfile);

    snew(index_all, nframes*nsolute);
    for (i=0; i< nframes*nsolute; i++)
    {
        index_all[i] = i;
    }

    /* Automatic determination of number of recursions */
    if (nrec <= 0)
    {
        nrec = WAXS_ENVELOPE_NREC_MIN;
        while (TRUE)
        {
            /* get length of side of face triangle */
            area1 = 4*M_PI*Rbsphere*Rbsphere/(20*pow(4.0, nrec));
            l1    = sqrt(4*area1/sqrt(3.0));
            if (l1 < d/2)
            {
                printf("\nAutomatic determination of number of recursions of icosphere:\n"
                       "\tR bounding sphere = %g\n"
                       "\tnrec              = %d\n"
                       "\tnumber of faces   = %d\n"
                       "\tface side length  = %g\n\n",
                       Rbsphere, nrec, (int) (20*pow(4, nrec)), l1);
                break;
            }
            nrec ++;
        }
    }

    /* Set # of OpenMP threads */
    gmx_omp_set_num_threads(nthreads);
    printf("\nNote: Will use %d OpenMP threads.\n", gmx_omp_get_max_threads());

    /* Finally, build the envelope - make sure all the output is file, so we can grep it */
    fflush(stdout);
    fflush(stderr);
    envelope = gmx_envelope_init(nrec, TRUE);
    gmx_envelope_buildEnvelope_omp(envelope, x, index_all,  nframes*nsolute, d,
                                   sigma_smooth, bSphere);
    fflush(stdout);
    fflush(stderr);

    /* Superimpose additional envelope if loaded */
    if ( inenvfile1 )
    {
        printf("\nSuper-imposing %s onto constructed envelope.\n", inenvfile1 );
        gmx_envelope_superimposeEnvelope(e1,envelope);
    }

    /* Writing envelope to pymol cgo file */
    if ( bCGO )
    {
        write_viewable_envelope( envelope, envfile, rgb, rgb_inside, alpha, bVMDout);
    }

    /* Write envelope */
    gmx_envelope_writeToFile(envelope, envfile);
    printf("Wrote envelope to %s\n", envfile);

    /* Write reference coordinates */
    sprintf(reftitle, "Reference coordinates for envelope file %s", envfile);
    write_sto_conf_indexed(outreffile, reftitle, atoms, xref, NULL, ePBC, box, nfit, ifit);

    /* Do Fourier transform */
    if (ftfile || ftabsfile || intfile)
    {
        qvecs = gen_qvecs_map( 0., qmax, nq, J, FALSE, NULL, ewaxsanisoNO, 0., NULL, 0, 0., FALSE);
        gmx_envelope_unitFourierTransform(envelope, qvecs->q, qvecs->n, &ft_re, &ft_im);
        env_writeFourier(qvecs, ft_re, ft_im, ftfile, ftabsfile, intfile, oenv);
    }




    /* For each frame in xtc, check if the enelope fits into the compact box */
    if (bCheckEnvInBox)
    {
        matrix Rinv = { {1,0,0}, {0,1,0}, {0,0,1}};
        rvec   boxcenter, env_cent, protRefInBox;
        int    nFrOutside = 0;
        t_pbc  pbc;
        real   env_R2;

        rewind_trj(status);
        read_next_x(oenv, status, &t, nat, xread, box);

        do
        {
            if (gpbc)
            {
                set_pbc(&pbc, ePBC, box);
            }

            /* Find the position of the envelope center in the box */
            calc_box_center(ecenterTRIC, box, boxcenter);
            if (TRICLINIC(box))
            {
                /* Triclinic box: we put the center of the bounding sphere to the center of the box */
                gmx_envelope_bounding_sphere(envelope, env_cent, &env_R2);
            }
            else
            {
                /* Cuboid box: we put the center in x/y/z to the center of the box */
                gmx_envelope_center_xyz(envelope, Rinv, env_cent);
            }
            rvec_sub(boxcenter, env_cent, protRefInBox);

            /* Check if in box, and write pymol cgo of vertices outside for the first frame */
            if (! gmx_envelope_bInsideCompactBox(envelope, Rinv, box, protRefInBox, &pbc, nFrOutside == 0, 0.1))
            {
                printf("WARNING, xtc time %10g : constructed envelope does not fit into the compact box (with tolerance 0.1nm)\n", t);
                nFrOutside++;
            }
        } while(read_next_x(oenv, status, &t, nat, xread, box));
        printf("\nNumber of frames for which the envelope does not fit into the compact box: %d of %d\n", nFrOutside, nframes);
    }

    if (nTry_vol_mc > 0)
    {
        gmx_envelope_volume_montecarlo_omp(envelope, nTry_vol_mc, seed);
    }

    if (solvdensfile)
    {
        if (!bTop)
        {
            gmx_fatal(FARGS, "Need a tpr file for computing the solvent density (option -od).\n");
        }

        rewind_trj(status);
        solvent_density_scan(nd_scan, dmin, dmax,
                             x, index_all, nframes*nsolute,
                             index, nsolute,
                             xref, w_rls, nfit, ifit, bFit, bCentBoundingSphere,
                             nat, sigma_smooth, status, oenv, &top,
                             ndxfile, nrec, ftp2fn(efTPS, NFILE, fnm), solvdensfile, ePBC);
    }

    thanx(stderr);

    return 0;
}
