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
 * Green Red Orange Magenta Azure Cyan Skyblue
 */
#include "gmxpre.h"

#include <math.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include "pdbio.h"
#include "confio.h"
#include "symtab.h"
#include "smalloc.h"
#include "macros.h"
#include "copyrite.h"
#include "statutil.h"
#include "string2.h"
#include "strdb.h"
#include "index.h"
#include "vec.h"
#include "typedefs.h"
#include "gbutil.h"
#include "strdb.h"
#include "physics.h"
#include "atomprop.h"
#include "tpxio.h"
#include "pbc.h"
#include "princ.h"
#include "txtdump.h"
#include "viewit.h"
#include "rmpbc.h"
#include "gmx_ana.h"

/* Portable version of ctime_r implemented in src/gmxlib/string2.c, but we do not want it declared in public installed headers */
char *
gmx_ctime_r(const time_t *clock, char *buf, int n);

typedef char charElem[10];

typedef struct
{
    real maxmassdiff;
    gmx_bool bGromos;
    gmx_bool bVsites;
} t_genscatt_opt;

struct{ char el[3]; real m; } elemmass[] =
{
    {"H", 1.0079},{"He", 4.0026},{"Li", 6.941},{"Be", 9.0122},{"B", 10.811},{"C", 12.0107},
    {"N", 14.0067},{"O", 15.9994},{"F", 18.9984},{"Ne", 20.1797},{"Na", 22.9897},{"Mg", 24.305},
    {"Al", 26.9815},{"Si", 28.0855},{"P", 30.9738},{"S", 32.065},{"Cl", 35.453},{"Ar", 39.948},
    {"K", 39.0983},{"Ca", 40.078},{"Sc", 44.9559},{"Ti", 47.867},{"V", 50.9415},{"Cr", 51.9961},
    {"Mn", 54.938},{"Fe", 55.845},{"Co", 58.9332},{"Ni", 58.6934},{"Cu", 63.546},{"Zn", 65.39},
    {"Ga", 69.723},{"Ge", 72.64},{"As", 74.9216},{"Se", 78.96},{"Br", 79.904},{"Kr", 83.8},
    {"Rb", 85.4678},{"Sr", 87.62},{"Y", 88.9059},{"Zr", 91.224},{"Nb", 92.9064},{"Mo", 95.94},
    {"Tc", 98},{"Ru", 101.07},{"Rh", 102.906},{"Pd", 106.42},{"Ag", 107.868},{"Cd", 112.411},
    {"In", 114.818},{"Sn", 118.71},{"Sb", 121.76},{"Te", 127.6},{"I", 126.904},{"Xe", 131.293},
    {"Cs", 132.905},{"Ba", 137.327},{"La", 138.905},{"Ce", 140.116},{"Pr", 140.908},{"Nd", 144.24},
    {"Pm", 145},{"Sm", 150.36},{"Eu", 151.964},{"Gd", 157.25},{"Tb", 158.925},{"Dy", 162.5},
    {"Ho", 164.93},{"Er", 167.259},{"Tm", 168.934},{"Yb", 173.04},{"Lu", 174.967},{"Hf", 178.49},
    {"Ta", 180.948},{"W", 183.84},{"Re", 186.207},{"Os", 190.23},{"Ir", 192.217},{"Pt", 195.078},
    {"Au", 196.966},{"Hg", 200.59},{"Tl", 204.383},{"Pb", 207.2},{"Bi", 208.98},{"Po", 209},
    {"At", 210},{"Rn", 222},{"Fr", 223},{"Ra", 226},{"Ac", 227},{"Th", 232.038},
    {"Pa", 231.036},{"U", 238.029},{"Np", 237},{"Pu", 244},{"Am", 243},{"Cm", 247},
    {"Bk", 247},{"Cf", 251},{"Es", 252},{"Fm", 257},{"Md", 258},{"No", 259},
    {"Lr", 262},{"Rf", 261},{"Db", 262},{"Sg", 266},{"Bh", 264},{"Hs", 277},
    {"Mt", 268}
};

char *protResNames[] = {"ABU", "ACE", "AIB", "ALA", "ARG", "ARGN", "ASN", "ASN1", "ASP", "ASP1", "ASPH", "ASH", "CT3", "CYS", "CYS1", "CYS2", "CYSH", "DALA", "GLN", "GLU", "GLUH", "GLH", "GLY", "HIS", "HIS1", "HISA", "HISB", "HISH", "HISD", "HISE", "HISP", "HSD", "HSE", "HSP", "HYP", "ILE", "LEU", "LSN", "LYS", "LYSH", "MELEU", "MET", "MEVAL", "NAC", "NME", "NHE", "NH2", "PHE", "PHEH", "PHEU", "PHL", "PRO", "SER", "THR", "TRP", "TRPH", "TRPU", "TYR", "TYRH", "TYRU", "VAL", "PGLU", "HID", "HIE", "HIP", "LYP", "LYN", "CYN", "CYM", "CYX", "DAB", "ORN", "HYP", "NALA", "NGLY", "NSER", "NTHR", "NLEU", "NILE", "NVAL", "NASN", "NGLN", "NARG", "NHID", "NHIE", "NHIP", "NHISD", "NHISE", "NHISH", "NTRP", "NPHE", "NTYR", "NGLU", "NASP", "NLYS", "NORN", "NDAB", "NLYSN", "NPRO", "NHYP", "NCYS", "NCYS2", "NMET", "NASPH", "NGLUH", "CALA", "CGLY", "CSER", "CTHR", "CLEU", "CILE", "CVAL", "CASN", "CGLN", "CARG", "CHID", "CHIE", "CHIP", "CHISD", "CHISE", "CHISH", "CTRP", "CPHE", "CTYR", "CGLU", "CASP", "CLYS", "CORN", "CDAB", "CLYSN", "CPRO", "CHYP", "CCYS", "CCYS2", "CMET", "CASPH", "CGLUH"};


static gmx_bool
isBackboneHydrogen(atom_id iHyd, atom_id iHeavy, t_atoms *atoms)
{
    gmx_bool bH, bN, bProtein = FALSE;
    int i;

    for (i = 0; i<asize(protResNames); i++)
    {
        if (! gmx_strcasecmp(protResNames[i], *(atoms->resinfo[atoms->atom[iHyd].resind].name)))
        {
            bProtein = TRUE;
            break;
        }
    }
    bN = (! gmx_strcasecmp(*(atoms->atomname[iHeavy]), "N"));
    bH = (! gmx_strcasecmp(*(atoms->atomname[iHyd]), "H") || ! gmx_strcasecmp(*(atoms->atomname[iHyd]), "HN"));

    return (bProtein && bN && bH);
}


static gmx_bool
isPolarizingHydrogen(const char* boundToElem)
{
    const char *elemIsPolarizingHydrogens[] = {"O", "N", "S", "P"};
    int i;

    for (i = 0; i<asize(elemIsPolarizingHydrogens); i++)
    {
        if (! gmx_strcasecmp(elemIsPolarizingHydrogens[i], boundToElem))
        {
            return TRUE;
        }
    }

    return FALSE;
}

/* Given a one- or two-character string, such as O, HE, or so, return the element such as O, He, or so,
 *  and the mass.
 */
static int
search_elements(const char *atom, real *mReturn, char *elem)
{
    int i, size;
    char el[3];

    strcpy(el, atom);
    el[0] = toupper(el[0]);
    if (strlen(el) == 2)
    {
        el[1] = tolower(el[1]);
    }
    size = asize(elemmass);
    for (i = 0; i < size; i++)
    {
        //printf("Compare .%s. with .%s.\n",el, elemmass[i].el);
        if (! strcmp(el, elemmass[i].el))
        {
            strcpy(elem, elemmass[i].el);
            *mReturn = elemmass[i].m;
            return 0;
        }
    }
    return 1;
}


#define ELEMENT_ALLOW_REL_MASS_DIFF 0.05
static int
get_elem(const char *aname, const char *resname, const real m, char *elemStr, int *maxwarn, t_genscatt_opt *opt, char *comment)
{
    char      nm[6], elem[3], tmp[2], *ptr, resnm[6];
    real      mass, frac, massdiff, relMassDiff;
    int       whole, i;
    gmx_bool  bOne=FALSE, bTwo=FALSE, bCarbon, bMaybeVsite, bNitrogen, bOxygen, bHydrogen, bSulfur;

    /* For now, avoid compiler warning */
    *maxwarn += 0;

    if (comment)
    {
        /* empty comment */
        comment[0] = '\0';
    }

    if (m < 0.0)
        gmx_fatal(FARGS, "Atom %s has mass < zero.\nThis is not implemented yet\n",
            aname);
    if (strlen(aname) > 5)
        gmx_fatal(FARGS, "Atom name %s too long\n", aname);
    /* remove numbers and make fist and second char upper and lower-case, respectively */
    strcpy(nm, aname);
    ptr = &nm[0];
    while (!isalpha(ptr[0]))
    {
        ptr++;
    }
    if (strlen(ptr) < 1)
    {
        gmx_fatal(FARGS,"Atom name %s is invalid. Maybe only numbers\n");
    }
    ptr[0] = toupper(ptr[0]);
    if (isalpha(ptr[1]))
    {
        ptr[1] = tolower(ptr[1]);
        ptr[2] = '\0';
    }
    else
    {
        ptr[1] = '\0';
    }

    if (m == 0.0)
    {
        /* If no mass is given:
           if (atom name == residue name) -> assume two letter element, otherwise use the first letter of the atom name */
        if (resname == NULL)
        {
            gmx_fatal(FARGS, "Inconsistency in get_elem()\n");
        }

        /* Check for water */
        if (!strcmp(resname, "HOH") || !strcmp(resname, "SOL"))
        {
            if (ptr[0] == 'O')
            {
                strcpy(elemStr, "Owat");
                return 0;
            }
            else if (ptr[0] == 'H')
            {
                strcpy(elemStr, "Hwat");
                return 0;
            }
        }

        strcpy(resnm, resname);
        resnm[0] = toupper(resnm[0]);
        i = 1;
        while (isalpha(resname[i]))
        {
            resnm[i] = tolower(resnm[i]);
            i++;
        }
        // printf ("comparing '%s' to '%s'\n", ptr, resnm);
        if ( ! strcmp(ptr, resnm))
        {
            strcpy(elemStr, ptr);
        }
        else
        {
            strncpy(elemStr, ptr, 1);
            elemStr[1] = '\0';
        }
        // printf("Returning %s for atom %s (resname %s)\n", elemStr, aname, resname);
        return 0;
    }

    /* Check one-character elements */
    strncpy(tmp, ptr, 1);
    tmp[1]='\0';

    if ( search_elements(tmp, &mass, &elem[0]) == 0)
    {
        /* fprintf(stderr,"Debug: mass comparisons. elem %s mass: %f, tmp %s m: %f.\n", elem, mass, tmp, m); */
        relMassDiff = fabs( (mass-m)/m);
        massdiff    = fabs(mass-m);
        /* Always first the strict test */
        if ( relMassDiff < ELEMENT_ALLOW_REL_MASS_DIFF )
        {
            // printf("Atom %s, mass %g: Found matching element: %s, mass %g\n",aname, m, elem, mass);
            bOne = TRUE;
            strcpy(elemStr, elem);
        }
        else
        {
            /* Catch for virtual sites C and N. Takes advantage of the fact that all 2-char elements have much larger masses  */
            /* Split into integral part and percentage deviation. */
            whole = round( (m-mass) / 1.008 ) ; /* Needs to be 1, 2, or 3 for vsite */
            frac  = fabs( m - mass - 1.008 * whole ) / 1.008 ;
            bMaybeVsite = (0 < whole && whole < 4);
            bCarbon   = (toupper(elem[0]) == 'C');
            bNitrogen = (toupper(elem[0]) == 'N');
            bHydrogen = (toupper(elem[0]) == 'H');
            bOxygen   = (toupper(elem[0]) == 'O');
            bSulfur   = (toupper(elem[0]) == 'S');

            /* fprintf(stderr,"frac: %f. whole: %i.\n", frac, whole); */

            if (opt->bGromos && bCarbon && (massdiff < 3.5))
            {
                sprintf(elemStr, "CH%d", whole);
                bOne = TRUE;
                if (comment)
                {
                    sprintf(comment, "Gromos united atom");
                }
            }
            else if (opt->bVsites && (massdiff < 3.5) && (bCarbon || bNitrogen))
            {
                bOne = TRUE;
                strcpy(elemStr, elem);
                if (comment)
                {
                    sprintf(comment, "mass difference of %g au due to v-sites", massdiff);
                }

                /*if ( *maxwarn > 0 )
                {
                    fprintf(stderr, "NB: Atom %-3s (mass %g) could be element %s - allowed for v-sites only, so please"
                            " check the assignments!\n", aname, m, elem);
                    *maxwarn = *maxwarn-1;
                    }*/
            }
            else if (massdiff < opt->maxmassdiff)
            {
                bOne = TRUE;
                strcpy(elemStr, elem);
                if (comment)
                {
                    sprintf(comment, "large mass difference of %g au - correct?", massdiff);
                }
                fprintf(stderr, "nAtom %-3s (mass %7g) could be element %s - "
                        "but better check the assignment (option -el)\n", aname, m, elem);
            }
        }
    }
    /* else
    {
        fprintf(stderr,"Debug: search_element has failed on single chars. Proceeding...\n");
    } */
    /* fprintf(stderr,"Debug: single-character comparison results: ptr: %s, tmp: %s, elemStr: %s.\n", ptr, tmp, elemStr);*/

    /* check two-character elements */
    if (strlen(ptr) == 2)
    {
        if (search_elements(ptr, &mass, &elem[0]) == 0)
        {
            relMassDiff = fabs( (mass-m)/m);
            massdiff = fabs(mass-m);
            if (relMassDiff < ELEMENT_ALLOW_REL_MASS_DIFF)
            {
                if (bOne)
                {
                    gmx_fatal(FARGS,"For atom %s (mass %g), both a one-letter and two-letter "
                              "chemical element seems to match\n", aname, m);
                }
                //printf("Atom %s: Found matching element: %s, mass %g\n",aname, elem, mass);
                bTwo = TRUE;
                strcpy(elemStr, elem);
            }
        }
    }
    if (bOne || bTwo)
    {
        return 0;
    }
    else
    {
        // gmx_fatal(FARGS,"Could not identify chemical element of atom %s (mass %g)\n",aname,m);
        strcpy(elemStr, "UNKNOWN");
        return 1;
    }
}

static void
write_dummy_top(const char *fn, const char *fnin,
                t_topology *top, int *maxwarn,  t_genscatt_opt *opt, gmx_bool bNSL)
{
    int        i,n,ntypes=0,j,ft;
    FILE      *fp;
    char      *grpname, *resname, defineName[STRLEN];
    charElem   elemStr, *elemHave=NULL;
    gmx_bool   bNew;

    /* Choose wheather we write neutron scattering lengths or Cromer-mann parameters */
    sprintf(defineName, "%s", bNSL ? "NEUTRON_SCATT_LEN_" : "CROMER_MANN_");
    ft = bNSL ? 2 : 1;

    n = top->atoms.nr;
    fp = ffopen(fn, "w");
    fprintf(fp,"; This is a trivial topology generated by g_genscatt from file %s\n;\n"
            "; This topology is intended to be used with mdrun -rerun to compute the scattering\n"
            "; intensity I(q) of a group of atoms (such as a CH3 group or similar)\n\n", fnin);
    fprintf(fp,"[ defaults ]\n1       2        yes      0.5     0.8333\n\n");
    fprintf(fp, "#include \"cromer-mann-defs.itp\"\n\n");

    fprintf(fp, "[ atomtypes ]\n");

    for (i=0; i<n; i++)
    {
        resname = *(top->atoms.resinfo[top->atoms.atom[i].resind].name);
        get_elem(*(top->atoms.atomname[i]), resname, 0.0, &elemStr[0], maxwarn, opt, NULL);
        bNew    = TRUE;
        for (j = 0; j < ntypes; j++)
        {
            if (!strcmp(elemStr, elemHave[j]))
            {
                bNew = FALSE;
            }
        }
        if (bNew)
        {
            fprintf(fp,"dummy_%-4s  %d   %5.2f  0.000  A  0.00000e+00  0.00000e+00\n",
                    elemStr, 1, top->atoms.atom[i].m);
            ntypes++;
            srenew(elemHave, ntypes);
            strcpy(elemHave[ntypes-1], elemStr);
        }
    }

    fprintf(fp,"\n[ moleculetype ]\nMolecule 2\n\n[ atoms ]\n");
    fprintf(fp, ";   nr     type         resnr residue  atom   cgnr     charge       mass\n");
    for (i = 0; i < n; i++)
    {
        resname = *(top->atoms.resinfo[top->atoms.atom[i].resind].name);
        get_elem(*(top->atoms.atomname[i]), resname, 0.0, &elemStr[0], maxwarn, opt, NULL);
        fprintf(fp,"   %3d     dummy_%-4s       1     %3s       %4s       %d  0.0  1.0\n",
                i+1, elemStr, resname, *(top->atoms.atomname[i]), i);
    }

    fprintf(fp, "\n[ scattering_params ]\n");
    if (!bNSL)
    {
        fprintf(fp, "; atom ft a1     a2      a3      a4      b1      b2      b3      b4      c\n");
    }
    else
    {
        fprintf(fp, "; atom ft NSL(Coh b)\n");
    }
    for (i = 0; i < n; i++)
    {
        resname = *(top->atoms.resinfo[top->atoms.atom[i].resind].name);
        get_elem(*(top->atoms.atomname[i]), resname, 0.0, &elemStr[0], maxwarn, opt, NULL);
        fprintf(fp, "%5d  1  %s%s\n", i+1, defineName, elemStr);
    }
    fprintf(fp, "\n[ system ]\nA trivial topology to compute scattering intensities with mdrun -rerun\n");
    fprintf(fp, "\n[ molecules ]\nMolecule       1\n");

    ffclose(fp);
    fprintf(stderr, "\nWrote %s\n\n", fn);
    sfree(elemHave);
}

int gmx_genscatt(int argc, char *argv[])
{
    const char
        *desc[] =
        {
            "[TT]g_genscatt[tt] produces include files (itp) for a topology containing ",
            "a list of atom numbers and definitions for X-ray or neutron scattering, ",
            "that is, definitions for Cromer-Mann parameters and Neutron Scattering Lengths ([TT]-nsl[tt]). ",
            "The tool will write one itp file per molecule type found in the selected index group."
            "For instance, if your protein contains two chains A and B, and these are defined in "
            "separate [TT][ moleculetype ][tt] blocks, then g_genscatt would write two itp files.[PAR]",
            "Like position restraint definitions, scattering-types are defined within molecules, ",
            "and therefore should be #included within the correct [TT][ moleculetype ][tt] ",
            "block in the topology. For instance, you can place it near the #include \"posre.itp\" statment ",
            "of the moleculetype definition.[PAR]",
            "Make sure to select a group that contains all physcial atoms of your solute, such as the protein, "
            "DNA/RNA, ligands, coordinated ions, heme groups etc. Maybe you will have to generate an index "
            "file first. Alternatively, you could also run g_genscatt once for each scattering molecule.[PAR]"
            "Because the chemical element is general not store in a tpr file, the tool guesses ",
            "the element using a combination of the atom name and the atom mass. Consequently, "
            "if the atomic masses deviate from the physical masses, either due to a united-atom force field, ",
            "or because you model hydrogen atoms as virtual sites, you must help g_genscatt with the options ",
            "[TT]-gromos[tt] or [TT]-vsites[tt], respectively.[PAR]",
            "Option [TT]-el[tt] writes files with more details on guessed chemical elements.[PAR]"
            "In addition, in case you use virtual sites, make sure to select a group ",
            "that contains only physical atoms, such as \"Prot-Masses\".[PAR]",
            "To merge the Cromer-Mann parameters of hydrogen atoms into the heavy atoms, use [TT]-ua[tt]. This is ",
            "mainly for testing.[PAR]"
        };

    static gmx_bool bDefault = FALSE, bUA = FALSE, bNSL = FALSE;
    static int maxwarn=20;
    t_genscatt_opt opt;
    opt.bGromos = FALSE;
    opt.bVsites = FALSE;
    opt.maxmassdiff = 0.3;

    t_pargs
        pa[] =
        {
            { "-def", FALSE, etBOOL, { &bDefault },
              "Write no scattering factors and allow grompp to read from the forcefield definitions later"},
            { "-maxwarn", 20, etINT, {&maxwarn},
              "Limit of v-site related warnings g_genscatt will show before they are suppressed." },
            { "-ua", FALSE, etBOOL, { &bUA },
              "Merge atomic scattering factors of hydrogens into bonded heavy atom" },
            { "-gromos", FALSE, etBOOL, { &opt.bGromos },
              "Expect united-atom carbon atoms" },
            { "-vsites", FALSE, etBOOL, { &opt.bVsites },
              "Expect vsites (H with mass zero & heavier C and N)" },
            { "-mdiff", FALSE, etREAL, { &opt.maxmassdiff },
              "Largest allowed mass difference for element assignments" },
            { "-nsl", FALSE, etBOOL, { &bNSL },
              "Write Neutron Scattering Lengths in addition to Cromer-Mann definitions" },
        };
#define NPA asize(pa)

    t_topology      *top=NULL;
    gmx_mtop_t       mtop;
    int              ePBC, *nHydOnHeavy=NULL;
    matrix           box;
    atom_id         *index, atom1, atom2, atom1ind, atom2ind, *bondedTo=NULL, **moltype_globalIndex = NULL;
    int              isize, i, f, j, ret,nHeavy=0, ncons, ft, maxwarnStart, *bHydrogen=NULL;
    int              moltype_firstAtomIndex, moltype_lastAtomIndex, *nInIndex_perMoltype = NULL, nMoltypeInSelection;
    int              mb, nmols, natoms_mol, molAtomIndex, moltype_instance, global_index, resid;
    gmx_bool       **bInIndex_perMoltype;
    char            *grpname, aname[5], fnAssign, buf[15], mulfn[STRLEN], *resname, fileHeader[STRLEN];
    FILE            *fpAssign = NULL, *fp=NULL;
    charElem         elemStr, *elems=NULL;
    t_iatom         *iatom = NULL;
    gmx_bool         bDummyTop, bHaveTpr, bAlert=FALSE, bBackboneH, bPolarH;
    t_state         dummystate;
    t_inputrec      dummyir;
    const char      *in_file, *fnUser;
    char            title[STRLEN], fnout[STRLEN], comment[STRLEN], defineName[STRLEN], *molname;
    rvec            *xtop;
    t_atoms        *atoms;
    time_t          timeNow;
    output_env_t     oenv;

    t_filenm fnm[] = {
        { efSTX, "-s",  NULL,         ffREAD },
        { efNDX, "-n",  NULL,         ffOPTRD },
        { efITP, "-o",  "scatter",    ffWRITE },
        { efDAT, "-el", "elemassign", ffOPTWR },
        { efTOP, "-p",  "dummy",      ffOPTWR },
    };
#define NFILE asize(fnm)

    CopyRight(stderr, argv[0]);
    parse_common_args(&argc, argv, PCA_CAN_VIEW, NFILE, fnm, NPA, pa,
                      asize(desc), desc, 0, NULL, &oenv);

    bDummyTop = opt2bSet("-p", NFILE, fnm);
    in_file   = ftp2fn(efSTX, NFILE, fnm);
    bHaveTpr  = (fn2ftp(in_file) == efTPR);
    snew(top,1);
    read_tps_conf(in_file, title, top, &ePBC, &xtop, NULL, box, 0);
    maxwarnStart = maxwarn;

    if (bDummyTop)
    {
        write_dummy_top(opt2fn("-p",NFILE,fnm), in_file, top, &maxwarn, &opt, bNSL);
        exit(0);
    }

    fprintf(stderr,"\nSelect atoms that scatter (e.g., Prot-Masses, Protein):\n");
    get_index(&top->atoms,ftp2fn_null(efNDX,NFILE,fnm),1,&isize,&index,&grpname);

    if (!bHaveTpr)
    {
        /* printf("NOTE: Having only a PDB file, I don't know about masses or bonds. So I am assuming\n" */
        /*        "      that we have only one molecule, and I will guess the element based on the atom\n" */
        /*        "      name and residue number\n" */
        /*        "      IMPORTANT: This will definitely fail if you have united atoms (GROMOS) or virtual sites.\n" */
        /*        "      so carefully check the itp file with the scattering info.\n\n"); */
        // guessFromPdbFile(&top, index, isize);
        gmx_fatal(FARGS,"Need a tpr file to write itp file with scattering information\n");
    }

    if (bNSL && bUA)
    {
        gmx_fatal(FARGS, "United atom form factors are not supported for neutron scattering.\n");
    }


    /*
     *  Read the molecule types from tpr, then check which atoms in the molecule types appear.
     *  This is done because we write one scatter.itp file for each molecule type.
     */
    read_tpx_state(in_file, &dummyir, &dummystate, NULL, &mtop);
    snew(bInIndex_perMoltype, mtop.nmolblock);
    snew(nInIndex_perMoltype, mtop.nmolblock);
    snew(moltype_globalIndex, mtop.nmolblock);
    moltype_firstAtomIndex = 0;
    nMoltypeInSelection = 0;

    printf("\nThe following molecule types are found in the tpr file:\n"
           "------------------------------------------------------\n");
    /* Loop over molecule types */
    for(mb = 0; mb < mtop.nmolblock; mb++)
    {
        nmols                  = mtop.molblock[mb].nmol;
        natoms_mol             = mtop.molblock[mb].natoms_mol;
        molname                = *(mtop.moltype[mtop.molblock[mb].type].name);
        moltype_lastAtomIndex  = moltype_firstAtomIndex + nmols*natoms_mol - 1;
        snew(bInIndex_perMoltype[mb], natoms_mol);
        snew(moltype_globalIndex[mb], natoms_mol);
        for (i = 0; i < natoms_mol; i++)
        {
            /* Make sure we get a Segfault in case we use below an atom that is not found in the index file */
            moltype_globalIndex[mb][i] = -999999999;
        }

        /* For this molecule type, make a boolean list that is TRUE if this atom appears in the index group */
        for (i = 0; i < isize; i++)
        {
            if (moltype_firstAtomIndex <= index[i] && index[i] <= moltype_lastAtomIndex)
            {
                molAtomIndex     = (index[i]-moltype_firstAtomIndex) % natoms_mol;
                moltype_instance = (index[i]-moltype_firstAtomIndex) / natoms_mol;

                if (moltype_instance > 0 && bInIndex_perMoltype[mb][molAtomIndex] == FALSE)
                {
                    gmx_fatal(FARGS, "Your selected index group \"%s\" contains multiple molecules of molecule type %s,\n"
                              "which is fine. However, your index groups seems to contain different atoms of the same\n"
                              "molecule type, which is not allowed. For instance, if you have two identical protein\n"
                              "chains, then your index group must contain exactly the same atoms of the two chains.\n"
                              "The problematic atom is atom with global index %d, or molecule no %d of type %s, \n"
                              "molecule-internal index %d\n",
                              grpname, molname, index[i]+1, moltype_instance+1, molname, molAtomIndex+1);
                }

                bInIndex_perMoltype[mb][molAtomIndex] = TRUE;

                /* Make an array that translates the atom number within a molecule type
                   into the global index */
                if (moltype_instance == 0)
                {
                    moltype_globalIndex[mb][molAtomIndex] = index[i];
                }
            }
        }
        /* Count the number of atoms of this moleucle type that appear somewhere in the index group. */
        for (i = 0; i < natoms_mol; i++)
        {
            if (bInIndex_perMoltype[mb][i])
            {
                nInIndex_perMoltype[mb]++;
            }
        }
        if (nInIndex_perMoltype[mb] > 0)
        {
            /* Count how many molecule types are represented in the index group */
            nMoltypeInSelection++;
        }

        printf("  %2d) %-25s  #molecules = %5d   #atoms = %4d   #atoms in user selection (\"%s\") = %d\n",
                mb+1, molname, nmols, natoms_mol, grpname, nInIndex_perMoltype[mb]);

        /* Set first index of the next molecule type */
        moltype_firstAtomIndex += nmols*natoms_mol;
    }
    printf("\nFound %d molecule types in the selection \"%s\".\n\n", nMoltypeInSelection, grpname);

    /* Get a time header into the scatter.itp files */
    time(&timeNow);
    gmx_ctime_r(&timeNow, fileHeader, STRLEN);

    /* Loop over moleucle types for writing itp files */
    for (mb = 0; mb < mtop.nmolblock; mb++)
    {
        /* Check if this moltype is in the index group */
        if (nInIndex_perMoltype[mb] == 0)
        {
            continue;
        }

        natoms_mol = mtop.molblock[mb].natoms_mol;
        molname    = *(mtop.moltype[mtop.molblock[mb].type].name);
        atoms      = &mtop.moltype[mtop.molblock[mb].type].atoms;

        fnUser = opt2fn("-o", NFILE, fnm);
        sprintf(fnout, "%.*s_%s.itp", (int)(strlen(fnUser)-4), fnUser, molname);
        printf("Writing scatter parameters of molecule \"%s\" into %s\n", molname, fnout);
        fp = ffopen(fnout, "w");
        fprintf(fp, "; Written by g_genscatt");
        fprintf(fp, "; This file was created %s\n", fileHeader);

        if (opt2bSet("-el", NFILE, fnm))
        {
            fnUser = opt2fn("-el", NFILE, fnm);
            sprintf(fnout, "%.*s_%s.dat", (int)(strlen(fnUser)-4), fnUser, molname);
            printf("Writing atom assignments of molecule %s into %s\n", molname, fnout);
            fpAssign = ffopen(fnout,"w");
        }

        /* array to store the element, whether the atom is a hydrogen atom, and to which heavy atom this
           H-atom is bound to (if it is an H-atom) */
        snew(elems,      natoms_mol);
        snew(bHydrogen,  natoms_mol);
        snew(bondedTo,   natoms_mol);
        for (i = 0; i < natoms_mol; i++)
        {
            bondedTo[i] = -1;
        }

        /* Next block: writing Cromer-Mann */
        fprintf(fp, "[ scattering_params ]\n");
        fprintf(fp, "; atom ft a1     a2      a3      a4      b1      b2      b3      b4      c\n");
        sprintf(defineName, "%s", "CROMER_MANN_");
        ft = 1;

        nHeavy = 0;
        for (i = 0; i < natoms_mol; i++)
        {
            if (! bInIndex_perMoltype[mb][i])
            {
                continue;
            }

            global_index = moltype_globalIndex[mb][i];

            if ( bDefault )
            {
                fprintf(fp,"%5d  1  \n", i+1);
                continue;
            }

            resname = *(atoms->resinfo[atoms->atom[i].resind].name);
            resid   = atoms->resinfo[atoms->atom[i].resind].nr;
            ret     = get_elem(*(atoms->atomname[i]), resname, atoms->atom[i].m,
                               &elemStr[0], &maxwarn, &opt, comment);
            if (ret != 0 && maxwarn > 0 )
            {
                fprintf(stderr,"WARNING -Atom nr %d, name %s, mass %g could not be assigned"
                        " to an element (maybe need option -vsites?)\n", global_index, *(atoms->atomname[i]),
                        atoms->atom[i].m);
                maxwarn--;
            }

            /* If we don't write united-atom form factors, we can write the CM parameters now. Otherwise,
               the CM parametes are written below */
            if (!bUA)
            {
                /* Write the Cromer-Mann or NSL definition */
                fprintf(fp, "%5d  %d  %s%-10s   ; %4s - %4s-%d\n", i+1, ft, defineName, elemStr,
                        *(atoms->atomname[i]), resname, resid);

                if (fpAssign)
                {
                    fprintf(fpAssign, "%4s-%-4d  %4s, mass %7.3f -> %-10s   %s\n", resname, resid,
                            *(atoms->atomname[i]), atoms->atom[i].m, elemStr, comment);
                }
            }

            strcpy(elems[i], elemStr);
            if (!strcmp(elemStr, "H"))
            {
                bHydrogen[i] = 1;
            }
            else
            {
                nHeavy++;
            }

            if ( !bAlert && !maxwarn )
            {
                fprintf(stderr, "\nMaximum number of mass and vsite-related warnings reached. Further notes suppressed.\n\n");
                fprintf(stderr,"NOTE: Warnings may occur because your system contains virtual sites. If so, make sure to select only the physical\n"
                        "atoms (e.g. group \"Prot-Masses\") but not the non-physical dummy atoms.\n\n");
                bAlert = TRUE;
            }
        }

        /* For each H-atom, find the heavy atom to which it is bound. This is needed for:
         *  1) United atom form factors.
         *  2) For Neutron Scattering Lengths to tell whether an H-atom is polar or not, and whether it is bound
         *     to a backbone nitrogen.
         */
        if (bUA || bNSL)
        {
            /* To do: The following routine should better pick the bonds from the mtop, not from the top. This way,
               we would not have to translate the molecule-internal atom numbers into the global atom numbers. Well,
               but it works for now...
            */
            /* Find out to which heavy atom the hydrogen is bound */
            printf("%s: Found %d hydrogen atoms and %d heavy atoms\n", molname, nInIndex_perMoltype[mb]-nHeavy, nHeavy);
            snew(nHydOnHeavy, nInIndex_perMoltype[mb]);

            for (f = 0; f < F_NRE; f++)
            {
                if (IS_CHEMBOND(f))
                {
                    /* Now loop over all chemical bonds */
                    iatom = top->idef.il[f].iatoms;
                    ncons = top->idef.il[f].nr/3;
                    for (j = 0; (j < top->idef.il[f].nr); j+=interaction_function[f].nratoms+1)
                    {
                        /* Caution: atom1 and atom2 are global atom indices, whereas the counter i, atom1ind, and atom2ind
                           are indices within the molecule type (0 <= atom1ind < natoms_mol)
                        */
                        atom1 = top->idef.il[f].iatoms[j+1];
                        atom2 = top->idef.il[f].iatoms[j+2];
                        /* find nr for index[] */
                        atom1ind = -1;
                        atom2ind = -1;
                        for (i = 0; i < natoms_mol; i++)
                        {
                            if (bInIndex_perMoltype[mb][i])
                            {
                                global_index = moltype_globalIndex[mb][i];
                                if (global_index == atom1)
                                {
                                    atom1ind = i;
                                }
                                if (global_index == atom2)
                                {
                                    atom2ind = i;
                                }
                            }
                            if (atom1ind >=0 && atom2ind >=0)
                            {
                                break;
                            }
                        }
                        if (atom1ind == -1 || atom2ind == -1)
                        {
                            /* Not both atoms are in index[] -> continue */
                            continue;
                        }
                        if ( (bHydrogen[atom1ind] == 0 && bHydrogen[atom2ind] == 1) ||
                             (bHydrogen[atom2ind] == 0 && bHydrogen[atom1ind] == 1))
                        {
                            /* increase nHydOnHeavy[] of the heavy atom */
                            if (bHydrogen[atom1ind] == 1)
                            {
                                nHydOnHeavy[atom2ind]++;
                                bondedTo[atom1ind] = atom2ind;
                            }
                            else
                            {
                                nHydOnHeavy[atom1ind]++;
                                bondedTo[atom2ind] = atom1ind;
                            }
                        }
                    }
                }
            }
        }

        if (bNSL)
        {
            /* Next block: writing NSL */
            fprintf(fp, "\n; atom ft NSL(Coh b)\n");
            sprintf(defineName, "%s", "NEUTRON_SCATT_LEN_");
            ft = 2;

            for (i = 0; i < natoms_mol; i++)
            {
                if (! bInIndex_perMoltype[mb][i])
                {
                    continue;
                }

                resname = *(atoms->resinfo[atoms->atom[i].resind].name);
                resid   = atoms->resinfo[atoms->atom[i].resind].nr;
                if (!bHydrogen[i])
                {
                    /* empty comment */
                    comment[0] = '\0';
                    /* Write the NSL definition */
                    fprintf(fp, "%5d  %d  %s%-10s   ; %4s - %s-%d\n", i+1, ft, defineName,
                            elems[i], *(atoms->atomname[i]), resname, resid);
                }
                else
                {
                    /* Check if the heavy atom to which this is bound is a backbone nitrogen */
                    bBackboneH = isBackboneHydrogen(i, bondedTo[i], atoms);

                    /* Check if this is a polar hydrogen since it is bound to O, N, S, or P */
                    bPolarH    = isPolarizingHydrogen(elems[bondedTo[i]]);

                    if (bBackboneH)
                    {
                        sprintf(comment, " -- Protein backbone hydrogen");
                        fprintf(fp, "%5d  %d  %-28s   ; %s-%s\n", i+1, ft,
                                "NSL_H_DEUTERATABLE_BACKBONE", resname, *(atoms->atomname[i]));
                    }
                    else if (bPolarH)
                    {
                        sprintf(comment, " -- deuteratable non-backbone hydrogen");
                        fprintf(fp, "%5d  %d  %-28s   ; %s-%s\n", i+1, ft,
                                "NSL_H_DEUTERATABLE", resname, *(atoms->atomname[i]));
                    }
                    else
                    {
                        sprintf(comment, " -- non-deuteratable non-polar hydrogen");
                        fprintf(fp, "%5d  %d  %-28s   ; %s-%s\n", i+1, ft,
                                "NEUTRON_SCATT_LEN_1H", resname, *(atoms->atomname[i]));
                    }
                }
                if (fpAssign)
                {
                    fprintf(fpAssign, "%4s, mass %7.3f -> %s%s\n", *(atoms->atomname[i]),
                            atoms->atom[i].m, buf, comment);
                }
            }
        }

        if (bUA)
        {
            /* Next block: writing unite-atom Cromer-Mann */
            sprintf(defineName, "%s", "CROMER_MANN_");
            ft = 1;

            /* For united atom scattering factors, write CH, CH2, NH3, etc. */
            for (i = 0; i < natoms_mol; i++)
            {
                if (! bInIndex_perMoltype[mb][i])
                {
                    continue;
                }

                if (!bHydrogen[i])
                {
                    /* printf("Found %d hydrogen bonded to heavy atom %d (%s)\n",nHydOnHeavy[i],index[i]+1,
                     *(atoms->atomname[i])); */
                    if (nHydOnHeavy[i] > 1)
                    {
                        sprintf(buf, "%sH%d", elems[i], nHydOnHeavy[i]);
                    }
                    else if (nHydOnHeavy[i] == 1)
                    {
                        sprintf(buf, "%sH", elems[i]);
                    }
                    else
                    {
                        sprintf(buf, "%s", elems[i]);
                    }

                    /* Write the Cromer-Mann or NSL definition */
                    fprintf(fp,"%5d  %d  %s%s\n", i+1, ft, defineName, buf);

                    if  (fpAssign)
                    {
                        fprintf(fpAssign, "%4s, mass %7.3f -> %s, %s bound to %d H\n", *(atoms->atomname[i]),
                                atoms->atom[i].m, buf, elems[i], nHydOnHeavy[i]);
                    }
                }
                else
                {
                    if (bondedTo[i] == -1)
                    {
                        /* Catch the case that this hydrogen is not bonded to any heavy atom */
                        fprintf(stderr,"WARNING -\n Atom %d (%s) of molecule type %s is not bonded to any heavy atom in the group.\n",
                                i+1, *(atoms->atomname[i]), molname);
                        fprintf(fp, "%5d  %d  %s%s\n", i+1, ft, defineName, elems[i]);
                        if (fpAssign)
                        {
                            fprintf(fpAssign, "%4s, mass %7.3f -> %s\n", *(atoms->atomname[i]),
                                    atoms->atom[i].m, elems[i]);
                        }
                    }
                    else
                    {
                        if  (fpAssign)
                        {
                            fprintf(fpAssign, "%4s, mass %7.3f -> merged into atom %d (%s)\n", *(atoms->atomname[i]),
                                    atoms->atom[i].m, bondedTo[i], *(atoms->atomname[bondedTo[i]]));
                        }
                    }
                }
            }
        } /* end if bUA */

        ffclose(fp);
        if (fpAssign)
        {
            ffclose(fpAssign);
        }
        if (nHydOnHeavy)
        {
            sfree(nHydOnHeavy);
        }
        sfree(elems);
        sfree(bHydrogen);
        sfree(bondedTo);
    }

    sfree(top);

    thanx(stderr);

    /* Return 1 if warnings were thrown */
    if (maxwarnStart == maxwarn)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}
