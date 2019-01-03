/*This file is part of the GROMACS molecular simulation package.
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

 *  Created on: Oct 28, 2016
 *  Last updated Version: Aug 17, 2018
 *  Author: fstrnad
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <maths.h>
#include "gromacs/utility/smalloc.h"
#include "gromacs/topology/index.h"
#include "gromacs/math/gmxcomplex.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/waxs/waxsrec.h"
#include "gromacs/waxs/waxstop.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/waxs/gmx_envelope.h"


#include"scattering_amplitude_cuda.cuh"
#include"device_averages.cu"

// Cuda libraries
#include <cuda.h>
#include <cuda_runtime.h>

#define waxs_debug(x)

#define GMX_DOUBLE_SUM

/* statically defined arrays which stay on GPU the whole time */
#define XX 0
#define YY 1
#define ZZ 2

#define DIM 3

#define MAX_NR_ATOMS_SHARED_MEM 800  // 48kB / (4*4*4)   4 SMs have the same shared memory, 4 =  x,y,z + atype , 4 = sizeof(real)

// Some definitions
const int threadsPerBlock = 256;        // Threads per block


//const int tot_shared_mem = 49152 ;    // Attention this is only by user and assumed it will be enough!
//const int shared_size = tot_shared_mem / (12) ;


/* Proper cuda errorchecking in case of GPU failure */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
//     More careful checking. However, this will affect performance.
//     Comment in if needed.
//        cudaError err = cudaGetLastError();
//        err = cudaDeviceSynchronize();
//            if( cudaSuccess != err )
//            {
//                fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
//                         file, line, cudaGetErrorString( err ) );
//                exit( -1 );
//        }

}

/* Neutron scattering lengths of water atoms, this is needed for the density correction */
#define NEUTRON_SCATT_LEN_1H        (-3.7406)
#define NEUTRON_SCATT_LEN_2H          6.671
#define NEUTRON_SCATT_LEN_O           5.803


/* Data structure for all data that will remain constantly on the GPU. */
typedef struct {
    int type;

    t_complex *avB;             /* Averaged water system scattering amplitude. Remains constant after some simulation steps */
    t_complex_d *A ;            /* Scattering Amplitude of sovlent system. It is the scattering amplitude of THIS waxs step
                                   and may not be confused with avA! */

    double elec2NSL_factor;

    int naff_table;            /* Nr of elements in atomic form factor table */
    real *aff_table;           /* The atomic form factors vs. |q| for each type  (NULL if this is a neutron grp)
                                Attention: This aff_table is linearized to be faster handled on the GPU. Therefore,
                                it is not a 2-dimensional array as on the CPU! */

    int  nnsl_table;           /* Nr of elements in nsl table */
    real *nsl_table;           /* The NSL, taking deuteration into account (NULL if this is a xray grp) */

    real *q_vec  ;             /* q-vectors for this scattering type
                                Attention: This array is linearized and not 2-dimensional as on the CPU */

    int *iTable;               /* stores for each q_vec its absolute value */

    int nabs ;                  /* Number of absolute q values */
    int *nabs_ind ;             /*delivers for each indices of nabs the number of q-vec which belong to this absolute |q| */

    t_complex_d *Orientational_Av;  /* time average of orientational average */
} t_gpudata_type;

/* Data structure for FT of the envelope that is stored on the GPU. Used for solvent FT as well for solute FT. */
typedef struct {
    real *re_ft;
    real *im_ft;
    int qhomenr;              /* stores number of q-vectors for 1 qabs on this node, important for solvent FT! */
} t_ft_GPU ;


/* Calculating in double precision on the GPU. Attention: Computationaly expansive! */
#if defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ double atomicAdd_double (double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
        /* Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) */
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ t_gpudata_type *d_gpudata_type;

/* for solvent density correction */
__device__ t_ft_GPU *d_unit_ft ;
__device__ t_ft_GPU *d_solvent_ft ;

__device__ double d_fac1A, d_fac2A, d_fac1B, d_fac2B;
__device__ real d_delta_rho_over_rho_bulk, d_delta_rho_B;  /* For correction terms in fix_solvent_density */

__device__ int count_blocks;
__device__ int this_block;

/*
 * This function does the solvent density correction on the GPU to avoid data transfer between GPU and CPU.
 * To do so, in waxsmd.c the relevant Fourier Transforms are always copied to GPU if they are newly calculated.
 */
__global__ void solvent_density_correction( int t, t_complex_d* scatt_amplitude, int qhomenr, int nabs , int nprot , gmx_bool bCalcForces)
{
    t_gpudata_type *this_gpudata_type;
    this_gpudata_type = &(d_gpudata_type[t]) ;

    t_ft_GPU *this_unit_ft, *this_solvent_ft;
    this_unit_ft    = &(d_unit_ft[t]) ;
    this_solvent_ft = &(d_solvent_ft[t]) ;
    int j;


    __syncthreads();
    /* Solute system A(q) including proteins */
    if (nprot > 0)
    {
        /* Solute density A(q) correction */
        for (j = threadIdx.x + blockIdx.x * gridDim.x ; j < qhomenr ; j += blockDim.x*gridDim.x )
        {
            /* ft_re/ft_im  are in units number of electrons. For neutron scattering, translate to units number of NSL
             * In case of SAXS-driven MD we need to store the values A[q] on the GPU to calculate forces afterwards.
            */
            if( bCalcForces == TRUE )
            {
                t_complex_d A_tmp ;

                A_tmp.re = scatt_amplitude[j].re += d_delta_rho_over_rho_bulk * this_gpudata_type->elec2NSL_factor * this_solvent_ft->re_ft[j];
                A_tmp.im = scatt_amplitude[j].im += d_delta_rho_over_rho_bulk * this_gpudata_type->elec2NSL_factor * this_solvent_ft->im_ft[j];

                this_gpudata_type->A[j].re =  (A_tmp.re) ;
                this_gpudata_type->A[j].im =  (A_tmp.im) ;
            }
            /* Rerun: No need to store A[j] since no forces are calculated. */
            else
            {
                scatt_amplitude[j].re += d_delta_rho_over_rho_bulk * this_gpudata_type->elec2NSL_factor * this_solvent_ft->re_ft[j];
                scatt_amplitude[j].im += d_delta_rho_over_rho_bulk * this_gpudata_type->elec2NSL_factor * this_solvent_ft->im_ft[j];
            }
        }
    }
    /* Solvent system B(q) */
    else if ( nprot == 0 )
    {
        /* Solvent density B(q) correction */
        for (j = threadIdx.x + blockIdx.x * gridDim.x ; j < qhomenr ; j += blockDim.x*gridDim.x )
        {
            t_complex tmp_water ;
            double d_fac1B_loc, d_fac2B_loc;
            d_fac1B_loc = d_fac1B;
            d_fac2B_loc = d_fac2B;

            /* Calculate temporal average of  <B(qvec)> over all q_vec */
            tmp_water.re = scatt_amplitude[j].re += d_delta_rho_B * this_gpudata_type->elec2NSL_factor * this_unit_ft->re_ft[j] ;
            tmp_water.im = scatt_amplitude[j].im += d_delta_rho_B * this_gpudata_type->elec2NSL_factor * this_unit_ft->im_ft[j] ;

            /* For Forces we need <B> in average to be stored on GPU. */
            if( bCalcForces == TRUE )
            {
                this_gpudata_type->avB[j] = d_c_accum_avg( this_gpudata_type->avB[j], tmp_water,  d_fac1B_loc, d_fac2B_loc );
            }
        }
    }
    /* This should never happen! */
    else
    {
        printf("ERROR! The scattering could not be calculated. nprot < 0. Check if nprot is calculated correctly! \n");
        __threadfence();
    }
}

/*
 * This function is the core of the GPU code. It calculated the scattering amplitudes A(q) resp. B(q) for each single q.
 * Each value is calculated in a different thread.
 * For the precise way of parallelizing see e.g. Felix Strnad: Computationally efficient prediction of SAXS patterns from MD simulations. (BA thesis).
 */
__global__ void calculate_scattering_amplitudes(int t, real* x_vec , t_complex_d* scatt_amplitude,
        int qhomenr, int isize, int nabs,  int *atype, int nprot,
        gmx_bool bCalcForces)
{
    t_gpudata_type *this_gpudata_type;
    this_gpudata_type = &(d_gpudata_type[t]) ;

    int   at , qabs , d ;
    real  aff_tmp , qdotx_tmp ;
    int   a , b , q ;
    int   nAtomsThisBlock , nBlocksUsed, start_atom ;

    nBlocksUsed = gridDim.x ;

    int atomsPerBlock = isize / nBlocksUsed ;
    int rest          = isize - (atomsPerBlock * nBlocksUsed) ;

    /* Pushing atom coordinates from global memory to shared memory (memory access is 10 to 100 times faster!)
     * Example: 4 blocks, number of atoms(isize) = 42:
     *          -> atomsPerBlock = 10, rest = 2
     *          Gives the following atoms_this_block: 11, 11, 10, 10
     */
    for ( b  = blockIdx.x ; b < gridDim.x ; b += gridDim.x)
    {
        if ( b >= rest )
        {
            start_atom       = b * atomsPerBlock + rest ;
            nAtomsThisBlock  = atomsPerBlock;
        }
        else
        {
            start_atom       = b * (atomsPerBlock + 1) ;
            nAtomsThisBlock  = atomsPerBlock + 1 ;
        }
    }

    /* Copy global atype and xvec to shared memory */
//    for (a = threadIdx.x + blockIdx.x * gridDim.x ; a < nAtomsThisBlock ; a+= blockDim.x*gridDim.x )
//    {
//        atypePerBlock[a] = atype[start_atom + a];
//        for (d = 0 ; d < DIM ; d++)
//        {
//            xvecSharedThisBlock[a * DIM + d] = x_vec[DIM * (start_atom + a) + d];
//        }
//    }

    /* loop over q_vec in each block *
     * Recall that every block has different atoms in shared memory!
     */
    for (q = threadIdx.x ; q < qhomenr ; q += blockDim.x )
    {

        double tmp_im , tmp_re ;
        float  f_exp_re, f_exp_im;
        float  sinvalue, cosvalue;
        float q_vec_this_thread[DIM];
        for (d = 0 ; d < DIM ; d++)
        {
            q_vec_this_thread[d] = this_gpudata_type->q_vec[q * DIM + d];
        }

        if(this_gpudata_type->type == escatterXRAY)
        {
            qabs = this_gpudata_type->iTable[q] ;
        }

        /* Loop over atoms in this block */
        for (b  = blockIdx.x ; b < gridDim.x ; b += gridDim.x)
        {
            tmp_im = 0. ;
            tmp_re = 0. ;

            for (a = 0 ; a < nAtomsThisBlock ; a++)
            {
//                at      = atypePerBlock[a];
                at      = atype[(a + start_atom)];
                if(this_gpudata_type->type == escatterXRAY)
                {
                    aff_tmp = this_gpudata_type->aff_table[ at * nabs  + qabs];
                }
                /* Neutron scattering NSL does NOT depend on |q| !
                 * But to keep this function generic for Xray and Neutron (for now), write the identical NSL into a parameter aff_tmp.
                 */
                else if(this_gpudata_type->type == escatterNEUTRON)
                {
                    aff_tmp = this_gpudata_type->nsl_table[at] ;
                }
                /* This should NEVER happen! */
                else
                {
                    printf("ERROR! The scattering type you chose could not be recognized. Allowed scattering types are xray and/or neutron! \n");
                    __threadfence();
                }

                qdotx_tmp = 0. ;

                for (d = 0 ; d < DIM ; d++)
                {
                    qdotx_tmp += (q_vec_this_thread[d] * x_vec[(a + start_atom) * DIM + d]);
                }

                sincosf (qdotx_tmp , &sinvalue , &cosvalue);

                f_exp_re = cosvalue * aff_tmp;
                f_exp_im = sinvalue * aff_tmp;

                tmp_re += f_exp_re ;
                tmp_im += f_exp_im ;
            }
            /*
             * For each q vector (index q), sum up contributions from all atoms
             * Atomic operations are needed to avoid race conditions.
            */
           atomicAdd_double ( &scatt_amplitude[q].im , tmp_im);
           atomicAdd_double ( &scatt_amplitude[q].re , tmp_re);
        }
    }
}

/*
 * Here, to calculate forces we need dkI(|q|). The computation is done in this function.
 * For details see Chen and Hub 2015.
 * Attention order of orientational average and time average are calculated in different order than in CPU code.
 * This was done to reduce memory that is needed on GPU. Especially for older GPUs (e.g. GTX 780 and older) this is relevant.
 */
__global__ void calculate_dkI( int t, double *dkI , real *x_vec, int *atype, int qstart, int qhomenr, int nabs , int nprot )
{
    t_gpudata_type *this_gpudata_type ;
    this_gpudata_type = &(d_gpudata_type[t]) ;
    int *nabs_ind = this_gpudata_type->nabs_ind ;

    int          i, l, p, d, j, jj, at;
    t_complex    A_tmp, dkAbar_tmp, A_avB_tmp;
    t_complex_d  tmpc1 ;
    t_complex_d  tmp_dkAbar_A_avB[DIM], tmp_Orientational_av ;
    float        q_vec_this_thread[DIM], x_vec_this_thread[DIM];
    float        sinvalue, cosvalue;
    real         aff_tmp, qdotx_tmp ;

   /* grad[k] D(q) = 2Re [ <A.grad[k]A*(q)> - <B>.<grad[k]A*(q)> ]
     * dkI = orientational average ( grad[k] D(q)  )
     * Here this calculation is splitted into 2 parts: the 1. summands is labeled as 1 the second summand as 2.
     */

    /*
     * Loop over the atoms. We calculate dkI for 1 atom per thread.
     * It is much faster to let each thread calculate as much as possible than to let each thread just do a few calculation.
     * Loop over protein atoms. Every thread deals with one protein atom.
     */
    for (p = threadIdx.x + blockDim.x * blockIdx.x; p < nprot; p += blockDim.x * gridDim.x)
    {
        for (d = 0; d < DIM ; d++)
        {
            x_vec_this_thread[d] = x_vec[p * DIM + d];
        }
        at = atype[p];

        aff_tmp = 0. ;
        for (i = 0 ; i < nabs ; i++)
        {

            for( d = 0; d< DIM; d++)
            {
                /* 1st summand */
                tmp_dkAbar_A_avB[d].re    = 0. ;
                tmp_dkAbar_A_avB[d].im    = 0. ;
            }

            if(this_gpudata_type->type == escatterXRAY)
            {
                aff_tmp = this_gpudata_type->aff_table[ at * nabs  + i];
            }
            /* Neutron scattering NSL does NOT depend on |q| !
             * But to keep this function generic for Xray and Neutron (for now), write the identical NSL into a parameter aff_tmp.
             */
            else if(this_gpudata_type->type == escatterNEUTRON)
            {
                aff_tmp = this_gpudata_type->nsl_table[at] ;
            }
            /* This should NEVER happen!*/
            else
            {
                printf("ERROR! The scattering type you chose could not be recognized. Allowed scattering types are xray and/or neutron! \n");
                __threadfence();
            }


            /* loop over q-vec which belong to this absolute value |q_i| */
            for (j = nabs_ind[i] ; j < nabs_ind[i+1] ; j++ )
            {
                if (j >= qstart && j < (qstart+qhomenr) )
                {
                    jj = j - qstart;              /* index of d_avB, d_A */

                    A_tmp.re = this_gpudata_type->A[jj].re ;
                    A_tmp.im = this_gpudata_type->A[jj].im ;

                    /* Direct evaluation of dkAbar. Attention on CPU this values are already stored while calculating A(q).
                     * Because memory access is that expensive on the GPU it is cheaper to calculate again the dkAbar!
                    */
                    qdotx_tmp = 0. ;
                    for(d = 0 ; d < DIM ; d++)
                    {
                        q_vec_this_thread[d] = this_gpudata_type->q_vec[jj * DIM + d] ;
                        qdotx_tmp += (q_vec_this_thread[d] * x_vec_this_thread[d]) ;
                    }

                    sincosf (qdotx_tmp , &sinvalue , &cosvalue);

                    dkAbar_tmp.re = -(sinvalue * aff_tmp);
                    dkAbar_tmp.im = -(cosvalue * aff_tmp);

                    /* Do product of A.grad[k]*.(A(q) -B_av)  (before time average)
                     * Not that after a certain number of steps <B> is not changing anymore, therefore, it can be taken as a constant
                    */

                    A_avB_tmp = d_cdiff(A_tmp, this_gpudata_type->avB[jj]);
                    tmpc1     = d_cmul_rd( A_avB_tmp , dkAbar_tmp );

                    for( d = 0 ; d < DIM ; d++)
                    {
                        tmp_dkAbar_A_avB[d].re   += tmpc1.re * q_vec_this_thread[d] ;
                        tmp_dkAbar_A_avB[d].im   += tmpc1.im * q_vec_this_thread[d] ;
                    }
                }
            }

            /* index of dkI */
            l = p * nabs + i;

            /* We changed order of calculation of orientational and time average.
             * That's why here has to be evaluated the time average
             */
            /* Do time average of <A.grad[k]A*(q)> */
            for( d = 0; d< DIM; d++)
            {
                tmp_Orientational_av = d_cd_accum_avg( this_gpudata_type->Orientational_Av[l * DIM + d] , tmp_dkAbar_A_avB[d] , d_fac1A , d_fac2A );
                this_gpudata_type->Orientational_Av[l * DIM + d] = tmp_Orientational_av ;
                /* Orientational average of 2Re [ <A.grad[k]*. (A(q)> - B_av>) ] */
                dkI[DIM * l + d] = 2. * tmp_Orientational_av.re ;
            }
        }

        for (i = 0 ; i < nabs; i++)
        {
            int J = (nabs_ind[i+1] - nabs_ind[i]);
            l = p * nabs + i;
            for (d = 0; d < DIM; d++)
            {
                dkI[DIM * l + d] /= J;       /* To fulfill the orientational average we divide each dkI by the number J of qvec which belong to 1 |q| */
            }
        }
    } /* End parallized loop over atoms */
}

/*
 * This function is needed to prepare the GPU for calculating the scattering amplitudes (Request memory, copy needed values to GPU and copy back results).
 */
void scattering_amplitude_initialize_GPU(int t, t_complex_d *scatt_amplitude , real *x_vec , int qhomenr , int isize , int *atype ,
        int nprot , int nabs , gmx_bool bCalcForces, gmx_bool bFixSolventDensity)
{
    real *dev_x_vec ;

    /* Initialization of atomic coordinates */
    gpuErrchk( cudaMalloc((void**)&dev_x_vec        , isize * DIM * sizeof(real)) );
    gpuErrchk ( cudaMemcpy(        dev_x_vec , x_vec, isize * DIM * sizeof(real), cudaMemcpyHostToDevice) );


    t_complex_d *dev_scatt_amplitude;
    gpuErrchk( cudaMalloc((void**)&dev_scatt_amplitude ,                   sizeof(t_complex_d) * qhomenr));
    gpuErrchk( cudaMemcpy(         dev_scatt_amplitude , scatt_amplitude,  sizeof(t_complex_d) * qhomenr, cudaMemcpyHostToDevice ) );

    int  *dev_atype ;
    gpuErrchk( cudaMalloc((void**)&dev_atype,          isize * sizeof(int)) );
    gpuErrchk( cudaMemcpy(         dev_atype , atype , isize * sizeof(int) , cudaMemcpyHostToDevice) );

    int nBlocksUsed = 0, number_SM = 0 , c , count ;

    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&count);
    for (c = 0; c < count; c++)
    {
        cudaGetDeviceProperties (&deviceProp, c);
        number_SM = deviceProp.multiProcessorCount ;
    }

    nBlocksUsed = isize / MAX_NR_ATOMS_SHARED_MEM ;

    /* Each GPU has a certain number of streaming multiprocessors (SM) on which the actual computation is done.
     * Each SM can have a certain number of blocks. Each thread is executed inside a block.
     * It is therefore useful, that at least all SMs have one block, such that no computation power is wasted.
     * If something went wrong with blocks and threads, i.g. see here: https://stackoverflow.com/questions/16125389/invalid-configuration-argument-error-for-the-call-of-cuda-kernel
     */
    if(nBlocksUsed < 1 )
    {
        nBlocksUsed = number_SM ;
    }
    else if (nBlocksUsed % number_SM != 0)
    {
        nBlocksUsed += number_SM - (nBlocksUsed % number_SM);
    }

    if (nBlocksUsed < number_SM)
    {
        gmx_fatal(FARGS, "Something went wrong with the calibration of the GPU! Check system size and blocksize: /n"
                         "system size: %d \n"
                         "number SMs:  %d \n"
                         "number of blocks: %d \n", isize, number_SM, nBlocksUsed) ;
    }

    /* This function calls the GPU. The scattering amplitudes are now calculated on the GPU.
     * Attention this function works only if GPU is available, it will produce an error otherwise! */
    calculate_scattering_amplitudes <<<  nBlocksUsed, threadsPerBlock >>> (t, dev_x_vec ,  dev_scatt_amplitude ,
            qhomenr , isize ,  nabs , dev_atype , nprot ,
            bCalcForces);

    gpuErrchk( cudaPeekAtLastError() );

    /*
     * Solvent density correction, if needed.
     * 1 single block is necessary to avoid race condition with calculate_scattering_amplitudes, which can still be calculated in other blocks.
     */
    if( bFixSolventDensity == TRUE )
    {
        solvent_density_correction <<< 1 , threadsPerBlock >>> ( t, dev_scatt_amplitude , qhomenr , nabs , nprot , bCalcForces) ;
    }
    gpuErrchk( cudaMemcpy(scatt_amplitude, dev_scatt_amplitude ,sizeof(t_complex_d) * qhomenr, cudaMemcpyDeviceToHost));
    gpuErrchk( cudaPeekAtLastError() );


    /* Free memomry on GPU to avoid overload. */
    cudaFree(dev_scatt_amplitude);
    cudaFree(dev_x_vec);
    cudaFree(dev_atype);
}

/*
 * This function is called in waxsmd.c . It transposes data from CPU code to code that can be better processed by GPU (e.g. 2d arrays are linearized).
 */
void compute_scattering_amplitude_cuda (t_waxsrec *wr , t_commrec *cr , int t,  t_complex_d *sf, rvec *atomEnvelope_coord, int *atomEnvelope_type , int isize, int nprot,
                                        int qhomenr , real **aff_table,
                                        real *nsl_table,  int nabs ,
                                        double normA, double normB, double scale, gmx_bool bCalcForces, gmx_bool bFixSolventDensity, double *elapsedTime_d)
{

    if (aff_table && nsl_table)
    {
        gmx_fatal(FARGS, "aff_table and nsl_table are both != NULL !\n");
    }
    if (!aff_table && !nsl_table)
    {
        gmx_fatal(FARGS, "aff_table and nsl_table are both == NULL !\n");
    }

    int  i, j;

    cudaEvent_t start, stop;
    cudaEventCreate (&start) ;
    cudaEventCreate (&stop );
    cudaEventRecord(  start , 0);

    real *x_linearized;
    snew(x_linearized, isize * DIM);

    for (i = 0; i < qhomenr; i++)
    {
        sf[i] = cnul_d; // sf[i] = { 0.0, 0.0 };
    }

    /* linear arrays are on GPU much faster than 2-dim arrays. That's why we linearize the x-coordinates */
    for (i = 0 ; i < isize ; i++)
    {
        for(j = 0 ; j < DIM ; j++)
        {
            x_linearized[i*DIM + j] = atomEnvelope_coord[i][j];
        }
    }

    int *atype;
    snew(atype , isize);
    /* The atomtype of each atom is needed for getting the correct Chromer-Mann-parameters */
    for(i = 0 ; i < isize; i++)
    {
        atype[i] = atomEnvelope_type[i];
    }

    if (bCalcForces)
    {
        /* updating scaling parameters, this has to be done because we calculate already the time averages during calculation of scattering Amplitude.
         * We need this values only in case of SAXS driven MD simulation.
         */
        double fac1A,fac2A,fac1B,fac2B;
        normA = 1.0 + scale * normA;
        normB = 1.0 + normB;            /* NormB is without scaling parameter */

        fac1A   = 1.0 * (normA - 1.) / normA;
        fac2A   = 1.0 / normA;

        fac1B   = 1.0 * (normB - 1.) / normB;
        fac2B  = 1.0 / normB;

        gpuErrchk( cudaMemcpyToSymbol(d_fac1A , &fac1A , sizeof(double) , 0 , cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpyToSymbol(d_fac2A , &fac2A , sizeof(double) , 0 , cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpyToSymbol(d_fac1B , &fac1B , sizeof(double) , 0 , cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpyToSymbol(d_fac2B , &fac2B , sizeof(double) , 0 , cudaMemcpyHostToDevice) );
    }
    if (PAR(cr))
    {
        gmx_bcast(sizeof(double), &wr->solElecDensAv, cr);
    }

    real delta_rho_over_rho_bulk = (wr->givenElecSolventDensity - wr->solElecDensAv) / wr->solElecDensAv;
    real delta_rho_B             = (wr->givenElecSolventDensity - wr->solElecDensAv_SysB);

    gpuErrchk( cudaMemcpyToSymbol(d_delta_rho_over_rho_bulk , &delta_rho_over_rho_bulk , sizeof(real) , 0 , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpyToSymbol(d_delta_rho_B , &delta_rho_B , sizeof(real) , 0 , cudaMemcpyHostToDevice) );


    /* In this function the GPU calculation is prepared and the GPU calculation will be executed */
    scattering_amplitude_initialize_GPU(t, sf, x_linearized ,  qhomenr , isize , atomEnvelope_type , nprot , nabs ,  bCalcForces , bFixSolventDensity);

    sfree(x_linearized);  sfree(atype);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime = 0.;
    cudaEventElapsedTime (&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    *elapsedTime_d = elapsedTime

    waxs_debug("End of compute_structure_factor_md()\n");
}

/*
 * This function prepares the GPU to calculate the values dkI directly on the GPU. This function is directly called in waxsmd.c .
 * In case of 2 (or even more) GPUs we need to perform a summation of values and gather(gmx_bcast) bewtween different MPI threads.
 */
void calculate_dkI_GPU(int this_type, dvec *dkI , rvec *atomEnvelope_coord, int *atomEnvelope_type ,
                       int qstart, int qhomenr, int nabs , int nprot, double *GPU_time ,  t_commrec *cr)
{
    int i , j, d;
    real *x_linearized, *dev_x_vec ;

    /* Compare to compute_scattering_amplitude_cuda */
    snew(x_linearized, nprot * DIM);
    for (i = 0 ; i < nprot ; i++)
    {
        for (j = 0 ; j < DIM ; j++)
        {
            x_linearized[i * DIM + j] = atomEnvelope_coord[i][j];
        }
    }

    /* Initialization of atomic coordinates */
    gpuErrchk( cudaMalloc((void**)&dev_x_vec        , nprot * DIM * sizeof(real)) );
    gpuErrchk ( cudaMemcpy(        dev_x_vec ,x_linearized , nprot * DIM * sizeof(real), cudaMemcpyHostToDevice) );

    int *atype, *dev_atype;
    snew(atype , nprot);
    /* The atomtype of each atom is needed for getting the correct Chromer-Mann-parameters */
    for(i = 0 ; i < nprot; i++)
    {
        atype[i] = atomEnvelope_type[i];
    }
    gpuErrchk( cudaMalloc((void**)&dev_atype,          nprot * sizeof(int)) );
    gpuErrchk( cudaMemcpy(         dev_atype , atype , nprot * sizeof(int) , cudaMemcpyHostToDevice) );

    double *dkI_linear , *dev_dkI;

    snew(dkI_linear, DIM * nabs * nprot);

    gpuErrchk( cudaMalloc((void**)&dev_dkI , sizeof(double) * DIM * nabs * nprot));
    gpuErrchk( cudaMemset(dev_dkI,   0.0 ,   sizeof(double) * DIM * nabs * nprot) );

    int blocksize_this_GPU, number_SM = 0 , c , count ;

    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&count);
    for (c = 0; c < count; c++)
    {
        cudaGetDeviceProperties (&deviceProp, c);
        number_SM = deviceProp.multiProcessorCount ;
    }
    blocksize_this_GPU = number_SM ;

    cudaEvent_t start_direct, stop_direct;
    cudaEventCreate (&start_direct) ;
    cudaEventCreate (&stop_direct );
    cudaEventRecord(  start_direct , 0);

    calculate_dkI <<< blocksize_this_GPU , threadsPerBlock >>> ( this_type, dev_dkI , dev_x_vec, dev_atype , qstart , qhomenr ,  nabs , nprot );

    gpuErrchk( cudaDeviceSynchronize() );

    /* Copy back results */
    gpuErrchk( cudaMemcpy(dkI_linear , dev_dkI , sizeof(double) * DIM * nabs * nprot, cudaMemcpyDeviceToHost) );

    cudaEventRecord(stop_direct,0);
    cudaEventSynchronize(stop_direct);
    float elapsedTime ;
    cudaEventElapsedTime (&elapsedTime, start_direct , stop_direct);
    cudaEventDestroy(start_direct);
    cudaEventDestroy(stop_direct);

    for (i = 0; i < nabs * nprot ; i++)
    {
        for (d = 0; d < DIM ; d++)
        {
            dkI[i][d] = dkI_linear[i * DIM + d] ;
        }
    }

    /* Sum up if 2 or more GPUs are used (includes All_gather of MPI threads. */
    if (PAR(cr))
    {
        gmx_sumd(nabs * nprot * DIM, dkI[0], cr);
    }

    if (MASTER(cr)) *GPU_time = elapsedTime;

    sfree(dkI_linear);
    sfree(atype);
    sfree(x_linearized);
    cudaFree(dev_dkI);
    cudaFree(dev_x_vec);
    cudaFree(dev_atype);
}


/*
 * Memory check to prevent computation on GPU if there is not enough memory space on the GPU!
 * Attention if to much memory is requested, the simulation ends with an gmx_fatal, it will
 * not fall back to CPU computation.
 */
void GPU_memory_check( t_waxsrec *wr, long int memory_limit)
{
    int t ;
    t_waxsrecType *wt = wr->wt ;
    gmx_bool bCalcForces = wr->bCalcForces ;
    int nTypes = wr->nTypes ;
    int nprot = wr->nindA_prot ;
    int nabs = 0 ;
    int this_qhomenr = 0;
    int isize = wr->isizeA ;
    long int sum   = 0;
    gmx_bool bMemoryOK;

    for (t = 0 ; t < nTypes ; t++)
    {
        nabs = wt[t].qvecs->nabs;
        this_qhomenr = wt[t].qvecs->qhomenr ;
        long int A     = sizeof(t_complex) * this_qhomenr ;
        long int q_vec = sizeof(real) * DIM * this_qhomenr;
        long int r_vec = sizeof(real) * DIM * isize;
        long int rerun = 2*A + q_vec + r_vec;

        fprintf(stderr, "Free Memory: %ld MB \n ", memory_limit/(1024*1024)) ;
        if (bCalcForces)
        {
            long int avB         = sizeof(t_complex)         * this_qhomenr;
            long int FT          = sizeof(real)              * this_qhomenr;
            long int av_Omega_av = sizeof(t_complex_d) * DIM * nprot * nabs;
            long int dkI         = sizeof(double)      * DIM * nprot * nabs;


            sum = rerun + avB + 2* FT + av_Omega_av + dkI;

            fprintf(stderr, ""
                    "A part      %ld \n"
                    "avB         %ld \n"
                    "FT          %ld \n"
                    "av_Omega_av %ld \n"
                    "dkI         %ld \n"
                    "----------------------------------------- \n"
                    "required memory GPU %f MB \n", rerun, avB, FT , av_Omega_av , dkI , sum/(1024*1024.));
        }
        else
        {
            sum = rerun;
            fprintf(stderr, ""
                    "A part  %ld \n"
                    "----------------------------------------- \n"
                    "required memory GPU %f MB \n", rerun, sum/(1024*1024.) );
        }
    }
    bMemoryOK = (sum < memory_limit);
    if (!bMemoryOK)
    {
        fprintf(stderr, "\nERROR, not enough memory on the GPU. The available free memory on the GPU is %ld but you are allocating %ld\n"
                "qhomenr %d \n "
                "nabs    %d \n", memory_limit, sum, this_qhomenr, nabs);

        gmx_fatal(FARGS, " You are allocating too much memory on the GPU! Please consider a calculation "
                  "on the CPU or use a smaller qhomenr and/or absolute number of qvec!\n");
    }
}

/*
 * This function is called in waxsmd.c during the initialization process.
 * Here we copy all relevant constants and data to GPU and request for memory for values that will stay on the GPU.
 * This function works for 1 single scattering as for multiple scattering types.
 * We differ between X-Ray scattering and Neutron scattering types.
 */
void init_gpudata_type(t_waxsrec *wr)
{
    int t, k, i, j ;
    t_waxsrecType *wt = wr->wt ;
    gmx_bool bCalcForces = wr->bCalcForces ;
    int nTypes = wr->nTypes ;
    int nprot = wr->nindA_prot ;
    int naff_table , nnsl_table, qstart, nabs ;
    double  elec2NSL_factor, nslPerMol;
    int this_qhomenr , qhomenr_max = 0 , nabs_max = 0 ;

    /*
     * It is complicated to allocate memory on GPU for data structures dynamically. We therefore allocate first the memory on the CPU, prepare the data,
     * and then copy the single entries of the data type t_gpudata_type to GPU. At the end the data structure as a whole is copied to GPU.
     * We chose the following labeling:
     * t_gpudata_type *h_gpudata contains the data on the host.
     * t_gpudata_type *host_to_device_data contains a pointer to the device memory in elements which can be passed to the kernels as parameters. The memory is on the GPU.
     * t_gpudata_type *d_data_alloc contains the data on the device, hence, it is completely on the GPU and can be used like data on the host.
     */
    t_gpudata_type *h_gpudata            = (t_gpudata_type*) malloc (nTypes * sizeof(t_gpudata_type) ) ;
    t_gpudata_type* host_to_device_data  = (t_gpudata_type*) malloc (nTypes * sizeof(t_gpudata_type) ) ;
    t_gpudata_type *d_gpudata_type_alloc = (t_gpudata_type*) malloc (nTypes * sizeof(t_gpudata_type) ) ;

    if (wr->debugLvl > 1)
    {
        fprintf(stderr, "Initiating GPU stuff, size of gpu_data %d -- nTypes = %d\n",
                int ( sizeof(t_gpudata_type) ), nTypes );
    }

    cudaDeviceProp deviceProp;
    int c, count;
    cudaGetDeviceCount(&count);
    long int totalMemory=0;
    for (c = 0; c < count; c++)
    {
        cudaGetDeviceProperties (&deviceProp, c);
        totalMemory = deviceProp.totalGlobalMem ;
    }

    GPU_memory_check(wr, totalMemory);


    /* prepare Data set on CPU, this will then be copied to GPU */
    for (t = 0 ; t < nTypes ; t++)
    {
        qstart            = wt[t].qvecs->qstart;
        nabs              = wt[t].qvecs->nabs;
        h_gpudata[t].nabs = nabs ;


        this_qhomenr = wt[t].qvecs->qhomenr ;

        if (this_qhomenr > qhomenr_max)
        {
            qhomenr_max = this_qhomenr ;
        }
        if (nabs > nabs_max)
        {
            nabs_max = nabs ;
        }

        h_gpudata[t].type = wt[t].type ;

        /* Differ between the possible scattering types */
        if (wt[t].type == escatterXRAY)
        {
            h_gpudata[t].elec2NSL_factor = 1. ;

            nnsl_table = 0 ;
            naff_table = wt[t].naff_table ;

            h_gpudata[t].nnsl_table = nnsl_table ;
            h_gpudata[t].naff_table = naff_table ;

            real *aff_linearized ;
            snew(aff_linearized , naff_table * nabs);
            for (k = 0 ;  k < naff_table ; k++)
            {
                for (j = 0 ; j < nabs ; j++)
                {
                    aff_linearized[k * nabs + j] =  wt[t].aff_table[k][j];
                }
            }

            h_gpudata[t].aff_table = (real*) malloc(        nabs * naff_table * sizeof(real) ) ;
            /* Prepare data structure on CPU that can than be copied to GPU !*/
            memcpy(h_gpudata[t].aff_table, aff_linearized , nabs * naff_table * sizeof(real) ) ;

            h_gpudata[t].nsl_table = NULL;

            if (wr->debugLvl > 1)
            {
                fprintf(stderr, "Copying new table of atomic form factors to GPU. naff_table = %d \n", naff_table);
            }

            h_gpudata[t].iTable = (int*) malloc(                         this_qhomenr * sizeof(int) ) ;
            memcpy(h_gpudata[t].iTable, (wt[t].qvecs->iTable + qstart) , this_qhomenr * sizeof(int) ) ;

            sfree(aff_linearized);
        }
        else if (wt[t].type == escatterNEUTRON)
        {
            real deuter_conc = wt[t].deuter_conc;

            nslPerMol = NEUTRON_SCATT_LEN_O +
                    2 * ( (1. - deuter_conc) * NEUTRON_SCATT_LEN_1H + deuter_conc * NEUTRON_SCATT_LEN_2H);
            elec2NSL_factor = nslPerMol/10;
            h_gpudata[t].elec2NSL_factor = elec2NSL_factor ;

            nnsl_table = wt[t].nnsl_table;
            naff_table = 0 ;

            h_gpudata[t].nnsl_table = nnsl_table ;
            h_gpudata[t].naff_table = naff_table ;

            h_gpudata[t].nsl_table = (real*) malloc( nnsl_table * sizeof(real) ) ;
            memcpy(h_gpudata[t].nsl_table, (wt[t].nsl_table) , nnsl_table * sizeof(real) ) ;

            h_gpudata[t].aff_table = NULL;

            if (wr->debugLvl > 1)
            {
                fprintf(stderr, "Copying new NSL table to GPU! nnsl_table = %d \n", nnsl_table);
            }
        }
        else
        {
            gmx_fatal(FARGS, "The scattering type you chose could not been recognized!\n");
        }

        rvec *qvec;
        real *q_linearized ;
        /* allocation of memory for qvec
         * ATTENTION in case of MPI parallelization: we only copy these q vec to the GPU which we really need on this node.
         * Therefore this d_gpudata_type[t].q_vec is NOT equal to wt->qvec->q:
         * Each GPU contains only these qvec with which it is calculating
         * */
        snew(q_linearized ,   this_qhomenr * DIM);
        qvec = wt[t].qvecs->q + qstart;
        for (i = 0 ; i < this_qhomenr; i++)
        {
            for (j = 0; j < DIM; j++)
            {
                q_linearized[i * DIM + j] = qvec[i][j];
            }
        }
        h_gpudata[t].q_vec = (real*) malloc(      this_qhomenr * DIM * sizeof(real) ) ;
        memcpy(h_gpudata[t].q_vec, q_linearized , this_qhomenr * DIM * sizeof(real) ) ;

        if ( bCalcForces )     /* memory is only allocated if we are calculating the forces */
        {
            h_gpudata[t].nabs_ind = (int*) malloc(                (nabs + 1) * sizeof(int) ) ;
            memcpy(h_gpudata[t].nabs_ind , (wt[t].qvecs->ind) ,   (nabs + 1) * sizeof(int) ) ;
        }
        else
        {
            h_gpudata[t].nabs_ind = NULL ;
            h_gpudata[t].avB = NULL ;
            h_gpudata[t].A = NULL ;
            h_gpudata[t].Orientational_Av = NULL ;
        }
        sfree(q_linearized);

    }
    /* Prepare data structure of different types on CPU that can than be copied to GPU !*/
    memcpy (host_to_device_data, h_gpudata, nTypes * sizeof(t_gpudata_type) ) ;

    for (t = 0 ; t < nTypes ; t++)
    {
        nabs         = wt[t].qvecs->nabs;
        qstart       = wt[t].qvecs->qstart;
        this_qhomenr = wt[t].qvecs->qhomenr ;

        if(wt[t].type == escatterXRAY)
        {
            naff_table = wt[t].naff_table;

            if (wr->debugLvl > 1)
            {
                fprintf(stderr, "starting copying constant X-Ray data to GPUs ... \n" );
            }

            gpuErrchk( cudaMalloc(&(host_to_device_data[t].aff_table) ,                        nabs * naff_table * sizeof(real) ) ) ;
            gpuErrchk( cudaMemcpy(  host_to_device_data[t].aff_table , h_gpudata[t].aff_table, nabs * naff_table * sizeof(real), cudaMemcpyHostToDevice) );

            gpuErrchk( cudaMalloc(&(host_to_device_data[t].iTable) ,                      this_qhomenr * sizeof(int) ) );
            gpuErrchk( cudaMemcpy(  host_to_device_data[t].iTable  , h_gpudata[t].iTable, this_qhomenr * sizeof(int), cudaMemcpyHostToDevice) );


        }
        else if (wt[t].type == escatterNEUTRON)
        {
            nnsl_table = wt[t].nnsl_table;

            if (wr->debugLvl > 1)
            {
                fprintf(stderr, "starting copying constant Neutron data to GPUs ... \n" );
            }

            gpuErrchk( cudaMalloc(&(host_to_device_data[t].nsl_table) ,                        nnsl_table * sizeof(real) ) );
            gpuErrchk( cudaMemcpy(  host_to_device_data[t].nsl_table , h_gpudata[t].nsl_table, nnsl_table * sizeof(real), cudaMemcpyHostToDevice) );
        }
        else
        {
            gmx_fatal(FARGS, "The scattering type you chose could not been recognized on the GPU!\n");
        }

        gpuErrchk( cudaMalloc(&(host_to_device_data[t].q_vec) ,                     this_qhomenr * DIM * sizeof(real) ) );
        gpuErrchk( cudaMemcpy(  host_to_device_data[t].q_vec  , h_gpudata[t].q_vec, this_qhomenr * DIM * sizeof(real), cudaMemcpyHostToDevice) );

        if( bCalcForces )
        {
            gpuErrchk( cudaMalloc(&(host_to_device_data[t].avB) ,   this_qhomenr * sizeof(t_complex) ) ) ;
            gpuErrchk( cudaMemset(  host_to_device_data[t].avB, 0 , this_qhomenr * sizeof(t_complex) ) ) ;

            gpuErrchk( cudaMalloc(&(host_to_device_data[t].A) ,   this_qhomenr * sizeof(t_complex_d) ) ) ;
            gpuErrchk( cudaMemset(  host_to_device_data[t].A, 0 , this_qhomenr * sizeof(t_complex_d) ) ) ;

            gpuErrchk( cudaMalloc(&(host_to_device_data[t].nabs_ind) ,                        (nabs + 1) * sizeof(int) ) );
            gpuErrchk( cudaMemcpy(  host_to_device_data[t].nabs_ind  , h_gpudata[t].nabs_ind, (nabs + 1) * sizeof(int), cudaMemcpyHostToDevice) );

            gpuErrchk( cudaMalloc(&(host_to_device_data[t].Orientational_Av) ,   DIM * nabs * nprot * sizeof(t_complex_d) ) ) ;
            gpuErrchk( cudaMemset(  host_to_device_data[t].Orientational_Av, 0 , DIM * nabs * nprot * sizeof(t_complex_d) ) ) ;
        }

        fprintf(stderr, "... finished copying constant data to all GPUs.\n" );
    }

    /* Push the whole t_gpudata structure to the GPU */
    gpuErrchk( cudaMalloc(&d_gpudata_type_alloc ,                        nTypes  *  sizeof(t_gpudata_type) ) );
    gpuErrchk( cudaMemcpy( d_gpudata_type_alloc,   host_to_device_data , nTypes  *  sizeof(t_gpudata_type), cudaMemcpyHostToDevice ) );

    /* Copying data structure t_gpudata_type in global memory of GPU. */
    gpuErrchk( cudaMemcpyToSymbol(d_gpudata_type , &d_gpudata_type_alloc ,   sizeof(t_gpudata_type*) )) ;
    gpuErrchk( cudaPeekAtLastError() );

    fprintf(stderr, "SWAXS/SANS-data copied to GPU!\n");
}

/* The FT of the envelope in the pure water system B is copied to GPU in this function.
 * Since the Unit FT remains constant during the whole simulation, this function is only called once during the
 * initialization process in waxsmd.c .
 */
void push_unitFT_to_GPU(t_waxsrec *wr)
{
    int t ;
    t_waxsrecType *wt = wr->wt ;
    int nTypes = wr->nTypes ;
    int this_qhomenr, qstart ;

    real *ft_reEnv, *ft_imEnv ;

    rvec *q;

    t_ft_GPU *h_unit_ft,  *host_to_device_unit_ft,  *d_unit_ft_alloc ;

    snew( h_unit_ft,  nTypes ) ;
    snew (host_to_device_unit_ft , nTypes ) ;

    fprintf(stderr, " Started coping unitFT data to GPU... \n " );

    /* prepare Data set on CPU, this will then be copied to GPU */
    for (t = 0 ; t < nTypes ; t++)
    {

        qstart       = wr->wt[t].qvecs->qstart;
        this_qhomenr = wr->wt[t].qvecs->qhomenr;
        q            = wr->wt[t].qvecs->q;

        fprintf(stderr, "InitMD: Doing FT of unit density in envelope ... \n");
        gmx_envelope_unitFourierTransform(wr->wt[t].envelope, q + qstart, this_qhomenr, &ft_reEnv, &ft_imEnv);
        fprintf(stderr, "... UnitFT  is done! \n");

        if (!ft_reEnv)
        {
            gmx_fatal(FARGS, "Reading of the Unit Fourier Transform failed!\n");
        }

        h_unit_ft[t].re_ft = (real*) malloc(  this_qhomenr * sizeof(real) ) ;
        memcpy(h_unit_ft[t].re_ft, ft_reEnv , this_qhomenr * sizeof(real) ) ;

        h_unit_ft[t].im_ft = (real*) malloc(  this_qhomenr * sizeof(real) ) ;
        memcpy(h_unit_ft[t].im_ft, ft_imEnv , this_qhomenr * sizeof(real) ) ;
    }

    memcpy (host_to_device_unit_ft, h_unit_ft, nTypes * sizeof(t_ft_GPU) ) ;

    for (t = 0 ; t < nTypes ; t++)
    {
        this_qhomenr = wt[t].qvecs->qhomenr ;

        fprintf(stderr, "Start copying unit FT data to all GPUs ...  \n" );

        gpuErrchk( cudaMalloc(&(host_to_device_unit_ft[t].re_ft) ,                      this_qhomenr * sizeof(real) ) ) ;
        gpuErrchk( cudaMemcpy(  host_to_device_unit_ft[t].re_ft  , h_unit_ft[t].re_ft,  this_qhomenr * sizeof(real), cudaMemcpyHostToDevice) );

        gpuErrchk( cudaMalloc(&(host_to_device_unit_ft[t].im_ft) ,                      this_qhomenr * sizeof(real) ) ) ;
        gpuErrchk( cudaMemcpy(  host_to_device_unit_ft[t].im_ft  , h_unit_ft[t].im_ft,  this_qhomenr * sizeof(real), cudaMemcpyHostToDevice) );

        fprintf(stderr, "... End copying unit FT data to all GPUs!  \n" );
    }

    /* Push the whole t_gpudata structure to the GPU */
    gpuErrchk( cudaMalloc(&d_unit_ft_alloc ,                           nTypes  *  sizeof(t_ft_GPU) ) );
    gpuErrchk( cudaMemcpy( d_unit_ft_alloc,   host_to_device_unit_ft , nTypes  *  sizeof(t_ft_GPU), cudaMemcpyHostToDevice ) );

    /* Copying FT data structure to global memory of GPU */
    gpuErrchk( cudaMemcpyToSymbol(d_unit_ft , &d_unit_ft_alloc ,   sizeof(t_ft_GPU*) )) ;
    gpuErrchk( cudaPeekAtLastError() );

    fprintf(stderr, "Unit FT - data copied to GPU!\n");

    sfree(h_unit_ft);
    sfree(host_to_device_unit_ft);
}

/*
 * Here the solvent FT is copied to GPU. This function will be called whenever the FT was recalculated in waxsmd.c .
 * Be aware that here the FT will not be recalculated but the values of the FT will be called by reference.
 */
void update_envelope_solvent_density_GPU(t_waxsrec *wr )
{

    int t ;
    t_waxsrecType *wt = wr->wt ;
    int nTypes = wr->nTypes ;
    int this_qhomenr, qstart ;

    real *ft_reEnv, *ft_imEnv ;

    rvec *q;

    t_ft_GPU *d_solvent_ft_alloc , *host_to_device_solvent_ft,  *h_solvent_ft  ;
    snew(host_to_device_solvent_ft, nTypes);
    snew(h_solvent_ft, nTypes) ;

    fprintf(stderr, " Start copying solvent FT data to GPU... \n " );

    /* prepare Data set on CPU, this will then be copied to GPU */
    for (t = 0 ; t < nTypes ; t++)
    {

        qstart  = wr->wt[t].qvecs->qstart;
        this_qhomenr = wr->wt[t].qvecs->qhomenr;
        q       = wr->wt[t].qvecs->q;
        gmx_envelope_solventFourierTransform(wt[t].envelope, q + qstart, this_qhomenr, FALSE, &ft_reEnv, &ft_imEnv);    /* Always FALSE because FT is already calculated in waxsmd.c before! */

        if (!ft_reEnv)
        {
            gmx_fatal(FARGS, "Reading of the Solvent Fourier Transform failed!\n");
        }

        h_solvent_ft[t].re_ft = (real*) malloc(  this_qhomenr * sizeof(real) ) ;
        memcpy(h_solvent_ft[t].re_ft, ft_reEnv , this_qhomenr * sizeof(real) ) ;

        h_solvent_ft[t].im_ft = (real*) malloc(  this_qhomenr * sizeof(real) ) ;
        memcpy(h_solvent_ft[t].im_ft, ft_imEnv , this_qhomenr * sizeof(real) ) ;
    }

    memcpy (host_to_device_solvent_ft, h_solvent_ft, nTypes * sizeof(t_ft_GPU) ) ;

    for (t = 0 ; t < nTypes ; t++)
    {
        this_qhomenr = wt[t].qvecs->qhomenr ;
        host_to_device_solvent_ft->qhomenr = this_qhomenr;

        gpuErrchk( cudaMalloc(&(host_to_device_solvent_ft[t].re_ft) ,                         this_qhomenr * sizeof(real) ) ) ;
        gpuErrchk( cudaMemcpy(  host_to_device_solvent_ft[t].re_ft  , h_solvent_ft[t].re_ft,  this_qhomenr * sizeof(real), cudaMemcpyHostToDevice) );

        gpuErrchk( cudaMalloc(&(host_to_device_solvent_ft[t].im_ft) ,                         this_qhomenr * sizeof(real) ) ) ;
        gpuErrchk( cudaMemcpy(  host_to_device_solvent_ft[t].im_ft  , h_solvent_ft[t].im_ft,  this_qhomenr * sizeof(real), cudaMemcpyHostToDevice) );
    }

    /* Push the whole t_gpudata structure to the GPU */
    gpuErrchk( cudaMalloc(&d_solvent_ft_alloc ,                              nTypes  *  sizeof(t_ft_GPU) ) );
    gpuErrchk( cudaMemcpy( d_solvent_ft_alloc,   host_to_device_solvent_ft , nTypes  *  sizeof(t_ft_GPU), cudaMemcpyHostToDevice ) );

    /* Copying data structure of FT to global memory of GPU */
    gpuErrchk( cudaMemcpyToSymbol(d_solvent_ft , &d_solvent_ft_alloc ,   sizeof(t_ft_GPU*) )) ;
    gpuErrchk( cudaPeekAtLastError() );
    fprintf(stderr, "... Solvent FT - data copied to GPU!\n");

    sfree(host_to_device_solvent_ft) ; sfree(h_solvent_ft) ;
}
