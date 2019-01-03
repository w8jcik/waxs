#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include"device_averages.cuh"

/* Update cumulative averages: new = fac1*sum + fac2*new */

__device__ t_complex d_cadd_d(t_complex a, t_complex b)
{
    t_complex c;

    c.re = a.re+b.re;
    c.im = a.im+b.im;

    return c;
}

__device__ t_complex d_cadd(t_complex a, t_complex b)
{
    t_complex c;

    c.re = a.re+b.re;
    c.im = a.im+b.im;

    return c;
}

__device__ t_complex d_cdiff(t_complex a , t_complex b)
{
    t_complex c;

    c.re = a.re - b.re;
    c.im = a.im - b.im;

    return c;
}

__device__ t_complex d_cmul_rd(t_complex a, t_complex b)
{
    t_complex c;

    c.re = a.re*b.re - a.im*b.im;
    c.im = a.re*b.im + a.im*b.re;

    return c;
}
__device__ t_complex d_cmul_d(t_complex a, t_complex b)
{
    t_complex c;

    c.re = a.re*b.re - a.im*b.im;
    c.im = a.re*b.im + a.im*b.re;

    return c;
}

__device__ t_complex d_rcmul(real r, t_complex c)
{
    t_complex d;
    d.re = r*c.re;
    d.im = r*c.im;

    return d;
}

__device__ t_complex d_rcmul_d(double r, t_complex c)
{
    t_complex d;
    d.re = r*c.re;
    d.im = r*c.im;

    return d;
}

__device__ real d_r_accum_avg( real sum, real new_value, real fac1, real fac2 )
{
    real tmp = fac1*sum + fac2*new_value;
    return tmp;
}

__device__ double d_accum_avg( double sum, double new_value, double fac1, double fac2 )
{
    double tmp = fac1*sum + fac2*new_value;
    return tmp;
}

__device__ t_complex d_c_accum_avg( t_complex sum, t_complex new_value, real fac1, real fac2 )
{
    t_complex tmp = d_cadd( d_rcmul(fac1,sum), d_rcmul(fac2,new_value) );
    return tmp;
}

__device__ t_complex d_cd_accum_avg( t_complex sum, t_complex new_value, double fac1, double fac2 )
{
    t_complex tmp = d_cadd_d( d_rcmul_d(fac1,sum), d_rcmul_d(fac2,new_value) );
    return tmp;
}

__device__ void d_cvec_accum_avg( cvec *sum, cvec new_value, real fac1, real fac2 )
{
    cvec tmp;
    int d;
    for (d=0 ; d<DIM; d++)
    {
        tmp [d] = d_c_accum_avg(*sum[d],new_value[d],fac1,fac2);
        *sum[d] = tmp[d];
    }
}
