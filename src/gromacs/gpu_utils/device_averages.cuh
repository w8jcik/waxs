#include "gromacs/math/gmxcomplex.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"


__device__ t_complex d_cadd_d(t_complex a, t_complex b);

__device__ t_complex d_cadd(t_complex a, t_complex b);

__device__ static t_complex d_rcmul(real r, t_complex c);

__device__ t_complex d_rcmul_d(double r, t_complex c);

__device__ t_complex d_cmul_rd(t_complex a, t_complex b);

__device__ real d_r_accum_avg( real sum, real new_value, real fac1, real fac2 );

__device__ double d_accum_avg( double sum, double new_value, double fac1, double fac2 );

__device__ t_complex d_c_accum_avg( t_complex sum, t_complex new_value, real fac1, real fac2 );

__device__ t_complex d_cd_accum_avg( t_complex sum, t_complex new_value, double fac1, double fac2 );

__device__ void d_cvec_accum_avg( cvec *sum, cvec new_value, real fac1, real fac2 );
