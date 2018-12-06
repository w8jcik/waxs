/* This is the Miniball implementation by Bernd Gaertner, received from
   http://www.inf.ethz.ch/personal/gaertner/miniball.html
   in November 2013

   The copyright statement is pasted below.

   The code was translated to C by Jochen Hub, November 2013
*/


//    Copright (C) 1999-2013, Bernd Gaertner
//    $Rev: 3581 $
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.

//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.

//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//    Contact:
//    --------
//    Bernd Gaertner
//    Institute of Theoretical Computer Science
//    ETH Zuerich
//    CAB G31.1
//    CH-8092 Zuerich, Switzerland
//    http://www.inf.ethz.ch/personal/gaertner


#include "gromacs/waxs/gmx_miniball.h"
//#include "gromacs/swax/sysstuff.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/math/vec.h"
#include "gromacs/utility/cstringutil.h"

// #define MINIBALL_DEBUG

#if defined( MINIBALL_DEBUG )
  #define DEBUG_PRINTF(x) printf x
#else
  #define DEBUG_PRINTF(x)
#endif


struct gmx_miniball{
    int d;
    NT **points_begin;
    NT **points_end;

    // NT *coord_accessor;  We don't this this in the C translation
    double time;

    // Zero
    NT nt0;

    //...for the algorithms
    /* L holds pointers to the support, that is, the points that currently form
       the surface of the sphere */
    Pit *L;
    Sit support_end;
    int fsize;   // number of forced points
    int ssize;   // number of support points

    // ...for the ball updates
    NT* current_c;
    NT  current_sqr_r;
    NT** c;
    NT* sqr_r;

    // helper arrays
    NT* q0;
    NT* z;
    NT* f;
    NT** v;
    NT** a;
};


/* private functions */
static void mtf_mb (gmx_miniball_t mb, Sit n);
static void mtf_move_to_front (gmx_miniball_t mb, Sit j);
static void pivot_mb (gmx_miniball_t mb, Pit n);
static void pivot_move_to_front (gmx_miniball_t mb, Pit j);
static NT excess (gmx_miniball_t mb, Pit pit);
static void pop (gmx_miniball_t mb);
static gmx_bool push (gmx_miniball_t mb, Pit pit);
static NT suboptimality (gmx_miniball_t mb);
static void create_arrays(gmx_miniball_t mb);
static void delete_arrays(gmx_miniball_t mb);

gmx_miniball_t
gmx_miniball_init(int d, Pit begin, int nPoints)
{
    gmx_miniball_t mb;
    int j;

    DEBUG_PRINTF(("\nMiniball: Starting gmx_miniball_init()\n"));

    snew(mb, 1);

    mb->d            = d;
    mb->points_begin = begin;
    mb->points_end   = begin+nPoints;
    mb->time          = clock();
    mb->nt0           = 0;
    mb->L             = NULL;
    mb->support_end   = NULL;
    mb->fsize = 0;
    mb->ssize = 0;
    mb->current_c = NULL;
    mb->current_sqr_r = -1;
    mb->c     = NULL;
    mb->sqr_r = NULL;
    mb->q0    = NULL;
    mb->z     = NULL;
    mb->f     = NULL;
    mb->v     = NULL;
    mb->a     = NULL;

    if (nPoints == 0)
    {
	gmx_fatal(FARGS, "No points to build minimum bounding sphere\n");
    }

    create_arrays(mb);

    // set initial center
    for (j=0; j<mb->d; ++j)
    {
	mb->c[0][j] = mb->nt0;
    }
    mb->current_c = mb->c[0];

    // compute miniball
    pivot_mb (mb, mb->points_end);

    // update time
    mb->time = (clock() - mb->time) / CLOCKS_PER_SEC;

    return mb;
}

void
gmx_miniball_destroy(gmx_miniball_t *mb)
{
    delete_arrays(*mb);
    sfree(*mb);
}

static void
create_arrays(gmx_miniball_t mb)
{
    int i, d = mb->d;

    snew(mb->c, d+1);
    snew(mb->v, d+1);
    snew(mb->a, d+1);
    for (i = 0; i < d+1; ++i)
    {
	snew(mb->c[i], d);
	snew(mb->v[i], d);
	snew(mb->a[i], d);
    }
    snew(mb->sqr_r, d+1);
    snew(mb->q0,    d);
    snew(mb->z,     d+1);
    snew(mb->f,     d+1);

    snew(mb->L, d+3);
    mb->support_end = mb->L;
}


static void
delete_arrays(gmx_miniball_t mb)
{
    int i, d = mb->d;

    sfree(mb->f);
    sfree(mb->z);
    sfree(mb->q0);
    for (i=0; i < d+1; ++i)
    {
	sfree(mb->a[i]);
	sfree(mb->v[i]);
	sfree(mb->c[i]);
    }
    sfree(mb->sqr_r);
    sfree(mb->a);
    sfree(mb->v);
    sfree(mb->c);
    sfree(mb->L);
}

NT*
gmx_miniball_center(gmx_miniball_t mb)
{
    return mb->current_c;
}

NT
gmx_miniball_squared_radius(gmx_miniball_t mb)
{
    return mb->current_sqr_r;
}

int
gmx_miniball_nr_support_points (gmx_miniball_t mb)
{
    if (! (mb->ssize < mb->d+2))
    {
	gmx_fatal(FARGS, "Inconsistency in gmx_miniball_nr_support_points()\n");
    }
    return mb->ssize;
}

Sit
gmx_miniball_support_points_begin (gmx_miniball_t mb)
{
    return mb->L;
}

Sit
gmx_miniball_support_points_end (gmx_miniball_t mb)
{
    return mb->support_end;
}


NT
gmx_miniball_relative_error (gmx_miniball_t mb, NT *subopt)
{
    NT e, max_e = mb->nt0;
    Pit i;
    Sit it;

    // compute maximum absolute excess of support points
    for (it = gmx_miniball_support_points_begin(mb);
	 it != gmx_miniball_support_points_end(mb); ++it)
    {
	e = excess (mb, *it);
	if (e < mb->nt0) e = -e;
	DEBUG_PRINTF(("gmx_miniball_relative_error(): e = %.15f\n", e));
	if (e > max_e) {
	    max_e = e;
	}
    }
    // compute maximum excess of any point
    for (i = mb->points_begin; i != mb->points_end; ++i)
    {
	if ((e = excess (mb, i)) > max_e)
	{
	    max_e = e;
	    DEBUG_PRINTF(("gmx_miniball_relative_error(): (all points) e = %.15f (for i = %d)\n", e,
			  (int)(i- mb->points_begin)));
	}
    }

    *subopt = suboptimality(mb);
    if (! (mb->current_sqr_r > mb->nt0 || max_e == mb->nt0))
    {
	gmx_fatal(FARGS, "Inconsistency in gmx_miniball_relative_error()\n");
    }
    DEBUG_PRINTF(("gmx_miniball_relative_error(): current_sqr_r = %.15lf\n", mb->current_sqr_r ));
    DEBUG_PRINTF(("gmx_miniball_relative_error(): max_e = %.15lf\n", max_e ));
    DEBUG_PRINTF(("gmx_miniball_relative_error(): nt=   = %.15lf\n", mb->nt0 ));

    return (mb->current_sqr_r == mb->nt0 ? mb->nt0 : max_e / mb->current_sqr_r);
  }


gmx_bool gmx_miniball_is_valid (gmx_miniball_t mb, NT tol)
{
    NT suboptimality, relerr;
    gmx_bool bValid;

    if (tol <= 0)
    {
	/* Note: 10 * EPS seem slighly too strict, as GMX_FLOAT_EPS is only half of
	   std::numeric_limits<NT>::epsilon() */
	if (sizeof (NT) == sizeof(float))
	    tol = 20*GMX_FLOAT_EPS;
	else
	    tol = 20*GMX_DOUBLE_EPS;
    }

    relerr = gmx_miniball_relative_error (mb, &suboptimality);
    bValid = ( (relerr <= tol) && (suboptimality <= tol ) );
    if (!bValid)
    {
        printf("\ngmx_miniball_is_valid (): Found invalid miniball:\n"
               "\trelative error = %g, tol = %g, suboptimality = %g\n", relerr, tol, suboptimality);
    }

    return ( (gmx_miniball_relative_error (mb, &suboptimality) <= tol) && (suboptimality <= tol) );
}


double
gmx_miniball_get_time(gmx_miniball_t mb)
{
    return mb->time;
}

static void
mtf_mb (gmx_miniball_t mb, Sit n)
{
    Sit j, i;
    int d = mb->d;

    // Algorithm 1: mtf_mb (L_{n-1}, B), where L_{n-1} = [L.begin, n)
    // B: the set of forced points, defining the current ball
    // S: the superset of support points computed by the algorithm
    // --------------------------------------------------------------
    // from B. Gaertner, Fast and Robust Smallest Enclosing Balls, ESA 1999,
    // http://www.inf.ethz.ch/personal/gaertner/texts/own_work/esa99_final.pdf

    //   PRE: B = S
    if (mb->fsize != mb->ssize)
    {
	gmx_fatal(FARGS, "Inconsistency in mtf_mb()\n");
    }

    mb->support_end = mb->L;
    if ((mb->fsize) == d+1) return;

    // incremental construction
    for (i = mb->L; i != n;)
    {
	// INV: (support_end - L.begin() == |S|-|B|)
	if (! ((mb->support_end - mb->L) == mb->ssize - mb->fsize))
	{
	    gmx_fatal(FARGS, "Inconsistency in mtf_mb()\n");
	}

	j = i++;
	if (excess(mb, *j) > mb->nt0)
	    if (push(mb, *j)) {              // B := B + p_i
		mtf_mb (mb, j);          // mtf_mb (L_{i-1}, B + p_i)
		pop(mb);                 // B := B - p_i
		mtf_move_to_front(mb, j);
	    }
    }
    // POST: the range [L.begin(), support_end) stores the set S\B
}


static void
mtf_move_to_front (gmx_miniball_t mb, Sit j)
{
    Pit tmp;
    Sit i;
    int jpos;

    if (mb->L == NULL)
	snew(mb->L, mb->d+3);

    if (mb->support_end == j)
    {
	mb->support_end++;
	DEBUG_PRINTF(("mtf_move_to_front: support_end increased.\n"));
    }

    /* This is a substitute for L.splice (L.begin(), L, j); */
    /* Get position of j in list L */
    jpos = 0;
    for (i = mb->L; i !=j; i++)
    {
	jpos++;
    }

    tmp = mb->L[jpos];

    /* If jpos is at least at support_end, then we insert an element between
       L and support_end, so support_end moves to the right */
    if (jpos >= (int)(mb->support_end-mb->L))
	mb->support_end++;

    if (jpos >= mb->d+2)
	gmx_fatal(FARGS, "Inconsistency error in mtf_move_to_front ()\n");

    /* more this element to the front */
    for (i = mb->L+jpos; i != mb->L; i--)
    {
	*i = *(i-1);
    }
    *mb->L = tmp;

    // L.splice (L.begin(), L, j);

    DEBUG_PRINTF(("END mtf_move_to_front support_end -L = %d\n", (int)(mb->support_end - mb->L)));
}


static void
pivot_mb (gmx_miniball_t mb, Pit n)
{
    // Algorithm 2: pivot_mb (L_{n-1}), where L_{n-1} = [L.begin, n)
    // --------------------------------------------------------------
    // from B. Gaertner, Fast and Robust Smallest Enclosing Balls, ESA 1999,
    // http://www.inf.ethz.ch/personal/gaertner/texts/own_work/esa99_final.pdf
    NT          old_sqr_r;
    const NT*   c;
    Pit         pivot, k;
    NT          e, max_e, sqr_r, tmp;
    Cit p;
    int j;

    int iloop = 0;
    do {
	DEBUG_PRINTF(("\n*** pivot_mb loop %d\n", iloop++));

	old_sqr_r = mb->current_sqr_r;
	sqr_r = mb->current_sqr_r;

	pivot = mb->points_begin;
	max_e = mb->nt0;
	for (k = mb->points_begin; k != n; ++k)
	{
	    //p = coord_accessor(k);
	    p = *k;  /* Jochen */
	    e = -sqr_r;
	    c = mb->current_c;
	    for (j = 0; j < mb->d; ++j)
	    {
		tmp = (*p++-*c++);
		e += tmp*tmp;
	    }
	    if (e > max_e)
	    {
		max_e = e;
		pivot = k;
	    }
	}

	Sit i;
	if (max_e > mb->nt0) {
	    // check if the pivot is already contained in the support set
	    for (i = mb->L; i != mb->support_end; i++)
	    {
		if (*i == pivot)
		{
		    break;
		}
	    }

	    if (i == mb->support_end)
	    {
		if (mb->fsize != 0)
		    gmx_fatal(FARGS, "Inconsistency in pivot_mb()\n");
		if (push (mb, pivot))
		{
		    mtf_mb(mb, mb->support_end);
		    pop(mb);
		    pivot_move_to_front(mb, pivot);
		}
	    }
	}
	DEBUG_PRINTF(("Leave pivot_mb loop? %g >= %g Result = %d\n", old_sqr_r, mb->current_sqr_r, old_sqr_r >= mb->current_sqr_r));
    } while (old_sqr_r < mb->current_sqr_r);
}



static void
pivot_move_to_front (gmx_miniball_t mb, Pit j)
{
    int sizeNow = mb->support_end-mb->L;
    Sit c;

    DEBUG_PRINTF(("BEGIN pivot_move_to_front support-end - L = %d\n",  (int) (mb->support_end - mb->L)));

    mb->support_end++;
    if (mb->support_end - mb->L > 0)
    {
	for (c = mb->support_end-1; c != mb->L; c--)
	{
	    *c = *(c-1);
	}
    }
    *mb->L = j;
    if (mb->support_end - mb->L == mb->d+2)
	mb->support_end--;

    DEBUG_PRINTF(("END pivot_move_to_front: support end = %d\n", (int) (mb->support_end - mb->L)));

}

static NT
excess (gmx_miniball_t mb, Pit pit)
{
    int k;
    Cit p = *pit;
    NT e = -mb->current_sqr_r, tmp;
    NT* c = mb->current_c;

    for (k=0; k<mb->d; ++k)
    {
	tmp = (*p++-*c++);
	e += tmp*tmp;
    }
    return e;
}


static void
pop (gmx_miniball_t mb)
{
    --mb->fsize;
}


static gmx_bool
push (gmx_miniball_t mb, Pit pit)
{
    int i, j, d = mb->d;
    NT eps = GMX_FLOAT_EPS;

    Cit cit = *pit;
    Cit p = cit;

    DEBUG_PRINTF(("Before push(): sqr_r[fsize] = %g, fsize = %d\n", mb->sqr_r[mb->fsize], mb->fsize));
    if (mb->fsize==0)
    {
	for (i=0; i<d; ++i)
	    mb->q0[i] = *p++;
	for (i=0; i<d; ++i)
	    mb->c[0][i] = mb->q0[i];
	mb->sqr_r[0] = mb->nt0;
    }
    else
    {
	// set v_fsize to Q_fsize
	for (i=0; i<d; ++i)
	    //v[fsize][i] = p[i]-q0[i];
	    mb->v[mb->fsize][i] = *p++ - mb->q0[i];

	// compute the a_{fsize,i}, i< fsize
	for (i=1; i<mb->fsize; ++i)
	{
	    mb->a[mb->fsize][i] = mb->nt0;
	    for (j=0; j<d; ++j)
		mb->a[mb->fsize][i] += mb->v[i][j] * mb->v[mb->fsize][j];
	    mb->a[mb->fsize][i] *= (2/mb->z[i]);
      }

      // update v_fsize to Q_fsize-\bar{Q}_fsize
      for (i=1; i<mb->fsize; ++i) {
	for (j=0; j<d; ++j)
	    mb->v[mb->fsize][j] -= mb->a[mb->fsize][i]*mb->v[i][j];
      }

      // compute z_fsize
      mb->z[mb->fsize] = mb->nt0;
      for (j=0; j<d; ++j)
	mb->z[mb->fsize] += (mb->v[mb->fsize][j])*(mb->v[mb->fsize][j]);
      mb->z[mb->fsize]*=2;

      // reject push if z_fsize too small
      if (mb->z[mb->fsize] < eps*mb->current_sqr_r) {
	return FALSE;
      }

      // update c, sqr_r
      p=cit;
      NT tmp, e = -mb->sqr_r[mb->fsize-1];
      for (i=0; i<d; ++i)
      {
	  tmp = (*p++ - mb->c[mb->fsize-1][i]);
	  e += tmp*tmp;
      }
      mb->f[mb->fsize] = e/mb->z[mb->fsize];

      for (i=0; i<d; ++i)
	  mb->c[mb->fsize][i] = mb->c[mb->fsize-1][i] + mb->f[mb->fsize] * mb->v[mb->fsize][i];

      mb->sqr_r[mb->fsize] = mb->sqr_r[mb->fsize-1] + e * mb->f[mb->fsize]/2;
    }
    mb->current_c = mb->c[mb->fsize];
    DEBUG_PRINTF(("After push(): sqr_r[fsize] = %g, fsize = %d\n", mb->sqr_r[mb->fsize], mb->fsize));
    mb->current_sqr_r = mb->sqr_r[mb->fsize];
    mb->ssize = ++mb->fsize;
    return TRUE;
  }


NT
suboptimality (gmx_miniball_t mb)
{
    int i, k;
    NT* l;
    NT min_l = mb->nt0;

    snew(l, mb->d+1);
    l[0] = 1;
    for (i = mb->ssize-1; i>0; --i)
    {
	l[i] = mb->f[i];
	for (k = mb->ssize-1; k>i; --k)
	    l[i] -= mb->a[k][i]*l[k];
	if (l[i] < min_l) min_l = l[i];
	  l[0] -= l[i];
    }
    if (l[0] < min_l) min_l = l[0];
    if (min_l < mb->nt0)
	return -min_l;
    sfree(l);

    return mb->nt0;
}

