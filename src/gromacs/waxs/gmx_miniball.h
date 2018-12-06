/* This is the Miniball implementation by Bernd Gaertner, received from
   http://www.inf.ethz.ch/personal/gaertner/miniball.html
   in November 2013

   The copyright statement is pasted below.

   The code was translated to C by Jochen Hub, November 2013
*/


//    Copright (C) 1999-2013, Bernd Gaertner
//    $Rev: 3581 $
//
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

//#include "visibility.h"
//#include "typedefs.h"
//#include "sysstuff.h"
//#include "gromacs/swax/sysstuff.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/pbcutil/pbc.h"

/* Here, specify if Miniball should run in single or double precision */
typedef double NT;


typedef struct gmx_miniball *gmx_miniball_t;

/* Point interators, Coordinate iterator, and Support iterator types */
typedef NT** Pit;
typedef NT*  Cit;
typedef Pit*   Sit;

/* Construct and destroy the smallest enclosing ball */
gmx_miniball_t
gmx_miniball_init(int d, Pit begin, int nPoints);

void
gmx_miniball_destroy(gmx_miniball_t *mb);

/* Get center, support points, etc */
NT*
gmx_miniball_center(gmx_miniball_t mb);

NT
gmx_miniball_squared_radius(gmx_miniball_t mb);

int
gmx_miniball_nr_support_points(gmx_miniball_t mb);

Sit
gmx_miniball_support_points_begin(gmx_miniball_t mb);
Sit
gmx_miniball_support_points_end(gmx_miniball_t mb);


// POST: returns the maximum excess of any input point w.r.t. the computed
//       ball, divided by the squared radius of the computed ball. The
//       excess of a point is the difference between its squared distance
//       from the center and the squared radius; Ideally, the return value
//       is 0. subopt is set to the absolute value of the most negative
//       coefficient in the affine combination of the support points that
//       yields the center. Ideally, this is a convex combination, and there
//       is no negative coefficient in which case subopt is set to 0.
NT
gmx_miniball_relative_error(gmx_miniball_t mb, NT *subopt);

// POST: return true if the relative error is at most tol, and the
//       suboptimality is 0; the tolerance is 10 times the
//       coordinate type's machine epsilon
gmx_bool
gmx_miniball_is_valid(gmx_miniball_t mb, NT tol);

// POST: returns the time in seconds taken by the constructor call for
//       computing the smallest enclosing ball
double
gmx_miniball_get_time(gmx_miniball_t mb);




