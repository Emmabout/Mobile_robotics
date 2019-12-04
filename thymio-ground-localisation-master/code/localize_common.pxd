# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import numpy as np
cimport numpy as np
cimport cython

# support functions

cpdef np.ndarray[double, ndim=2] rot_mat2(double angle)

# main class

cdef class AbstractLocalizer:

	# map
	cdef double[:,:] ground_map_left
	cdef double[:,:] ground_map_right

	# motion model
	cdef double alpha_xy
	cdef double alpha_theta

	# observation model
	cdef double sigma_obs

	# distance until which prob. mass is considered for confidence computation
	cdef double conf_theta
	cdef double conf_xy

	# support methods
	cpdef bint is_in_bound_cell(self, int x, int y)
	cpdef bint is_in_bound(self, double[:] pos)
	cpdef double xyC2W(self, int pos)
	cpdef int xyW2C(self, double pos)
	cpdef double dxyC2W(self, int dpos)
	cpdef int dxyW2C(self, double dpos)
