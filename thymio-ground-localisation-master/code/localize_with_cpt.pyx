# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import pyximport; pyximport.install()
import numpy as np
import math
import scipy.misc
from scipy.stats import norm
from libc.math cimport floor, sqrt, log, exp
cimport numpy as np
cimport cython
import localize_common
cimport localize_common
from localize_common import rot_mat2

# some useful constants in local scope

cdef double _pi = math.pi
cdef double _1pi = 1. / math.pi
cdef double _1sqrt2pi = 1. / sqrt(2. * math.pi)

# support functions

# taken from http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
def _norm_pdf_multivariate(x, mu, sigma):
	""" multivariate PDF for Gaussian """
	size = len(x)
	if size == len(mu) and (size, size) == sigma.shape:
		det = np.linalg.det(sigma)
		if det == 0:
			raise NameError("The covariance matrix can't be singular")

		norm_const = 1.0/ ( math.pow((2*_pi),float(size)/2) * math.pow(det,1.0/2) )
		x_mu = np.matrix(x - mu)
		inv = np.linalg.inv(sigma)
		try:
			result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
		except ArithmeticError as e:
			print 'Error computing Gaussian with following parameters'
			print 'x_mu', x_mu
			print 'inv_sigma', inv
			raise e
		return norm_const * result
	else:
		raise NameError("The dimensions of the input don't match")

@cython.cdivision(True) # turn off division-by-zero checking
cdef double _norm(double x, double u, double s):
	cdef double factor = _1sqrt2pi / s
	cdef double dxus = (x - u) / s
	return factor * exp(- (dxus * dxus) / 2.)

# main class

cdef class CPTLocalizer(localize_common.AbstractLocalizer):

	# user parameters
	cdef int angle_N
	cdef double max_prob_error
	cdef double prob_uniform

	# fixed/computed parameters
	cdef int N

	# probability distribution for latent space
	cdef double[:,:,:] PX
	cdef long [:] estimated

	# constructor

	def __init__(self, np.ndarray[double, ndim=2] ground_map_left, np.ndarray[double, ndim=2] ground_map_right, int angle_N, double sigma_obs, double max_prob_error, double prob_uniform, double alpha_xy, double alpha_theta):
		""" Fill the tables obs_left/right_black/white of the same resolution as the ground_map and an angle discretization angle_N """

		super(CPTLocalizer, self).__init__(ground_map_left, ground_map_right, alpha_xy, alpha_theta, sigma_obs)

		# copy parameters
		assert angle_N != 0
		self.angle_N = angle_N
		self.max_prob_error = max_prob_error
		self.prob_uniform = prob_uniform
		self.N = angle_N * ground_map_left.shape[0] * ground_map_left.shape[1]

		# initialize PX
		cdef shape = [angle_N, ground_map_left.shape[0], ground_map_left.shape[1]]
		self.PX = np.ones(shape, np.double) / float(np.prod(shape))
		self.estimated = np.zeros([3], dtype=int)


	# main methods

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.cdivision(True) # turn off division-by-zero checking
	def apply_obs(self, double left_color, double right_color):
		""" Update the latent space with observation """

		# create a view on the array to perform numpy operations such as *= or /=
		cdef np.ndarray[double, ndim=3] PX_view = np.asarray(self.PX)

		# parameters copy
		cdef int w = self.ground_map_left.shape[0]
		cdef int h = self.ground_map_left.shape[1]
		cdef int i, j, k
		cdef np.ndarray[double, ndim=2] R
		cdef np.ndarray[double, ndim=1] shift_left, shift_right
		cdef double c_x, c_y
		cdef int x_i, y_i
		cdef double d_obs
		cdef double sigma = self.sigma_obs
		cdef double ground_val
		cdef double lowest_prob_upate_left = _norm(np.max(self.ground_map_left), np.min(self.ground_map_left), sigma)
		cdef double lowest_prob_upate_right = _norm(np.max(self.ground_map_right), np.min(self.ground_map_right), sigma)

		# iterate on all angles
		for i in range(self.angle_N):
			# compute deltas
			R = rot_mat2(self.thetaC2W(i))
			shift_left = R.dot([7.2, 1.1])
			shift_right = R.dot([7.2, -1.1])
			# iterate on all positions
			for j in range(w):
				for k in range(h):
					c_x = self.xyC2W(j)
					c_y = self.xyC2W(k)
					# left sensor
					x_i = self.xyW2C(c_x + shift_left[0])
					y_i = self.xyW2C(c_y + shift_left[1])
					if self.is_in_bound_cell(x_i, y_i):
						# update PX
						PX_view[i, j, k] *= \
							_norm(left_color, self.ground_map_left[x_i, y_i], sigma)
					else:
						PX_view[i, j, k] *= lowest_prob_upate_left
					# right sensor
					x_i = self.xyW2C(c_x + shift_right[0])
					y_i = self.xyW2C(c_y + shift_right[1])
					if self.is_in_bound_cell(x_i, y_i):
						# update PX
						PX_view[i, j, k] *= \
							_norm(right_color, self.ground_map_right[x_i, y_i], sigma)
					else:
						PX_view[i, j, k] *= lowest_prob_upate_right

		# adding a bit of uniform and renormalize
		PX_view *= (1-self.prob_uniform) / PX_view.sum()
		PX_view += self.prob_uniform / self.N

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	# we have to keep this check, as if turned off there is some segfault in float to double conversion, I assume it is a bug in Cython's optimiser
	#@cython.cdivision(True) # turn off division-by-zero checking
	@cython.wraparound(False) # turn off wrap-around checking
	@cython.nonecheck(False) # turn off
	def apply_command(self, double d_x, double d_y, double d_theta):
		""" Apply a command for a displacement of d_x,d_y (in local frame) and a rotation of d_theta """

		# variables
		cdef int i, j, k                 # outer loops indices
		cdef int d_i, d_j, d_k           # inner loops indices
		cdef double s_theta, s_x, s_y    # source pos in world coordinates
		cdef double t_theta, t_x, t_y    # center target pos in world coordinates (mean)
		cdef int u_theta_i, u_x_i, u_y_i # current target pos in cell coordinates
		cdef double d_x_r, d_y_r         # rotated d_x and d_y in function of theta in world coordinates
		cdef int d_theta_i, d_x_r_i, d_y_r_i     # rotated d_x and d_y in function of theta in cell coordinates
		cdef double d_theta_d, d_x_r_d, d_y_r_d  # diff between d_x/y_r and d_x/y_r_i
		cdef np.ndarray[double, ndim=2] T, sigma # cov. motion model

		# assertion and copy some values for optimisation
		assert self.ground_map_left is not None
		assert self.ground_map_right is not None
		assert self.PX is not None
		assert self.ground_map_left.shape[0] == self.PX.shape[1]
		assert self.ground_map_left.shape[1] == self.PX.shape[2]
		assert self.ground_map_right.shape[0] == self.PX.shape[1]
		assert self.ground_map_right.shape[1] == self.PX.shape[2]
		cdef int w = self.ground_map_left.shape[0]
		cdef int h = self.ground_map_right.shape[1]
		cdef int angle_N = self.angle_N

		# error model for motion, inspired from
		# http://www.mrpt.org/tutorials/programming/odometry-and-motion-models/probabilistic_motion_models/
		# sum of factors from translation (x,y), rotation (theta), and half a cell (for sampling issues)
		cdef double norm_xy = sqrt(d_x*d_x + d_y*d_y)
		cdef double e_theta = self.alpha_theta * math.fabs(d_theta) + self.dthetaC2W(1) / 2.
		assert e_theta > 0, e_theta
		cdef double e_xy = self.alpha_xy * norm_xy + self.dxyC2W(1) / 2.
		assert e_xy > 0, e_xy
		cdef np.ndarray[double, ndim=2] e_xy_mat = np.array([[e_xy, 0], [0, e_xy]])

		# special case if e_xy is huge, robot is most likely lost

		# compute how many steps around we have to compute to have less than 1 % of error in transfering probability mass
		# and allocate arrays for fast lookup
		# for theta
		cdef object e_theta_dist = norm(0, e_theta)
		cdef double e_theta_max = e_theta_dist.pdf(0)
		cdef double e_i
		#print 'e_theta', e_theta
		#print 'e_theta_max', e_theta_max
		for i in range(1, angle_N/2):
			e_i = e_theta_dist.pdf(self.dthetaC2W(i))
			if e_i < self.max_prob_error * e_theta_max:
				break
		cdef int d_theta_range = i
		cdef int d_theta_shape = i*2 + 1
		cdef np.ndarray[double, ndim=1] e_theta_p = np.empty(d_theta_shape, np.double)

		# for x,y
		cdef object e_xy_dist = norm(0, e_xy)
		cdef double e_xy_max = e_xy_dist.pdf(0)
		for i in range(1, self.ground_map_left.shape[0]/2):
			e_i = e_xy_dist.pdf(self.dxyC2W(i))
			if e_i < self.max_prob_error * e_xy_max:
				break
		cdef int d_xy_range = i
		cdef int d_xy_shape = i*2 + 1
		cdef np.ndarray[double, ndim=2] e_xy_p = np.empty([d_xy_shape, d_xy_shape], np.double)

		# pre-compute values for fast lookup in inner loop for theta
		d_theta_i = self.dthetaW2C(d_theta)
		d_theta_d = d_theta - self.dthetaC2W(d_theta_i)
		e_theta_dist = norm(d_theta_d, e_theta)
		for i in range(e_theta_p.shape[0]):
			e_theta_p[i] = e_theta_dist.pdf(self.dthetaC2W(i - e_theta_p.shape[0]/2))
		e_theta_p /= e_theta_p.sum()
		#print e_theta_p
		#print d_theta_i, d_theta_d

		# view to remove some safety checks in the inner-most loop
		cdef np.ndarray[double, ndim=3] PX_view = np.asarray(self.PX)
		# temporary storage for probability mass
		cdef np.ndarray[double, ndim=3] PX_new = np.zeros(np.asarray(self.PX).shape, np.double)

		# mass probability transfer loops, first iterate on theta on source cells
		for i in range(angle_N):
			# change in angle
			s_theta = self.thetaC2W(i)
			t_theta = s_theta + d_theta
			# rotation matrix for theta
			T = rot_mat2(t_theta)

			# compute displacement for this theta
			d_x_r, d_y_r = T.dot([d_x, d_y])
			d_x_r_i = self.dxyW2C(d_x_r)
			d_y_r_i = self.dxyW2C(d_y_r)
			d_x_r_d = d_x_r - self.dxyW2C(d_x_r_i)
			d_y_r_d = d_y_r - self.dxyW2C(d_y_r_i)

			# compute covariance
			sigma = T.dot(e_xy_mat).dot(T.transpose())

			# then pre-compute arrays for fast lookup in inner loop for x,y
			#mu = np.array([d_x_r_d, d_y_r_d])
			e_x_dist = norm(d_x_r_d, e_xy)
			e_y_dist = norm(d_y_r_d, e_xy)
			#print 'mu', mu
			for j in range(e_xy_p.shape[0]):
				for k in range(e_xy_p.shape[1]):
					t_x = self.dxyC2W(j - e_xy_p.shape[0]/2)
					t_y = self.dxyC2W(k - e_xy_p.shape[1]/2)
					#e_xy_p[j,k] = _norm_pdf_multivariate(np.array([t_x, t_y]), mu , sigma)
					e_xy_p[j,k] = e_x_dist.pdf(t_x) * e_y_dist.pdf(t_y)
			e_xy_p /= e_xy_p.sum()
			#print e_xy_p
			#scipy.misc.imsave('/tmp/toto/e_xy_p-'+str(i)+'.png', e_xy_p)

			# outer loops for x,y iterating on source cells
			for j in range(w):
				for k in range(h):
					# inner loops iterating on target cells
					for d_i in range(d_theta_shape):
						u_theta_i = i + d_theta_i + d_i - d_theta_range
						for d_j in range(d_xy_shape):
							u_x_i = j + d_x_r_i + d_j - d_xy_range
							if u_x_i >= 0 and u_x_i < w:
								for d_k in range(d_xy_shape):
									u_y_i = k + d_y_r_i + d_k - d_xy_range
									if u_y_i >= 0 and u_y_i < h:
										u_theta_i = (u_theta_i + angle_N) % angle_N
										# copy probability mass
										PX_new[u_theta_i, u_x_i, u_y_i] += PX_view[i, j, k] * e_theta_p[d_i] * e_xy_p[d_j, d_k]

		# copy back probability mass
		self.PX = PX_new

	def estimate_state(self):
		""" return a (x,y,theta,self_confidence) numpy array representing the estimated state """

		# find best index
		cdef int x_i, y_i, theta_i
		theta_i, x_i, y_i = np.unravel_index(np.asarray(self.PX).argmax(), (<object>self.PX).shape)
		self.estimated[0] = x_i
		self.estimated[1] = y_i
		self.estimated[2] = theta_i

		# compute probability mass around it
		# first see which range we must explore
		cdef int best_range_theta = 0
		cdef double theta_half_CinW = self.dthetaC2W(1) / 2.
		if theta_half_CinW < self.conf_theta:
			best_range_theta = self.dthetaW2C(self.conf_theta - theta_half_CinW)
		cdef int best_range_xy = 0
		cdef double xy_half_CinW = self.dxyC2W(1) / 2.
		if xy_half_CinW < self.conf_xy:
			best_range_xy = self.dxyW2C(self.conf_xy - xy_half_CinW)
		# then sum around it
		cdef int i, j, k
		cdef int w = self.ground_map_left.shape[0]
		cdef int h = self.ground_map_left.shape[1]
		cdef double sum_p = 0.
		for i in range(theta_i - best_range_theta, theta_i + best_range_theta + 1):
			i = (i + self.angle_N) % self.angle_N
			for j in range(x_i - best_range_xy, x_i + best_range_xy + 1):
				if j >= 0 and j < w:
					for k in range(y_i - best_range_xy, y_i + best_range_xy + 1):
						if k >= 0 and k < h:
							sum_p += self.PX[i,j,k]
		# then divide by all probability mass
		cdef double confidence = sum_p / np.asarray(self.PX).sum()

		return np.array([self.xyC2W(x_i), self.xyC2W(y_i), self.thetaC2W(theta_i), confidence])

	def estimate_logratio(self, double x, double y, double theta):
		""" return the log ratio between the probability at estimate and at given location (x,y,theta).
		No bound check is performed on input """
		log_estimate = log(np.asarray(self.PX).max())
		log_query = log(self.PX[self.thetaW2C(theta), self.xyW2C(x), self.xyW2C(y)])
		return log_estimate - log_query


	# debug methods

	def dump_obs_model(self, str base_filename):
		""" Write images of observation model """
		cdef int i
		for i in range(self.angle_N):
			scipy.misc.imsave(base_filename+'-'+str(i)+'-left_black.png', self.obs_left_black[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-left_white.png', self.obs_left_white[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-right_black.png', self.obs_right_black[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-right_white.png', self.obs_right_white[i])

	def dump_PX(self, str base_filename, float gt_x = -1, float gt_y = -1, float gt_theta = -1):
		""" Write images of latent space """

		# dump image in RGB
		def write_image(np.ndarray[double, ndim=2] array_2D, str filename):
			cdef np.ndarray[double, ndim=3] zeros = np.zeros([self.PX.shape[1], self.PX.shape[2], 1], np.double)
			array_rgb = np.concatenate((array_2D[:,:,np.newaxis], array_2D[:,:,np.newaxis], array_2D[:,:,np.newaxis]), axis = 2)
			# ground truth
			cdef int i_x = self.xyW2C(gt_x)
			cdef int i_y = self.xyW2C(gt_y)
			if self.is_in_bound_cell(i_x, i_y):
				#array_rgb[i_x,i_y,1] = array_rgb[i_x,i_y,0]
				#array_rgb[i_x,i_y,2] = array_rgb[i_x,i_y,0]
				max_value = array_2D.max()
				array_rgb[i_x,i_y,:] = [0,max_value,0]
			else:
				print 'WARNING: ground-truth position {},{} is outside map bounds'.format(gt_x, gt_y)
			# estimated position
			array_rgb[self.estimated[0],self.estimated[1],:] = [0,0,max_value]
			scipy.misc.imsave(filename, array_rgb)

		# for every angle
		#cdef int i
		#for i in range(self.angle_N):
		#	write_image(np.asarray(self.PX[i]), base_filename+'-'+str(i)+'.png')

		# and the sum
		write_image(np.asarray(self.PX).sum(axis=0), base_filename+'-sum.png')

	def get_PX(self):
		return np.asarray(self.PX)

	# support methods

	cpdef double thetaC2W(self, int angle):
		""" Transform an angle in cell coordinates into an angle in radian """
		return ((angle+0.5) * 2. * _pi) / self.angle_N

	cpdef int thetaW2C(self, double angle):
		""" Transform an angle in radian into an angle in cell coordinates """
		return int(floor((angle * self.angle_N) / (2. * _pi))) % self.angle_N

	cpdef double dthetaC2W(self, int dangle):
		""" Transform an angle difference in cell coordinates into a difference in radian """
		return ((dangle) * 2. * _pi) / self.angle_N

	cpdef int dthetaW2C(self, double dangle):
		""" Transform an angle difference in radian into a difference in cell coordinates """
		return int(round((dangle * self.angle_N) / (2. * _pi)))

