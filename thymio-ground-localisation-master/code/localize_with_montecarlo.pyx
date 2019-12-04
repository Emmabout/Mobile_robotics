# -*- coding: utf-8 -*-
# cython: profile=True
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import pyximport; pyximport.install()
import numpy as np
import math
import bisect
import random
from libc.math cimport floor, sqrt, log, atan2, sin, cos, exp, pow
cimport numpy as np
cimport cython
import localize_common
cimport localize_common
from localize_common import rot_mat2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# some useful constants in local scope

cdef double _pi = math.pi
cdef double _1sqrt2pi = 1. / sqrt(2. * math.pi)

@cython.profile(False)
@cython.cdivision(True) # turn off division-by-zero checking
cdef double _norm(double x, double u, double s):
	cdef double factor = _1sqrt2pi / s
	cdef double dxus = (x - u) / s
	return factor * exp(- (dxus * dxus) / 2.)

@cython.profile(False)
cdef double _normalize_angle(double alpha):
	while alpha > _pi:
		alpha -= 2. * _pi
	while alpha < -_pi:
		alpha += 2. * _pi
	return alpha

@cython.profile(False)
cdef double _normalize_angle_0_2pi(double alpha):
	while alpha >= 2 * _pi:
		alpha -= 2. * _pi
	while alpha < 0:
		alpha += 2. * _pi
	return alpha

# main class

cdef class MCLocalizer(localize_common.AbstractLocalizer):

	# user parameters
	cdef int N_uniform

	# particles
	cdef double[:,:] particles # 2D array of particles_count x (x,y,theta)
	cdef double[:] estimated_particle # last best estimate

	def __init__(self, np.ndarray[double, ndim=2] ground_map_left, np.ndarray[double, ndim=2] ground_map_right, int particles_count, double sigma_obs, double prob_uniform, double alpha_xy, double alpha_theta):
		""" Create the localizer with the ground map and some parameters """

		super(MCLocalizer, self).__init__(ground_map_left, ground_map_right, alpha_xy, alpha_theta, sigma_obs)

		# setup parameters
		self.N_uniform = int(prob_uniform*particles_count)

		# create initial particles filled the whole space
		cdef np.ndarray[double, ndim=2] particles = np.random.uniform(0,1,[particles_count, 3])
		particles *= [ground_map_left.shape[0], ground_map_left.shape[1], _pi*2]
		self.particles = particles
		self.estimated_particle = np.empty([3], dtype=float)

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False) # turn off wrap-around checking
	@cython.nonecheck(False) # turn off none checks
	def apply_obs(self, double left_color, double right_color):
		""" Apply observation and resample """

		cdef int i, j
		cdef double theta
		cdef np.ndarray[double, ndim=2] R
		cdef np.ndarray[double, ndim=1] left_sensor_pos, right_sensor_pos
		cdef double left_weight, right_weight
		cdef double ground_val
		cdef double sigma = self.sigma_obs
		cdef int particles_count = self.particles.shape[0]
		cdef int resample_count = particles_count - self.N_uniform
		cdef int uniform_count = self.N_uniform
		cdef np.ndarray[double, ndim=2] particles_view = np.asarray(self.particles)
		cdef np.ndarray[double, ndim=1] weights = np.empty([particles_count])
		cdef np.ndarray[double, ndim=2] ground_map_left_view = np.asarray(self.ground_map_left)
		cdef np.ndarray[double, ndim=2] ground_map_right_view = np.asarray(self.ground_map_right)

		# matching particles
		nb_ok = 0
		# apply observation to every particle
		for i in range(particles_count):
			theta = particles_view[i, 2]

			# compute position of sensors in world coordinates
			R = rot_mat2(theta)
			left_sensor_pos = R.dot([7.2, 1.1]) + particles_view[i, 0:2]
			right_sensor_pos = R.dot([7.2, -1.1]) + particles_view[i, 0:2]

			if not self.is_in_bound(left_sensor_pos) or not self.is_in_bound(right_sensor_pos):
				# kill particle if out of map
				weights[i] = 0.
			else:
				# otherwise, compute weight in function of ground color

				# left sensor
				ground_val = ground_map_left_view[self.xyW2C(left_sensor_pos[0]), self.xyW2C(left_sensor_pos[1])]
				left_weight = _norm(left_color, ground_val, sigma)

				# right sensor
				ground_val = ground_map_right_view[self.xyW2C(right_sensor_pos[0]), self.xyW2C(right_sensor_pos[1])]
				right_weight = _norm(right_color, ground_val, sigma)

				# compute weight
				weights[i] = left_weight * right_weight

			# update matching particles
			if weights[i] > 0.5:
				nb_ok += 1

		# ratio matching particles
		print "  Proportion of matching particles:", 1.*nb_ok/len(weights)

		# resample
		assert weights.sum() > 0.
		weights /= weights.sum()
		cdef np.ndarray[double, ndim=2] resampled = particles_view[np.random.choice(particles_count, resample_count, p=weights)]
		cdef np.ndarray[double, ndim=2] new_particles = np.random.uniform(0,1,[uniform_count, 3]) * [self.ground_map_left.shape[0], self.ground_map_left.shape[1], _pi*2]
		particles_view[:resample_count] = resampled
		particles_view[resample_count:] = new_particles
		# FIXME I don't know why that doesn't work so I copy manually -_-
		#for i, p in enumerate(resampled):
			#for j, v in enumerate(p):
				#self.particles[i,j] = v
		#for i, p in enumerate(new_particles):
			#for j, v in enumerate(p):
				#self.particles[i+resample_count,j] = v
		# end FIXME

		# add adaptive noise to fight particle depletion
		cdef double one_N3 = 1. / pow(particles_count, 1./3.)
		cdef double range_x = self.ground_map_left.shape[0] * one_N3
		cdef double range_y = self.ground_map_left.shape[1] * one_N3
		cdef double range_theta = 2. * _pi * one_N3
		for i in range(particles_count):
			particles_view[i, 0] += np.random.uniform(-range_x / 2., range_x / 2.)
			particles_view[i, 1] += np.random.uniform(-range_y / 2., range_y / 2.)
			particles_view[i, 2] += np.random.uniform(-range_theta / 2., range_theta / 2.)


	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False) # turn off wrap-around checking
	@cython.nonecheck(False) # turn off none checks
	def apply_command(self, d_x, d_y, d_theta):
		""" Apply command to each particle """

		cdef int i
		cdef double theta
		cdef np.ndarray[double, ndim=1] d_xy = np.array([d_x, d_y])
		cdef np.ndarray[double, ndim=2] particles_view = np.asarray(self.particles)
		cdef int particles_count = self.particles.shape[0]

		# error model, same as with CPT, but without added half cell
		cdef double norm_xy = sqrt(d_x*d_x + d_y*d_y)
		cdef double e_theta = self.alpha_theta * math.fabs(d_theta) + math.radians(0.25)
		assert e_theta > 0, e_theta
		cdef double e_xy = self.alpha_xy * norm_xy + 0.01
		assert e_xy > 0, e_xy

		# apply command and sampled noise to each particle
		for i in range(particles_count):
			theta = particles_view[i, 2]
			particles_view[i, 0:2] += rot_mat2(theta).dot(d_xy) + np.random.normal(0, e_xy, [2])
			particles_view[i, 2] = theta + d_theta + np.random.normal(0, e_theta)


	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False) # turn off wrap-around checking
	@cython.nonecheck(False) # turn off none checks
	def estimate_state(self):

		# limits for considering participating to the state estimation
		cdef double theta_lim = math.radians(5)
		cdef double xy_lim = 1.5

		# RANSAC to find best index
		cdef int iterations_count = 500
		cdef int tests_count = 500
		cdef int index, o_index, best_index = -1
		cdef int support, best_support = 0
		cdef double x, y, theta
		cdef double o_x, o_y, o_theta
		cdef double dist_xy, dist_theta
		cdef np.ndarray[double, ndim=2] particles_view = np.asarray(self.particles)
		cdef int max_index = particles_view.shape[0]-1
		cdef int i, j
		cdef long [:] iteration_indices = np.random.randint(0, max_index, [iterations_count])
		cdef long [:] test_indices = np.random.randint(0, max_index, [tests_count])
		# tries a certain number of times
		for i in range(iterations_count):
			index = iteration_indices[i]
			x = particles_view[index, 0]
			y = particles_view[index, 1]
			theta = particles_view[index, 2]
			support = 0
			for j in range(tests_count):
				o_index = test_indices[j]
				o_x = particles_view[o_index, 0]
				o_y = particles_view[o_index, 1]
				o_theta = particles_view[o_index, 2]
				# compute distance
				dist_xy = sqrt((x-o_x)*(x-o_x) + (y-o_y)*(y-o_y))
				dist_theta = _normalize_angle(theta - o_theta)
				if dist_xy < xy_lim and dist_theta < theta_lim:
					support += 1
			# if it beats best, replace best
			if support > best_support:
				best_index = index
				best_support = support

		# then do the averaging for best index
		x = particles_view[best_index, 0]
		y = particles_view[best_index, 1]
		theta = particles_view[best_index, 2]
		cdef list sins = []
		cdef list coss = []
		cdef double xs = 0.
		cdef double ys = 0.
		cdef int count = 0
		cdef int conf_count = 0
		for j in range(tests_count):
			o_index = test_indices[j]
			o_x = particles_view[o_index, 0]
			o_y = particles_view[o_index, 1]
			o_theta = particles_view[o_index, 2]
			dist_xy = sqrt((x-o_x)*(x-o_x) + (y-o_y)*(y-o_y))
			dist_theta = _normalize_angle(theta - o_theta)
			if dist_xy < xy_lim and dist_theta < theta_lim:
				sins.append(sin(o_theta))
				coss.append(cos(o_theta))
				xs += o_x
				ys += o_y
				count += 1
			if dist_xy < self.conf_xy and dist_theta < self.conf_theta:
				conf_count += 1

		assert count > 0, count
		cdef double x_m = xs / count
		cdef double y_m = ys / count
		cdef double s_sins = sum(sins)
		cdef double s_coss = sum(coss)
		cdef double theta_m = atan2(s_sins, s_coss)
		self.estimated_particle[0] = x_m
		self.estimated_particle[1] = y_m
		self.estimated_particle[2] = theta_m

		return np.array([x_m, y_m, theta_m, float(conf_count) / float(tests_count)])


	def estimate_logratio(self, double x, double y, double theta):
		# TODO
		return 0


	# debug methods

	def dump_PX(self, str base_filename, float gt_x = -1, float gt_y = -1, float gt_theta = -1):
		""" Write particles to an image """
		fig = Figure((3,3), tight_layout=True)
		canvas = FigureCanvas(fig)
		ax = fig.gca()
		ax.set_xlim([0, self.ground_map_left.shape[0]])
		ax.set_ylim([0, self.ground_map_left.shape[1]])

		for (x, y, theta) in self.particles:
			ax.arrow(x, y, math.cos(theta), math.sin(theta), head_width=0.8, head_length=1, fc='k', ec='k', alpha=0.3)

		ax.arrow(gt_x, gt_y, math.cos(gt_theta)*2, math.sin(gt_theta)*2, head_width=1, head_length=1.2, fc='green', ec='green')

		ax.arrow(self.estimated_particle[0], self.estimated_particle[1], math.cos(self.estimated_particle[2])*2, math.sin(self.estimated_particle[2])*2, head_width=1, head_length=1.2, fc='blue', ec='blue')

		canvas.print_figure(base_filename+'.png', dpi=300)
