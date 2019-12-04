#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

# note: encode videos with
#       mencoder mf://PX-*-B_*-sum.png -ovc lavc -o sum.mp4

import pyximport; pyximport.install()
import numpy as np
import localize_with_cpt
import localize_with_montecarlo
from localize_with_cpt import rot_mat2
import math
from termcolor import colored
import argparse
import sys
import os.path
import scipy.misc
import time

# support functions

def sensors_from_pos(x, y, theta):
	R = rot_mat2(theta)
	return R.dot([7.2, 1.1]) + [x,y], R.dot([7.2, -1.1]) + [x,y]

def normalize_sensor(v):
	return (float(v) - 60.) / 690.

def normalize_angle(alpha):
	while alpha > math.pi:
		alpha -= 2 * math.pi
	while alpha < -math.pi:
		alpha += 2 * math.pi
	return alpha

def create_localizer(ground_map, args):
	if args.ml_angle_count:
		return localize_with_cpt.CPTLocalizer(ground_map, ground_map, args.ml_angle_count, args.sigma_obs, args.max_prob_error, args.prob_uniform, args.alpha_xy, args.alpha_theta)
	elif args.mcl_particles_count:
		return localize_with_montecarlo.MCLocalizer(ground_map, ground_map, args.mcl_particles_count, args.sigma_obs, args.prob_uniform, args.alpha_xy, args.alpha_theta)
	else:
		print 'You must give either one of --ml_angle_count or --mcl_particles_count argument to this program'
		sys.exit(1)

def dump_error(localizer, i, duration, text, gt_x, gt_y, gt_theta, performance_log = None):
	estimated_state = localizer.estimate_state()
	dist_xy = np.linalg.norm(estimated_state[0:2]-(gt_x,gt_y))
	dist_theta = math.degrees(normalize_angle(estimated_state[2]-gt_theta))
	logratio_P = localizer.estimate_logratio(gt_x, gt_y, gt_theta)
	confidence = estimated_state[3]
	if abs(dist_xy) < math.sqrt(2)*2 and abs(dist_theta) < 15:
		color = 'green'
	elif abs(dist_xy) < math.sqrt(2)*4 and abs(dist_theta) < 30:
		color = 'yellow'
	else:
		color = 'red'
	print colored('{} {} - x,y dist: {}, theta dist: {}, log ratio P: {}, duration: {} [s], confidence: {:.2%}'.format(i, text, dist_xy, dist_theta, logratio_P, duration, confidence), color)
	if performance_log:
		performance_log.write('{} {} {} {} {} {} {} {} {} {} {} {}\n'.format(\
			i, \
			duration, \
			gt_x, \
			gt_y, \
			gt_theta, \
			estimated_state[0], \
			estimated_state[1], \
			estimated_state[2], \
			dist_xy, \
			dist_theta, \
			logratio_P,
			confidence
		))


# main test function

def test_command_sequence(ground_map, localizer, sequence):
	""" Evaluate a sequence of positions (x,y,theta) """

	# initialise position
	x, y, theta = sequence.next()

	for i, (n_x, n_y, n_theta) in enumerate(sequence):
		# observation
		# get sensor values from gt
		lpos, rpos = sensors_from_pos(x, y, theta)
		lval = ground_map[localizer.xyW2C(lpos[0]), localizer.xyW2C(lpos[1])]
		rval = ground_map[localizer.xyW2C(rpos[0]), localizer.xyW2C(rpos[1])]
		# apply observation
		start_time = time.time()
		localizer.apply_obs(lval, rval)
		duration = time.time() - start_time
		if args.debug_dump:
			localizer.dump_PX(os.path.join(args.debug_dump, 'PX-{:0>4d}-A_obs'.format(i)), x, y, theta)
		# compare observation with ground truth before movement
		dump_error(localizer, i, duration, "after obs", x, y, theta)

		# compute movement
		d_theta = n_theta - theta
		d_x, d_y = rot_mat2(-theta).dot([n_x-x, n_y-y])
		# do movement
		x, y, theta =  n_x, n_y, n_theta
		start_time = time.time()
		localizer.apply_command(d_x, d_y, d_theta)
		duration = time.time() - start_time
		if args.debug_dump:
			localizer.dump_PX(os.path.join(args.debug_dump, 'PX-{:0>4d}-B_mvt'.format(i)), x, y, theta)
		# compare observation with ground truth after movement
		dump_error(localizer, i, duration, "after mvt", x, y, theta)


def self_test(args):
	""" Self tests """

	# trajectory generators

	def traj_linear_on_x(x_start, x_end, delta_x, y, d_t):
		""" generator for linear trajectory on x """
		for x in np.arange(x_start, x_end, delta_x):
			yield x, y, 0.

	def traj_linear_on_y(y_start, y_end, delta_y, x, d_t):
		""" generator for linear trajectory on x """
		for y in np.arange(y_start, y_end, delta_y):
			yield x, y, math.pi/2

	def traj_circle(x_center, y_center, radius, d_alpha, d_t):
		""" generator for circular trajectory """
		for alpha in np.arange(0, 2 * math.pi, d_alpha):
			x = x_center + math.cos(alpha) * radius
			y = y_center + math.sin(alpha) * radius
			yield x, y, alpha + math.pi/2

	# collection of generators
	generators = [
		("traj linear on x, y centered", traj_linear_on_x(5, 45, 1, 30, 1)),
		("traj linear on x, y low", traj_linear_on_x(5, 45, 1, 10, 1)),
		("traj linear on x, y high", traj_linear_on_x(5, 45, 1, 50, 1)),
		("traj linear on y, x centered", traj_linear_on_y(5, 45, 1, 30, 1)),
		("traj circle", traj_circle(30, 30, 20, math.radians(360/120), 1))
	]

	# ground map, constant
	ground_map = np.kron(np.random.choice([0.,1.], [20,20]), np.ones((3,3)))

	for title, generator in generators:

		# run test
		print title
		test_command_sequence(ground_map, create_localizer(ground_map, args), generator)
		print ''


def eval_data(args):
	# load image as ground map and convert to range 0.0:1.0
	if args.custom_map:
		ground_map = scipy.misc.imread(args.custom_map).astype(float)
	else:
		ground_map = scipy.misc.imread(os.path.join(args.eval_data, 'map.png')).astype(float)
	ground_map /= ground_map.max()

	# build localizer
	localizer = create_localizer(ground_map, args)

	# if given, opens performance log
	if args.performance_log:
		performance_log = open(args.performance_log, 'w')
	else:
		performance_log = None

	skip_at_start_counter = args.skip_at_start
	o_odom_x, o_odom_y, o_odom_theta = None, None, None
	o_gt_x, o_gt_y, o_gt_theta = None, None, None
	processed_counter = 0
	for i, (gt_line, odom_pos_line, odom_quat_line, sensor_left_line, sensor_right_line) in enumerate(zip(\
		open(os.path.join(args.eval_data, 'gt.txt')), \
		open(os.path.join(args.eval_data, 'odom_pose.txt')), \
		open(os.path.join(args.eval_data, 'odom_quaternion.txt')), \
		open(os.path.join(args.eval_data, 'sensor_left.txt')), \
		open(os.path.join(args.eval_data, 'sensor_right.txt')) \
	)):
		# if skip frames, ignore all but one every skip_steps
		if args.skip_steps and (i % args.skip_steps) != 0:
			continue

		#print gt_line, odom_pos_line, odom_quat_line, sensor_left_line, sensor_right_line
		gt_x, gt_y, gt_theta = map(float, gt_line.split())
		gt_x *= 100; gt_y *= 100
		#print 'gt', gt_x, gt_y, gt_theta
		odom_x, odom_y = map(float, odom_pos_line.split())
		odom_x *= 100; odom_y *= 100
		z, w = map(float, odom_quat_line.split())
		odom_theta = np.arcsin(z) * 2. * np.sign(w)
		#print odom_x, odom_y, odom_theta
		sensor_left = normalize_sensor(sensor_left_line.strip())
		sensor_right = normalize_sensor(sensor_right_line.strip())
		#print sensor_left, sensor_right

		# optionally skip frames
		if skip_at_start_counter > 0:
			skip_at_start_counter -= 1
			continue

		# if requested to stop after a certain duration
		if args.duration:
			if processed_counter > args.duration:
				break
			else:
				processed_counter += 1

		# if first line, just store first data for local frame computation
		if not o_odom_x:
			o_odom_x, o_odom_y, o_odom_theta = odom_x, odom_y, odom_theta
			o_gt_x, o_gt_y, o_gt_theta = gt_x, gt_y, gt_theta
			continue

		# else compute movement
		# odom
		odom_d_theta = odom_theta - o_odom_theta
		odom_d_x, odom_d_y = rot_mat2(-o_odom_theta).dot([odom_x-o_odom_x, odom_y-o_odom_y])
		o_odom_x, o_odom_y, o_odom_theta = odom_x, odom_y, odom_theta
		# ground truth
		gt_d_theta = gt_theta - o_gt_theta
		gt_d_x, gt_d_y = rot_mat2(-o_gt_theta).dot([gt_x-o_gt_x, gt_y-o_gt_y])
		o_gt_x, o_gt_y, o_gt_theta = gt_x, gt_y, gt_theta

		# start time
		start_time = time.time()

		# do movement
		localizer.apply_command(odom_d_x, odom_d_y, odom_d_theta)

		# have we been asked to fake observation?
		if args.fake_observations:
			# if so, regenerate it from map and ground-truth position
			lpos, rpos = sensors_from_pos(gt_x, gt_y, gt_theta)
			# left sensor
			if localizer.is_in_bound(lpos):
				sensor_left = ground_map[localizer.xyW2C(lpos[0]), localizer.xyW2C(lpos[1])]
			else:
				sensor_left = 0.5
			sensor_left += np.random.uniform(-args.sigma_obs, args.sigma_obs)
			# right sensor
			if localizer.is_in_bound(rpos):
				sensor_right = ground_map[localizer.xyW2C(rpos[0]), localizer.xyW2C(rpos[1])]
			else:
				sensor_right = 0.5
			sensor_right += np.random.uniform(-args.sigma_obs, args.sigma_obs)

		# apply observation
		localizer.apply_obs(sensor_left, sensor_right)
		if args.debug_dump:
			localizer.dump_PX(os.path.join(args.debug_dump, 'PX-{:0>4d}'.format(i)), gt_x, gt_y, gt_theta)

		# end time
		duration = time.time() - start_time

		# dump error
		dump_error(localizer, i, duration, "after mvt+obs", gt_x, gt_y, gt_theta, performance_log)

	# close log file
	if performance_log:
		performance_log.close()


if __name__ == '__main__':

	# command line parsing
	parser = argparse.ArgumentParser(description='Test program for Thymio localization')
	parser.add_argument('--ml_angle_count', type=int, help='Use Markov localization with a discretized angle of angle_count')
	parser.add_argument('--mcl_particles_count', type=int, help='Use Monte Carlo localization with a particles_count particles')
	parser.add_argument('--self_test', help='run unit-testing style of tests on synthetic data', action='store_true')
	parser.add_argument('--eval_data', type=str, help='eval data from directory given as parameter')
	parser.add_argument('--sigma_obs', type=float, default=0.5, help='standard deviation of the observation model for ground color (default: 0.5)')
	parser.add_argument('--max_prob_error', type=float, default=0.01, help='max. error ratio with mode value when spilling over probability in Markov localisation (default: 0.01)')
	parser.add_argument('--prob_uniform', type=float, default=0.0, help='uniform probability added to fight depletion and detect kidnapping (default: 0.0)')
	parser.add_argument('--alpha_xy', type=float, default=0.1, help='relative linear error in motion model (default: 0.1)')
	parser.add_argument('--alpha_theta', type=float, default=0.1, help='relative angular error in motion model (default: 0.1)')
	parser.add_argument('--debug_dump', type=str, help='directory where to dump debug information (default: do not dump)')
	parser.add_argument('--performance_log', type=str, help='filename in which to write performance log (default: do not write log)')
	parser.add_argument('--custom_map', type=str, help='use a custom map (default: use map.png in data directory)')
	parser.add_argument('--skip_steps', type=int, default=3, help='only process one step every N when loading the data file (default: 3)')
	parser.add_argument('--skip_at_start', type=int, default=0, help='optionally, some steps to skip at the beginning of the data file (multiplied by --skip_steps) (default: 0)')
	parser.add_argument('--duration', type=int, help='optionally, process only a certain number of steps (multiplied by --skip_steps) (default: use until the end)')
	parser.add_argument('--fake_observations', help='regenerate observations from map and ground truth', action='store_true')


	args = parser.parse_args()

	if args.self_test:
		self_test(args)
	elif args.eval_data:
		eval_data(args)
	else:
		print 'No task given, use either one of --self_test or --eval_data'
