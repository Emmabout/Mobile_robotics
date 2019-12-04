#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import sys

def rot_mat2(angle):
	""" Create a 2D rotation matrix for angle """
	return np.array([[np.cos(angle), -np.sin(angle)],
	                 [np.sin(angle),  np.cos(angle)]])

def normalize_angle(alpha):
	while alpha > math.pi:
		alpha -= 2. * math.pi
	while alpha < -math.pi:
		alpha += 2. * math.pi
	return alpha

if __name__ == '__main__':

	# parse command line
	if len(sys.argv) == 1:
		print 'Usage: ' + sys.argv[0] + ' DATA_DIR [STEPS_TO_SHOW]'
		exit(1)
	if len(sys.argv) > 2:
		steps_to_show = int(sys.argv[2])
	else:
		steps_to_show = 1000

	o_odom_x, o_odom_y, o_odom_theta = None, None, None
	o_gt_x, o_gt_y, o_gt_theta = None, None, None
	odom_d_xs, odom_d_ys, odom_d_thetas = [], [], []
	gt_d_xs, gt_d_ys, gt_d_thetas = [], [], []

	for i, (gt_line, odom_pos_line, odom_quat_line) in enumerate(zip(\
		open(os.path.join(sys.argv[1], 'gt.txt')), \
		open(os.path.join(sys.argv[1], 'odom_pose.txt')), \
		open(os.path.join(sys.argv[1], 'odom_quaternion.txt')) \
	)):
		gt_x, gt_y, gt_theta = map(float, gt_line.split())
		gt_x *= 100; gt_y *= 100
		#print 'gt', gt_x, gt_y, gt_theta
		odom_x, odom_y = map(float, odom_pos_line.split())
		odom_x *= 100; odom_y *= 100
		z, w = map(float, odom_quat_line.split())
		odom_theta = np.arcsin(z) * 2. * np.sign(w)

		# if first line, just store first data for local frame computation
		if not o_odom_x:
			o_odom_x, o_odom_y, o_odom_theta = odom_x, odom_y, odom_theta
			o_gt_x, o_gt_y, o_gt_theta = gt_x, gt_y, gt_theta
			continue

		# else compute movement
		# odom
		odom_d_theta = odom_theta - o_odom_theta
		odom_d_x, odom_d_y = rot_mat2(-o_odom_theta).dot([odom_x-o_odom_x, odom_y-o_odom_y])
		odom_d_xs.append(odom_d_x)
		odom_d_ys.append(odom_d_y)
		odom_d_thetas.append(odom_d_theta)
		o_odom_x, o_odom_y, o_odom_theta = odom_x, odom_y, odom_theta
		# ground truth
		gt_d_theta = gt_theta - o_gt_theta
		gt_d_x, gt_d_y = rot_mat2(-o_gt_theta).dot([gt_x-o_gt_x, gt_y-o_gt_y])
		gt_d_xs.append(gt_d_x)
		gt_d_ys.append(gt_d_y)
		gt_d_thetas.append(gt_d_theta)
		o_gt_x, o_gt_y, o_gt_theta = gt_x, gt_y, gt_theta

	# plot
	gt_d_thetas = map(normalize_angle, gt_d_thetas)
	odom_d_thetas = map(normalize_angle, odom_d_thetas)

	# for gt
	plt.plot(gt_d_xs, label='ground truth - dx')
	plt.plot(gt_d_ys, label='ground truth - dy')
	plt.plot(gt_d_thetas, label='ground truth - dtheta')

	# for odom
	plt.plot(odom_d_xs, label='odometry - dx')
	plt.plot(odom_d_ys, label='odometry - dy')
	plt.plot(odom_d_thetas, label='odometry - dtheta')

	# add legend and show plot
	plt.legend()
	plt.show()
