#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys

def rot_mat2(angle):
	""" Create a 2D rotation matrix for angle """
	return np.array([[np.cos(angle), -np.sin(angle)],
	                 [np.sin(angle),  np.cos(angle)]])

if __name__ == '__main__':

	# parse command line
	if len(sys.argv) == 1:
		print 'Usage: ' + sys.argv[0] + ' DATA_DIR [STEPS_TO_SHOW]'
		exit(1)
	if len(sys.argv) > 2:
		steps_to_show = int(sys.argv[2])
	else:
		steps_to_show = 1000

	# for gt
	gt = np.loadtxt(os.path.join(sys.argv[1], 'gt.txt'))
	gt[:,0:2] -= gt[0,0:2]
	gt[:,0:2] *= 100.
	gt_xy_aligned = rot_mat2(-gt[0,2]).dot(gt[:,0:2].transpose())
	plt.plot(gt_xy_aligned[0,0:steps_to_show], gt_xy_aligned[1,0:steps_to_show], label='ground truth')

	# for odom
	odom_xy = np.loadtxt(os.path.join(sys.argv[1], 'odom_pose.txt'))
	odom_xy -= odom_xy[0,0:2]
	odom_xy *= 100.
	odom_quat = np.loadtxt(os.path.join(sys.argv[1], 'odom_quaternion.txt'))
	odom_theta = np.arcsin(odom_quat[:,0]) * 2. * np.sign(odom_quat[:,1])
	odom_xy_aligned = rot_mat2(-odom_theta[0]).dot(odom_xy.transpose())
	plt.plot(odom_xy_aligned[0,0:steps_to_show], odom_xy_aligned[1,0:steps_to_show], label='odometry')

	# add legend and show plot
	plt.legend()
	plt.show()
