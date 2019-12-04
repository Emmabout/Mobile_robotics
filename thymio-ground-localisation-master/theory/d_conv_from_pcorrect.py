#!/usr/bin/env python

# This script takes as input a value for p_correct and computes the minimal necessary distance for the filter to converge with the 150x150 cm, b/w, 3cm grid size pattern
# It is useful to compare with measured results and estimate the true p_correct and sigma_obs

import sys
import math

log = math.log

def H_b(p):
	return -p * log(p, 2) - (1-p) * log(1-p, 2)

def buffon_laplace(h, d):
	return (4 * d * h - d * d) / (math.pi * h * h) 

p_correct = float(sys.argv[1])
H_noise = H_b(1 - p_correct)
print("H_noise ", H_noise)

h = 0.03 # printed cells are 3x3 cm
move_d = 0.03 * 0.3 # the robot moves at 3 cm/s and data are sampled with a period of 0.3 s
sensor_d = 0.022 # the sensors are 2.2 cm apart

p_diff_move = buffon_laplace(h, move_d)
print("p_diff_move ", p_diff_move)

H_loss = 1 - H_b(p_diff_move / 2)
print("H_loss", H_loss)

p_diff_sensor = buffon_laplace(h, sensor_d)
print("p_diff_sensor ", p_diff_sensor)

H_sensors = 1 - H_b(p_diff_sensor / 2)
print("H_sensors", H_sensors)

H_step = 2 * (1 - H_noise - H_loss) - H_sensors
print("H_step ", H_step)

H_cm = (H_step / 0.3) / 3
print("H_cm ", H_cm)

d_conv = 20.6 / H_cm
print("d_conv ", d_conv)
