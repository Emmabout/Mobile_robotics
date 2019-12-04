#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import os

if __name__ == '__main__':
	try:
		os.mkdir('../result/small_maps')
	except OSError as e:
		pass

	map_names =  ['map', 'map-xhalf', 'map-xhalf-yhalf']
	start_positions = [80, 85, 90]
	for map_name in map_names:
		for start_pos in start_positions:
			command = './test.py --eval_data ../data/forward_x_minus_slow_1/ --ml_angle_count 36 --performance_log ../result/small_maps/forward_x_minus_slow_1_{}_{}_ml_36 --skip_at_start {} --duration 35 --custom_map ../data/{}.png'.format(map_name, start_pos, start_pos, map_name)
			print 'Executing:', command
			os.system(command)