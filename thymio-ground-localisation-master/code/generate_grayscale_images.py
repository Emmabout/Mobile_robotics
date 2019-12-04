#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import os

if __name__ == '__main__':
	try:
		os.mkdir('../result/grayscale_images')
	except OSError as e:
		pass

	map_names =  ['breugel_babel', 'van-gogh_starry-night', 'kandinsky_comp-8', 'vermeer_girl-pearl', 'babar', 'childs-drawing_tooth-fairy']
	runs = ['random_1', 'random_2']
	for map_name in map_names:
		for run in runs:
			command = './test.py --eval_data ../data/{} --ml_angle_count 36 --performance_log ../result/grayscale_images/{}_{}_ml_36 --custom_map ../data/{}.png --fake_observations --sigma_obs 0.1'.format(run, run, map_name, map_name)
			print 'Executing:', command
			os.system(command)