#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import os

if __name__ == '__main__':
	try:
		os.mkdir('../result/multiple_mcl')
	except OSError as e:
		pass
	#runs = ['random_1', 'random_2']
	runs = ['random_long']
	mcl_params = [50, 100, 200, 400]
	for i in range(0,10):
		print 'Total percentage done:', i*10, '%'
		for run in runs:
			for mcl_param in mcl_params:
				if run == 'random_long':
					command = './test.py --eval_data ../data/{} --mcl_particles_count {} --prob_uniform 0.1 --duration 700 --performance_log ../result/multiple_mcl/{}_{}_mcl_{}k'.format(run, mcl_param*1000, run, i, mcl_param)
				else:
					command = './test.py --eval_data ../data/{} --mcl_particles_count {} --performance_log ../result/multiple_mcl/{}_{}_mcl_{}k'.format(run, mcl_param*1000, run, i, mcl_param)
				print 'Executing:', command
				os.system(command)

