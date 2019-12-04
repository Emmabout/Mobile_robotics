#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import os
import dbus
import time
import math
import json

import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.misc

import sys

import pyximport; pyximport.install()
import localize_with_cpt

from dbus.mainloop.glib import DBusGMainLoop
import gobject
# required to prevent the glib main loop to interfere with Python threads
gobject.threads_init()
dbus.mainloop.glib.threads_init()

x = 0.
y = 0.
th = 0.


def ground_values_received(id, name, values):
	global x, y, th
	global odom_plot
	global localizer
	global prob_img
	global orient_plot
	global performance_log
	if name == 'ground_values':
		# sensors
		sensor_left = float(values[0]) # input to the localizer is within 0 to 1000
		sensor_right = float(values[1]) # input to the localizer is within 0 to 1000
		# odometry
		dx_local = float(values[2]) * 0.01 * 0.1 # input to the localizer is in cm
		dy_local = float(values[3]) * 0.01 * 0.1 # input to the localizer is in cm
		dth_local = float(values[4]) * math.pi / 32767. # input to the localizer is in radian
		x += dx_local * math.cos(th) - dy_local * math.sin(th)
		y += dx_local * math.sin(th) + dy_local * math.cos(th)
		th += dth_local
		#if dx_local != 0.0 or dy_local != 0.0 or dth_local != 0.0:
		print sensor_left, sensor_right, x, y, th

		#odom_plot.set_xdata(numpy.append(odom_plot.get_xdata(), x))
		#odom_plot.set_ydata(numpy.append(odom_plot.get_ydata(), y))

		# localization
		start_time = time.time()
		localizer.apply_command(dx_local, dy_local, dth_local)
		localizer.apply_obs(sensor_left, sensor_right)
		est_x, est_y, est_theta, confidence = localizer.estimate_state()
		duration = time.time() - start_time

		# update plot
		PX = localizer.get_PX().sum(axis=0)
		PX = PX * 255. / PX.max()
		prob_img.set_data(numpy.transpose(PX))
		orient_plot.set_offsets(numpy.column_stack((
			[est_x, est_x + 2 * math.cos(est_theta)],
			[est_y, est_y + 2 * math.sin(est_theta)]
		)))
		plt.draw()

		# dump data
		if performance_log:
			performance_log.write('{} {} {} {} {}\n'.format(\
				duration, \
				est_x, \
				est_y, \
				est_theta, \
				confidence
			))


def unitToSensor(value, table):
	assert len(table) == 17
	tableBin = int(value*16)
	r = value - (1./16.)*tableBin
	if tableBin == 16:
		return table[16]
	return float(table[tableBin]) * (1.-r) + float(table[tableBin+1]) * r

if __name__ == '__main__':

	# if simulation
	if '--simulation' in sys.argv:
		# ... generate configuration on the fly
		c_factor = 0.44 # taken from Enki::Thymio2
		s_factor = 9.   # taken from Enki::Thymio2
		m_factor = 884. # taken from Enki::Thymio2
		a_factor = 60.  # taken from Enki::Thymio2
		# support functions
		def sigm(x, s): return 1. / (1. + numpy.exp(-x * s))
		def response_function(v): return sigm(v - c_factor, s_factor) * m_factor + a_factor
		# fill the table
		calib_table = map(response_function, numpy.linspace(0,1,17))
		config = { 'left': calib_table, 'right': calib_table }
	else:
		# ... otherwise load config file
		config_filename = 'config.json'
		with open(config_filename) as infile:
			config = json.load(infile)

	# load stuff
	vUnitToSensor = numpy.vectorize(unitToSensor, excluded=[1])
	ground_map = numpy.flipud(scipy.misc.imread(sys.argv[1]).astype(float))
	localizer = localize_with_cpt.CPTLocalizer(
		vUnitToSensor(numpy.transpose(ground_map) / 255., config['left']),
		vUnitToSensor(numpy.transpose(ground_map) / 255., config['right']),
		36, 150., 0.01, 0, 0.1, 0.1)

	# log
	if len(sys.argv) > 2:
		performance_log = open(sys.argv[2], 'w')
	else:
		performance_log = None

	# Glib main loop
	DBusGMainLoop(set_as_default=True)

	# open DBus
	bus = dbus.SessionBus()

	# get Aseba network
	try:
		network = dbus.Interface(
		bus.get_object('ch.epfl.mobots.Aseba', '/'),  dbus_interface='ch.epfl.mobots.AsebaNetwork')
	except dbus.exceptions.DBusException:
		raise AsebaException('Can not connect to Aseba DBus services! Is asebamedulla running?')

	# load AESL
	dir_path = os.path.dirname(os.path.realpath(__file__))
	aesl_file = os.path.join(dir_path, 'thymio-localisation.aesl')
	network.LoadScripts(aesl_file)

	# create filter for our event
	eventfilter = network.CreateEventFilter()
	events = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', eventfilter), dbus_interface='ch.epfl.mobots.EventFilter')
	events.ListenEventName('ground_values')
	dispatch_handler = events.connect_to_signal('Event', ground_values_received)

	# matplotlib init
	plt.ion()
	plt.figure(figsize=(17,7))
	plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0)
	# maps
	plt.subplot(1, 2, 1)
	plt.imshow(vUnitToSensor(ground_map / 255., config['left']), origin='lower', interpolation="nearest", cmap='gray')
	plt.subplot(1, 2, 2)
	prob_img = plt.imshow(ground_map, origin='lower', interpolation="nearest", cmap='gray')
	orient_plot = plt.scatter([0,2],[0,0], c=['#66ff66', '#ff4040'], s=40)
	#plt.subplot(1, 3, 3)
	plt.draw()

	# run gobject loop
	loop = gobject.MainLoop()
	loop.run()

