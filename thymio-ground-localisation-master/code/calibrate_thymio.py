#!/usr/bin/env python
# -*- coding: utf-8 -*-
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import dbus
import os
import json

from dbus.mainloop.glib import DBusGMainLoop
import gobject


# required to prevent the glib main loop to interfere with Python threads
gobject.threads_init()
dbus.mainloop.glib.threads_init()


# global variables
left_values = []
left_min_value = 1000
left_max_value = 0
right_values = []
right_min_value = 1000
right_max_value = 0
counting = False
step_count = 0


# helper function
def process_calibration_data():
	""" Process the acquired calibration data and write calibration file """
	global left_values, left_min_value, left_max_value, right_values, right_min_value, right_max_value

	# recompute range and middle values
	left_range = left_max_value - left_min_value
	left_middle_value = left_min_value + left_range / 2.
	right_range = right_max_value - right_min_value
	right_middle_value = right_min_value + right_range / 2.
	print 'Left sensor: min', left_min_value, 'max', left_max_value
	print 'Right sensor: min', right_min_value, 'max', right_max_value

	# find start and stop index
	start_index = 0
	stop_index = len(left_values)
	for i in range(len(left_values)):
		if left_values[i] < left_middle_value:
			start_index = i
			break
	for i in range(start_index+200, len(left_values)):
		if left_values[i] > left_middle_value:
			stop_index = i
			break
	print 'Processing data between indices', start_index, 'and', stop_index
	n = stop_index - start_index
	delta = float(n) / 36.
	left_calib_table = []
	right_calib_table = []
	for i in range(17):
		index = start_index + int((3 + i * 2) * delta)
		print index, left_values[index], right_values[index]
		left_calib_table.append(left_values[index])
		right_calib_table.append(right_values[index])
	left_calib_table.reverse()
	right_calib_table.reverse()
	config_filename = 'config.json'
	print 'Writing calibration tables to', config_filename
	with open(config_filename, 'w') as outfile:
		json.dump({ 'left': left_calib_table, 'right': right_calib_table }, outfile)


# callback
def ground_values_received(id, name, values):
	""" Process received ground values """
	global left_values, left_min_value, left_max_value, right_values, right_min_value, right_max_value
	global counting, step_count
	global network
	global loop

	# data of this event contains only left and right raw ground values
	left_value, right_value = values
	left_values.append(left_value)
	right_values.append(right_value)

	# compute min/max
	left_min_value = min(left_min_value, left_value)
	left_max_value = max(left_max_value, left_value)
	left_range = left_max_value - left_min_value
	right_min_value = min(right_min_value, right_value)
	right_max_value = max(right_max_value, right_value)
	right_range = right_max_value - right_min_value

	# start counting
	if left_value < 400 and not counting:
		counting = True
		print 'Calibration pattern detected'

	# values and step
	if counting:
		step_count += 1
	#print len(left_values)-1, left_value, left_min_value, left_max_value, right_value, right_min_value, right_max_value

	# stop condition
	if step_count > 200 and left_value > left_min_value + (left_range * 3) / 4:
		network.SendEventName('stop', [])
		process_calibration_data()
		loop.quit()


# main
if __name__ == '__main__':

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
	aesl_file = os.path.join(dir_path, 'thymio-calibration.aesl')
	network.LoadScripts(aesl_file)

	# create filter for the group values event
	eventfilter = network.CreateEventFilter()
	events = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', eventfilter), dbus_interface='ch.epfl.mobots.EventFilter')
	events.ListenEventName('ground_values')
	dispatch_handler = events.connect_to_signal('Event', ground_values_received)

	# start the robot
	network.SendEventName('start', [])

	# run gobject loop
	loop = gobject.MainLoop()
	loop.run()
