######################################################################
#
# robosim_controller.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import numpy
from color_blob_detector import BlobDetection
from transform2d import Transform2D

from collections import namedtuple

ControllerOutput = namedtuple('ControllerOutput',
                              'forward_vel, angular_vel')

RobotState = namedtuple('RobotState',
                        'odom_pose, '
                        'bump_left, bump_center, bump_right,'
                        'wheel_vel_l, wheel_vel_r, '
                        'forward_vel, anglular_vel')

LaserScan = namedtuple('LaserScan',
                       'angles, ranges')

######################################################################

class Controller:

    def __init__(self):
        pass

    def initialize(self, time, odom_pose):
        pass

    def update(self, time, dt, bump, detections, scan, odom_pose):
        return ControllerOutput(forward_vel=0, angular_vel=0)
    
