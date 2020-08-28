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
                              'vel_forward, vel_angle')

######################################################################

class Controller:

    def __init__(self):
        pass

    def initialize(self, time, odom_pose):
        pass

    def update(self, time, dt, bump, detections, odom_pose):
        return ControllerOutput(0.0, 0.0)
    
######################################################################
