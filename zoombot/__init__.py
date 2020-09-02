######################################################################
#
# zoombot/__init__.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################
#
# This module implements a basic top-down 2D simulation of a modbile
# robot similar to the Turtlebot 2.
#
######################################################################

from .app import RoboSimApp
from .find_path import find_path

__version__ = '0.0.1'
__version_info__ = (0,0,1)
__license__ = 'GPL v3'