######################################################################
#
# zoombot/find_path.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import os

SELF_DIR = os.path.abspath(os.path.dirname(__file__))

def find_path(fileB):
    return os.path.join(SELF_DIR, fileB)

    
