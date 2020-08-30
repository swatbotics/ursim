######################################################################
#
# zoombot/find_path.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import os

class FindPath:

    def __init__(self, fileA):
        self.dirname = os.path.abspath(os.path.dirname(fileA))

    def __call__(self, fileB):
        return os.path.join(self.dirname, fileB)

    
