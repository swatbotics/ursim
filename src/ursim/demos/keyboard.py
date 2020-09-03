######################################################################
#
# ursim/demos/keyboard.py
#
# Demonstrates control of the robot with the keyboard.
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################
#
# Demo for keyboard control of robot in default environment.
#
######################################################################

import numpy
import glfw
import sys, os
from importlib_resources import open_text

from ursim import RoboSimApp, ctrl

######################################################################

class KeyboardController(ctrl.Controller):

    def __init__(self):
        super().__init__()
        self.app = None

    def update(self, time, dt, robot_state, camera_data):
        
        la = numpy.zeros(2)

        app = self.app

        if app.key_is_down(glfw.KEY_I):
            la += (0.5, 0)
            
        if app.key_is_down(glfw.KEY_K):
            la += (-0.5, 0)
            
        if app.key_is_down(glfw.KEY_J):
            la += (0, 2.0)
            
        if app.key_is_down(glfw.KEY_L):
            la += (0, -2.0)
            
        if app.key_is_down(glfw.KEY_U):
            la += (0.5, 1.0)
            
        if app.key_is_down(glfw.KEY_O): 
            la += (0.5, -1.0)

        if numpy.any(la):
            return ctrl.ControllerOutput(
                forward_vel=la[0],
                angular_vel=la[1])
        else:
            return None
            
######################################################################

def main():

    svg_file = open_text('ursim.environments', 'first_environment.svg')
    
    if len(sys.argv) > 1 and sys.argv[1].endswith('svg'):
        svg_file = sys.argv[1]
        if not os.path.exists(svg_file):
            svg_file = open_text('ursim.environments', svg_file)

    kbctrl = KeyboardController()

    app = RoboSimApp(kbctrl)
    kbctrl.app = app

    app.sim.load_svg(svg_file)
    
    app.run()

if __name__ == '__main__':
    main()
