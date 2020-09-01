######################################################################
#
# zoombot/demos/keyboard.py
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

from .. import ctrl
from ..app import RoboSimApp
from ..find_path import find_path

######################################################################

class KeyboardController(ctrl.Controller):

    def __init__(self):
        super().__init__()
        self.app = None

    def update(self, time, dt, robot_state, scan, detections):
        
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

    kbctrl = KeyboardController()

    app = RoboSimApp(kbctrl, filter_setpoints=False)
    
    kbctrl.app = app
    
    app.sim.load_svg(find_path('environments/first_environment.svg'))
    
    app.run()

if __name__ == '__main__':
    main()
