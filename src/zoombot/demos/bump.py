######################################################################
#
# zoombot/demos/bump.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import numpy
import sys

from .. import ctrl
from ..app import RoboSimApp

######################################################################

def rand_range(lo, hi):
    return lo + numpy.random.random()*(hi-lo)

######################################################################

MIN_DIST = 0.4

class BumpController(ctrl.Controller):

    def __init__(self, is_virtual):
        
        super().__init__()
        
        self.is_virtual = is_virtual

    def initialize(self, time, odom_pose):
        self.set_state(time, 'straight', None)

    def set_state(self, time, state, duration):
        print('set state to {} with '
              'duration {} at time {}'.format(
                  state, duration, time))
        self.init_time = time
        self.state = state
        self.duration = duration

    def update(self, time, dt, robot_state, scan, detections):
        
        ##################################################
        # state transition logic

        elapsed = time - self.init_time

        is_done = (self.duration is not None and
                   elapsed.total_seconds() >= self.duration)

        if self.is_virtual:
            bump_left = scan.ranges[0] < MIN_DIST
            bump_center = scan.ranges[len(scan.ranges)//2] < MIN_DIST
            bump_right = scan.ranges[-1] < MIN_DIST
        else:
            bump_left = robot_state.bump_left
            bump_center = robot_state.bump_center
            bump_right = robot_state.bump_right

        if bump_left or bump_right or bump_center:
            print('BUMP{}{}{}'.format(
                ' LEFT' if bump_left else '',
                ' CENTER' if bump_center else '',
                ' RIGHT' if bump_right else''))

        if self.state == 'straight':
            if bump_center:
                self.set_state(time, 'back_left_big', 1.0)
            elif bump_right and self.state == 'straight':
                self.set_state(time, 'back_left', 1.0)
            elif bump_left and self.state == 'straight':
                self.set_state(time, 'back_right', 1.0)
        elif is_done:
            if self.state == 'back_left':
                self.set_state(time, 'left', rand_range(1.0, 2.0))
            elif self.state == 'back_left_big':
                self.set_state(time, 'left', rand_range(2.5, 3.5))
            elif self.state == 'back_right':
                self.set_state(time, 'right', rand_range(1.0, 2.0))
            else:
                self.set_state(time, 'straight', None)

        ##################################################
        # output logic

        if self.state == 'straight':
            f, a = 0.6, 0.0
        elif self.state == 'left':
            f, a = 0.0, 1.05
        elif self.state == 'right':
            f, a = 0.0, -1.05
        else: # back_left or back_right or back_left_big
            f, a = -0.3, 0.0

        return ctrl.ControllerOutput(forward_vel=f,
                                     angular_vel=a)

######################################################################

if __name__ == '__main__':

    if len(sys.argv) == 2 and sys.argv[1] in ['-v', '--virtual']:
        is_virtual = True
    else:
        print('run this demo with the -v or --virtual flag to use the laser scan')
        is_virtual = False

    controller = BumpController(is_virtual)
    
    app = RoboSimApp(controller)

    app.sim.set_dims(3.0, 3.0)
    app.sim.initialize_robot((1.5, 1.8), 0.1)
    app.sim.add_box((0.5, 1.0, 0.5), (2.7, 0.6), 0.0)
    app.sim.add_wall((0.5, 2.5), (0.5, 1.75))
    app.sim.add_ball((0.5, 0.5))

    app.run()

