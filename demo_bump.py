######################################################################
#
# demo_bump.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import robosim
import robosim_controller as ctrl
import numpy

######################################################################

def rand_range(lo, hi):
    return lo + numpy.random.random()*(hi-lo)

######################################################################

class BumpController(ctrl.Controller):

    def __init__(self):
        super().__init__()

    def initialize(self, time, odom_pose):
        self.set_state(time, 'straight', None)

    def set_state(self, time, state, duration):
        print('set state to {} with '
              'duration {} at time {:.2f}'.format(
                  state, duration, time))
        self.init_time = time
        self.state = state
        self.duration = duration

    def update(self, time, dt, robot_state, detections):
        
        ##################################################
        # state transition logic

        elapsed = time - self.init_time

        is_done = (self.duration is not None and
                   elapsed > self.duration - 0.5*dt)

        if robot_state.bump_center:
            print('BUMP CENTER')
            self.set_state(time, 'back_left_big', 1.0)
        elif robot_state.bump_right:
            print('BUMP RIGHT')
            self.set_state(time, 'back_left', 1.0)
        elif robot_state.bump_left or robot_state.bump_center:
            print('BUMP LEFT')
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

    controller = BumpController()
    
    app = robosim.RoboSimApp(controller)

    app.sim.set_dims(3.0, 3.0)
    app.sim.initialize_robot((1.5, 1.5), 0.5)
    app.sim.add_box((0.5, 1.0, 0.5), (2.7, 0.6), 0.0)
    app.sim.add_wall((0.5, 2.5), (0.5, 1.75))
    app.sim.add_ball((0.5, 0.5))

    app.run()

