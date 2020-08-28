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

        if robot_state.bump_left or robot_state.bump_center:
            self.set_state(time, 'back_right', 1.0)
        elif robot_state.bump_right:
            self.set_state(time, 'back_left', 1.0)
        elif is_done:
            if self.state == 'back_left':
                self.set_state(time, 'left', 1.0 + numpy.random.random())
            elif self.state == 'back_right':
                self.set_state(time, 'right', 1.0 + numpy.random.random())
            else:
                self.set_state(time, 'straight', None)

        ##################################################
        # output logic

        if self.state == 'straight':
            f, a = 0.6, 0.0
        elif self.state == 'left':
            f, a = 0.0, 0.75
        elif self.state == 'right':
            f, a = 0.0, -0.75
        else: # back_left or back_right
            f, a = -0.3, 0.0

        return ctrl.ControllerOutput(forward_vel=f,
                                     angular_vel=a)

######################################################################

if __name__ == '__main__':

    controller = BumpController()
    
    app = robosim.RoboSimApp(controller)

    app.sim.set_dims(3.0, 3.0)
    app.sim.initialize_robot((1.5, 2.0), 0.0)
    app.sim.add_box((0.5, 1.0, 0.5), (2.7, 0.6), 0.0)
    app.sim.add_wall((0.5, 2.5), (0.5, 1.75))
    app.sim.add_ball((0.5, 0.5))

    app.run()
