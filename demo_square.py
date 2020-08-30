######################################################################
#
# demo_square.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import robosim
import robosim_controller as ctrl
import numpy

######################################################################

class SimpleSquareController(ctrl.Controller):

    def __init__(self):
        super().__init__()

    def initialize(self, time, odom_pose):
        self.set_state(time, 'straight', odom_pose)

    def set_state(self, time, state, odom_pose):
        print('set state to {} at time {:.2f}'.format(state, time))
        self.init_time = time
        self.state = state
        self.init_odom_pose = odom_pose.copy()

    def update(self, time, dt, robot_state, scan, detections):

        ##################################################
        # state transition logic
        
        elapsed = time - self.init_time

        # note rounding to nearest update period
        is_done = (elapsed >= 2.0 - 0.5*dt)

        world_from_cur_robot = robot_state.odom_pose

        if is_done:
            
            world_from_prev_robot = self.init_odom_pose
            prev_robot_from_world = world_from_prev_robot.inverse()

            prev_from_cur = prev_robot_from_world * world_from_cur_robot
            
            print('done, current pose in frame of initial pose =',
                  prev_from_cur)
            
            if self.state == 'straight':
                new_state = 'turn'
            else:
                new_state = 'straight'
            self.set_state(time, new_state, robot_state.odom_pose)

        ##################################################
        # output logic
            
        if self.state == 'straight':

            return ctrl.ControllerOutput(
                forward_vel=0.5, angular_vel=0.0)

        else: # self.state == 'turn'

            return ctrl.ControllerOutput(
                forward_vel=0.0, angular_vel=numpy.pi/4)

######################################################################

if __name__ == '__main__':

    controller = SimpleSquareController()
    
    app = robosim.RoboSimApp(controller)

    app.sim.set_dims(5.0, 5.0)
    app.sim.initialize_robot((2.0, 2.0), 0.0)

    app.run()

