######################################################################
#
# ursim/demos/square.py
#
# Demonstrates a very simple state machine inside a controller that
# drives the robot around a square.
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import numpy

from ursim import RoboSimApp, ctrl

######################################################################

class SimpleSquareController(ctrl.Controller):

    def __init__(self):
        super().__init__()

    # called once when the controller starts up
    def initialize(self, time, odom_pose):
        self.set_state(time, 'straight', odom_pose)

    # at the start of each segment of motion
    def set_state(self, time, state, odom_pose):
        print('set state to {} at time {}'.format(state, time))
        self.init_time = time
        self.state = state
        self.init_odom_pose = odom_pose.copy()

    # called every control cycle
    def update(self, time, dt, robot_state, camera_data):

        ##################################################
        # state transition logic
        
        elapsed = time - self.init_time

        # change state after just over 2 seconds to give a little time
        # for a brief stop after every segment
        is_done = elapsed.total_seconds() > 2.15

        if is_done:

            # print out some information about relative transform
            # since the start of this segment
            world_from_cur_robot = robot_state.odom_pose
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

        if elapsed.total_seconds() >= 2.0: # brief stop

            return ctrl.ControllerOutput(
                forward_vel=0.0, angular_vel=0.0)
            
        elif self.state == 'straight': # go straight

            return ctrl.ControllerOutput(
                forward_vel=0.5, angular_vel=0.0)

        else: # self.state == 'turn' # turn left

            return ctrl.ControllerOutput(
                forward_vel=0.0, angular_vel=numpy.pi/4)

######################################################################

if __name__ == '__main__':

    controller = SimpleSquareController()
    
    app = RoboSimApp(controller)

    app.sim.set_dims(5.0, 5.0)
    app.sim.initialize_robot((2.0, 2.0), 0.0)

    app.run()

