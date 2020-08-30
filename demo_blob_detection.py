######################################################################
#
# demo_blob_detection.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import robosim
import robosim_controller as ctrl
import numpy

######################################################################

class LookAtController(ctrl.Controller):

    sequence = [
        ('orange_pylon', 1),
        ('green_pylon', -1),
        ('purple_ball', 1)
    ]

    def __init__(self):
        
        super().__init__()

    def initialize(self, time, odom_pose):
        self.set_state(time, idx=0, move=True)

    def set_state(self, time, idx=None, move=None):

        print()
        print('set idx={}, move={} at time={}'.format(idx, move, time))
        
        self.init_time = time
        
        if idx is not None:
            self.cur_idx = idx
            
        if move is not None:
            if move:
                print('looking for {}'.format(self.sequence[self.cur_idx][0]))
            else:
                print('gonna stare at {} for a bit'.format(self.sequence[self.cur_idx][0]))
            self.cur_move = move

    def update(self, time, dt, robot_state, scan, detections):

        color_name, direction = self.sequence[self.cur_idx]
        
        ##################################################
        # state transition logic
        
        if not self.cur_move:
            
            elapsed = time - self.init_time

            # note comparison with rounding
            if elapsed > 3.0 - 0.5*dt:
                
                print('done staring!')
                
                next_idx = (self.cur_idx+1) % len(self.sequence)
                self.set_state(time, idx=next_idx, move=True)
                
        elif len(detections[color_name]):
            
            blob = detections[color_name][0]
            
            angle = numpy.arctan2(blob.xyz_mean[1], blob.xyz_mean[0])
            
            if numpy.abs(angle) < 0.05:
                print('found blob with area', blob.area, 'at angle', angle)
                self.set_state(time, move=False)

        ##################################################
        # output logic

        if not self.cur_move:
            angular_vel = 0
        else:
            angular_vel = 0.5*direction

        return ctrl.ControllerOutput(forward_vel=0, angular_vel=angular_vel)

######################################################################

if __name__ == '__main__':

    controller = LookAtController()
    
    app = robosim.RoboSimApp(controller)

    app.sim.set_dims(5.0, 5.0)

    app.sim.add_pylon((1.0, 3.5), 'orange')
    app.sim.add_pylon((4.0, 3.5), 'green')
    app.sim.add_ball((2.5, 3.5))

    app.sim.initialize_robot((2.5, 1.5), numpy.pi/2)

    app.run()

