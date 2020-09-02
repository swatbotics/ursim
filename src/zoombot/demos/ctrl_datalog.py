######################################################################
#
# zoombot/demos/ctrl_datalog.py
#
# Demonstrates how to add logging capability to a robot controller.
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import numpy

from ..app import RoboSimApp
from .. import ctrl

######################################################################

class DataLogExampleController(ctrl.Controller):

    ALL_COLORS = [
        'blue_tape',
        'green_pylon',
        'orange_pylon',
        'purple_ball'
    ]

    def __init__(self):
        super().__init__()

    def setup_log(self, datalog):

        names = []

        for idx, color_name in enumerate(self.ALL_COLORS):
            names.append('pos_x.location.' + color_name)
            names.append('pos_y.location.' + color_name)

        assert len(names) == 2*len(self.ALL_COLORS)

        names += ['mood', 'sinewave']

        self.log_vars = numpy.zeros(len(names), dtype=numpy.float32)
    
        datalog.add_variables(names, self.log_vars)
 
        self.mood = datalog.register_enum('mood', ['happy', 'bored', 'angry'])
       
        print('setting up log!')

    def can_see(self, detections, color_name):
        return color_name in detections and len(detections[color_name])

    def update(self, time, dt, robot_state, camera_data):

        # default value is NaN to suppress plots of things we don't
        # have blob detections for
        self.log_vars[:8] = numpy.nan

        detections = camera_data.detections

        for idx, color_name in enumerate(self.ALL_COLORS):

            if color_name not in detections:
                continue
            
            blobs = detections[color_name]
            
            if not len(blobs):
                continue

            biggest = blobs[0]
            pos = biggest.xyz_mean

            if pos is None:
                continue

            odom_pos = robot_state.odom_pose * pos[:2]
            
            self.log_vars[2*idx+0] = odom_pos[0]
            self.log_vars[2*idx+1] = odom_pos[1]

        if self.can_see(detections, 'orange_pylon'):
            mood = 'angry'
        elif self.can_see(detections, 'purple_ball'):
            mood = 'happy'
        else:
            mood = 'bored'

        self.mood.store_value(mood)

        self.log_vars[-1] = numpy.sin(time.total_seconds()*4.0*numpy.pi)
            
        return ctrl.ControllerOutput(
            forward_vel=0.5, angular_vel=0.5)

    
######################################################################

if __name__ == '__main__':

    controller = DataLogExampleController()
    
    app = RoboSimApp(controller)

    app.sim.set_dims(5.0, 5.0)

    app.sim.add_tape_strip(numpy.array([
        [1.0, 1.0],
        [1.0, 4.0],
        [4.0, 4.0],
        [4.0, 1.0],
        [1.0, 1.0]]), 'blue')

    app.sim.add_pylon((1.5, 1.5), 'green')
    app.sim.add_pylon((3.5, 3.5), 'green')
    app.sim.add_pylon((3.5, 1.5), 'orange')
    app.sim.add_pylon((1.5, 3.5), 'orange')

    app.sim.add_ball((2.5, 4.0))
    
    app.sim.initialize_robot((2.5, 1.5), 0.0)

    app.run()

