######################################################################
#
# zoombot/ctrl.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################
#
# Data structures used for defining controllers for simulated robots.
# Users of the zoombot module should be familiar with these. See the
# demos for example usage.
#
######################################################################

import numpy
from .datalog import DataLog
from .color_blob_detector import BlobDetection
from .transform2d import Transform2D

from collections import namedtuple

######################################################################

ControllerOutput = namedtuple('ControllerOutput',
                              'forward_vel, angular_vel')

ControllerOutput.__doc__ = """

Output for robot controller, a tuple of (forward velocity, angular
velocity). Your controller can also output None, which completely
disables the robot's motors. 

"""

######################################################################

RobotState = namedtuple('RobotState',
                        'odom_pose, '
                        'bump_left, bump_center, bump_right,'
                        'forward_vel, anglular_vel')

RobotState.__doc__ = """

Data from non-camera robot sensors. It contains the following fields:

  * odom_pose: the relative odometry pose since startup, as an object
    of type zoombot.transform2d.Transform2D

  * bump_left, bump_center, bump_right: status of bump sensors, expressed
    as boolean False/True values for contact at each location

  * forward_vel, anglular_vel: filtered (smoothed) measurements of robot
    forward and angular velocity in its own coordinate frame
"""

######################################################################

CameraData = namedtuple('CameraData',
                        'scan, detections')

CameraData.__doc__ = """

Data from camera sensors. It contains the following fields:

  * scan: An object of type LaserScan (see documentation in this
    module).

  * detections: Dictionary mapping color name strings to lists of
    color_blob_detector.BlobDetection. See the documentation of that
    class for more details, or look at the
    zoombot.demos.blob_detection example.


"""

######################################################################

LaserScan = namedtuple('LaserScan',
                       'angles, ranges')

LaserScan.__doc__ == """

Simulated laser scan data from the RGBD camera. It has two fields:

  * angles is a flat numpy array of angles from left (positive) to
    right (negative) of the robot's field of view

  * ranges is a flat numpy array of distances to objects along each
    angular direction, with NaN values inserted for missing/max depths

"""

######################################################################

class Controller:

    """Controller is the base class for robot controllers. You will want
    to create your own subclasses of controller for each
    application. See the examples in the zoombot.demos package for
    usage.

    """
    def initialize(self, time, odom_pose):
        """Initialize the controller. Parameters:

          * time: time of simulation startup as an object of type
            datetime.timedelta. Frequently zero, but not necessarily
            guaranteed to be.

          * odom_pose: the relative pose of the robot as an object of
            type zoombot.transform2d.Transform2D. Frequently the
            identity transform, but not necessarily guaranteed to be.

        This function is guaranteed to be called before the first call
        to update() after simulation begins, or after the simulation
        is restarted. 

        Override this function in your own subclass to record any
        necessary information before the controller begins running.

        The return value is ignored.

        """
        pass

    def update(self, time, dt, robot_state, camera_data):
        """Called by the simulator at regular intervals to get a controller
        output from this controller. Parameters:

          * time: the current simulated time, as an object of type
            datetime.timedelta. This is guaranteed to be greater than the
            time of the previous initialize call and all subsequent
            update calls after it.

          * dt: the update period of the controller as an object of
            type datetime.timedelta. To convert to floating-point
            seconds, call dt.total_seconds().

          * robot_state: Robot sensor data as an object of type
            RobotState (see documentation in this module).

          * camera_data: Camera sensor data as an object of type
            CameraData (see documentation in this module).

        Returns an object of type ControllerOutput, or None to disable
        motors entirely.
        """
        return ControllerOutput(forward_vel=0, angular_vel=0)

    def setup_log(self, datalog):

        """Called by the simulator once to set up logging for this controller.
        The parameter datalog is an object of type datalog.DataLog,
        which you may call add_variables() on.

        The default behavior is to not log any variables. See
        zoombot.demos.ctrl_datalog for example usage.

        """
        pass
