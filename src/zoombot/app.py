######################################################################
#
# zoombot/app.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################
#
# Defines the RoboSimApp class that users will use to interact with
# the robot simulation.
#
######################################################################

from datetime import timedelta
import os, time

import glfw
import numpy

from . import core, gfx, ctrl, camera
from .find_path import find_path
from .clean_gl import gl

# DONE: teardown graphics
# DONE: teardown sim
# DONE: reset sim and env
# DONE: sim robot
# DONE: virtual bump sensors
# DONE: implement renderbuffers in graphics
# DONE: robot camera
# DONE: object detection
# DONE: clean up code to split into separate files with cleaner dependencies
# DONE: logging
# DONE: deal with slower-than-realtime
# DONE: implement 2D rigid xform class
# DONE: odometry (Euler integrator + Gaussian noise)
# DONE: log odometry pose relative to initial position
# DONE: put transform2d in its own python file with documentation
# DONE: controller API in its own python file with documentation
# DONE: frame rate improvements on windows
# DONE: laser scan interface
# DONE: log reader load latest log
# DONE: add laser scan to controller inputs
# DONE: visual feedback for bump sensors?
# DONE: wall checkerboard texture
# DONE: draw nice arrow for robot
# DONE: refactor into module/package structure
# DONE: move Renderables back into core
# DONE: angle diff function in transform2D?
# DONE: dt, sim_time as numpy timedelta?
# DONE: document Controller?
# DONE: document BlobDetection
# DONE: implement dynamic motor model from http://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=SystemModeling
# DONE: friction model to muck up odometry without adding quite so much noise?
# TODO: overshooting on turns? maybe not enough floor grip?
# TODO: add logger to controller interface
# TODO: docs for plotter?
# TODO: nicer GUI/camera interface?
# TODO: more sophisticated frame rate control?

LOG_PROFILING_DELTA = 0
LOG_PROFILING_PHYSICS = 1
LOG_PROFILING_CAMERA = 2
LOG_PROFILING_RENDERCALL = 3
LOG_PROFILING_COUNT = 4

LOG_PROFILING_NAMES = [
    'profiling.overall',
    'profiling.physics',
    'profiling.camera',
    'profiling.rendercalls'
]

######################################################################

class RoboSimApp(gfx.GlfwApp):

    def __init__(self, controller, filter_setpoints=False):

        super().__init__()

        self.create_window('Robot simulator', 640, 480, units='window')

        gfx.IndexedPrimitives.DEFAULT_SPECULAR_EXPONENT = 100.0
        gfx.IndexedPrimitives.DEFAULT_SPECULAR_STRENGTH = 0.1

        gl.Enable(gl.FRAMEBUFFER_SRGB)
        gl.Enable(gl.DEPTH_TEST)
        gl.Enable(gl.CULL_FACE)
        gl.Disable(gl.PROGRAM_POINT_SIZE)

        gl.PointSize(5.0)

        self.sim = core.RoboSim()

        self.perspective = None
        self.view = None

        self.xrot = 0
        self.yrot = 0

        self.mouse_pos = numpy.array(self.framebuffer_size/2, dtype=numpy.float32)

        self.handle_mouse_rot()

        self.animating = True
        self.was_animating = False

        self.scan_gfx_object = None
        self.tan_scan_angles = None
        self.scan_vertex_data = None

        self.should_render_scan = False
        self.should_render_detections = False
        self.should_render_robocam = True

        self.detection_gfx_object = None

        assert self.sim.dt == timedelta(milliseconds=10)
        assert self.sim.physics_ticks_per_update == 4

        self.frame_budget = (self.sim.dt * self.sim.physics_ticks_per_update).total_seconds()
        
        self.sim_camera = camera.SimCamera(self.sim)

        self.last_update_time = None

        self.log_time = numpy.zeros(LOG_PROFILING_COUNT, dtype=numpy.float32)
        self.sim.logger.add_variables(LOG_PROFILING_NAMES, self.log_time)

        self.controller_initialized = False

        self.controller = controller
        self.sim.robot.filter_setpoints = filter_setpoints

    def get_robot_pose(self):

        return core.b2xform(self.sim.robot.body.transform,
                            core.ROBOT_CAMERA_LENS_Z)

    def update_sim(self):

        cam = self.sim_camera

        with self.sim.logger.timer('profiling.camera', self.frame_budget):
            self.sim_camera.update()

        if not self.controller_initialized:
            
            self.controller.initialize(self.sim.sim_time,
                                       self.sim.robot.odom_pose.copy())
            
            self.controller_initialized = True

        robot = self.sim.robot

        robot_state = ctrl.RobotState(
            robot.odom_pose.copy(),
            bool(robot.bump[0]),
            bool(robot.bump[1]),
            bool(robot.bump[2]),
            robot.odom_linear_angular_vel_filtered[0],
            robot.odom_linear_angular_vel_filtered[1])

        scan = ctrl.LaserScan(
            angles=self.sim_camera.scan_angles.copy(),
            ranges=self.sim_camera.scan_ranges.copy())
            
        result = self.controller.update(self.sim.sim_time,
                                        self.frame_budget,
                                        robot_state,
                                        scan,
                                        self.sim_camera.detections)

        if result is None:
            self.sim.robot.motors_enabled = False
            self.sim.robot.desired_linear_angular_vel[:] = 0
        else:
            self.sim.robot.motors_enabled = True
            self.sim.robot.desired_linear_angular_vel[:] = (
                result.forward_vel, result.angular_vel)

        with self.sim.logger.timer('profiling.physics', self.frame_budget):
            self.sim.update()

    def set_animating(self, a):

        if not a:
            self.was_animating = False
            self.prev_update = None

        self.animating = a

    def destroy(self):
        self.sim.logger.finish()

    def key(self, key, scancode, action, mods):

        if action != glfw.PRESS:
            return
        
        if key == glfw.KEY_ESCAPE:
            
            glfw.set_window_should_close(self.window, gl.TRUE)

        elif key == glfw.KEY_ENTER:

            self.set_animating(not self.animating)

            print('toggled animating =', self.animating)

        elif key == glfw.KEY_SPACE:

            print('single step')

            if self.animating:
                self.set_animating(False)

            self.update_sim()
            self.need_render = True

        elif key == glfw.KEY_R:

            self.sim.reset(reload_svg=True)
            self.need_render = True
            self.sim_camera.update()
            self.last_update_time = None
            self.controller_initialized = False
            self.cur_robot_pose = None

        elif key == glfw.KEY_M:

            self.sim.robot.motors_enabled = not self.sim.robot.motors_enabled

        elif key == glfw.KEY_C:

            self.sim_camera.save_images()

        elif key == glfw.KEY_B:
            self.sim.kick_ball()

        elif key == glfw.KEY_1:
            self.should_render_detections = not self.should_render_detections
            self.need_render = True
            
        elif key == glfw.KEY_2:
            self.should_render_scan = not self.should_render_scan
            self.need_render = True

        elif key == glfw.KEY_3:
            self.should_render_robocam = not self.should_render_robocam
            self.need_render = True
            
    ############################################################
            
    def mouse(self, button_index, is_press, x, y):
        if button_index == 0 and is_press:
            self.handle_mouse_rot()

    ############################################################

    def motion(self, x, y):
        if self.mouse_state[0]:
            self.handle_mouse_rot()

    ############################################################

    def handle_mouse_rot(self):

        foo = (self.mouse_pos / self.framebuffer_size)

        self.yrot = gfx.mix(-2*numpy.pi, 2*numpy.pi, foo[0])
        self.xrot = gfx.mix(numpy.pi/2, 0, numpy.clip(foo[1], 0, 1))
        #self.xrot = gfx.mix(numpy.pi/2, -numpy.pi/2, numpy.clip(foo[1], 0, 1))

        self.need_render = True
        self.view = None

    ############################################################
            
    def framebuffer_resized(self):
        self.perspective = None

    ############################################################
        
    def update(self):
        
        if self.animating:
            now = glfw.get_time()
            if self.was_animating:
                deadline = self.prev_update + self.frame_budget
                while now < deadline:
                    now = glfw.get_time()
                delta_t = now - self.prev_update
                self.log_time[LOG_PROFILING_DELTA] = delta_t/self.frame_budget
            self.prev_update = now
            self.was_animating = True
            self.update_sim()

    def render_scan(self):

        if self.tan_scan_angles is None:
            self.tan_scan_angles = numpy.tan(self.sim_camera.scan_angles)

        if self.scan_vertex_data is None:
            self.scan_vertex_data = numpy.zeros((len(self.tan_scan_angles)+1, 8), dtype=numpy.float32)

        r = self.sim_camera.scan_ranges.copy()
        r[numpy.isnan(r)] = 0.0
        
        self.scan_vertex_data[1:, 0] = r
        self.scan_vertex_data[1:, 1] = r * self.tan_scan_angles

        if self.scan_gfx_object is None:

            scan_indices = numpy.zeros(len(self.tan_scan_angles)*2, dtype=numpy.uint8)
            scan_indices[::2] = numpy.arange(len(self.tan_scan_angles))
            
            self.scan_gfx_object = gfx.IndexedPrimitives(
                self.scan_vertex_data,
                mode=gl.LINES,
                indices=scan_indices,
                color=gfx.vec3(1, 0, 0),
                enable_lighting=False,
                draw_type=gl.DYNAMIC_DRAW)
        else:
            self.scan_gfx_object.update_geometry(self.scan_vertex_data)

        cam = self.sim_camera
        self.scan_gfx_object.model_pose = cam.rendered_robot_poses[cam.frame_to_grab]

        self.scan_gfx_object.render()

    def render_detections(self):

        if self.detection_gfx_object is None:

            ncircle = 64
            vertex_data = numpy.zeros((ncircle, 8), dtype=numpy.float32)

            theta = numpy.linspace(0, 2*numpy.pi, ncircle, False)
            vertex_data[:, 0] = numpy.cos(theta)
            vertex_data[:, 1] = numpy.sin(theta)

            self.detection_gfx_object = gfx.IndexedPrimitives(
                vertex_data,
                mode=gl.LINE_LOOP,
                indices=None,
                color=gfx.vec3(1, 0, 1),
                enable_lighting=False)

        cam = self.sim_camera
        detector = cam.detector

        gl.Enable(gl.LINE_SMOOTH)

        for color_idx, color_name in enumerate(detector.color_names):

            detections = self.sim_camera.detections[color_name]

            color = detector.palette[color_idx]
            color = color.astype(numpy.float32)/255
            color = color*0.25

            for blob in detections:

                scale = numpy.eye(4, dtype=numpy.float32)
                scale[[0,1,2],[0,1,2]] = blob.axes

                rotate = numpy.eye(4, dtype=numpy.float32)
                rotate[:3,:3] = blob.principal_components.T

                mean = blob.xyz_mean
                delta = 0.01 * blob.principal_components[2]
                if numpy.dot(delta, mean) > 0:
                    delta = -delta

                M = cam.rendered_robot_poses[cam.frame_to_grab]
                M = numpy.dot(M, gfx.translation_matrix(blob.xyz_mean + delta))
                M = numpy.dot(M, rotate)
                M = numpy.dot(M, scale)

                self.detection_gfx_object.model_pose = M
                self.detection_gfx_object.color = color
                self.detection_gfx_object.render()

        gl.Disable(gl.LINE_SMOOTH)

    def render_robocam(self):

        src_w = self.sim_camera.framebuffer.width
        src_h = self.sim_camera.framebuffer.height

        dst_h = int(numpy.round(self.framebuffer_size[1] // 6))
        dst_w = int(numpy.round(dst_h * src_w / src_h))

        dst_x0 = self.framebuffer_size[0] // 2 - dst_w //2

        cam = self.sim_camera

        gl.BindFramebuffer(gl.READ_FRAMEBUFFER,
                           cam.framebuffer.fbos[cam.frame_to_grab])

        gl.BindFramebuffer(gl.DRAW_FRAMEBUFFER, 0)

        gl.BlitFramebuffer(0, 0, src_w, src_h,
                           dst_x0, self.framebuffer_size[1]-dst_h,
                           dst_x0 + dst_w, self.framebuffer_size[1],
                           gl.COLOR_BUFFER_BIT, gl.LINEAR)

        gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
        
        
    def render(self):


        with self.sim.logger.timer('profiling.rendercalls', self.frame_budget):

            gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
            gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

            gl.Viewport(0, 0,
                       self.framebuffer_size[0],
                       self.framebuffer_size[1])

            if self.perspective is None:

                w, h = self.framebuffer_size
                aspect = w / max(h, 1)

                self.perspective = gfx.perspective_matrix(
                    45, aspect, 0.10, 50.0)

                gfx.set_uniform(gfx.IndexedPrimitives.uniforms['perspective'],
                                self.perspective)

            if self.view is None:

                Rx = gfx.rotation_matrix(self.xrot, gfx.vec3(1, 0, 0))
                Ry = gfx.rotation_matrix(self.yrot, gfx.vec3(0, 1, 0))

                R_mouse = numpy.dot(Rx, Ry)

                w, h = self.sim.dims
                m = max(numpy.linalg.norm([w, h]), 8.0)

                self.view = gfx.look_at(
                    eye=gfx.vec3(0.5*w, 0.5*h - 0.5*m, 0.25*core.ROOM_HEIGHT),
                    center=gfx.vec3(0.5*w, 0.5*h, 0.25*core.ROOM_HEIGHT),
                    up=gfx.vec3(0, 0, 1),
                    Rextra=R_mouse)

            gfx.IndexedPrimitives.set_perspective_matrix(self.perspective)
            gfx.IndexedPrimitives.set_view_matrix(self.view)

            for obj in self.sim.objects:
                obj.render()

            if self.should_render_scan:
                self.render_scan()

            if self.should_render_detections:
                self.render_detections()

            if self.should_render_robocam:
                self.render_robocam()
        
######################################################################

