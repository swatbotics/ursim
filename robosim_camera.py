######################################################################
#
# robosim_camera.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import numpy
import cv2
import color_blob_detector as blob
import graphics as gfx
from PIL import Image
from CleanGL import gl
import robosim_core as core
import os
import glfw
import sys
from collections import namedtuple

CAMERA_WIDTH = 448
CAMERA_HEIGHT = 336
CAMERA_TOTAL_PIXELS = CAMERA_WIDTH*CAMERA_HEIGHT

CAMERA_ASPECT = CAMERA_WIDTH / CAMERA_HEIGHT
CAMERA_FOV_Y = 49

CAMERA_MIN_POINT_DIST = 0.35
CAMERA_MAX_POINT_DIST = 25.0

CAMERA_NEAR = 0.15
CAMERA_FAR = 50.0

CAMERA_A = CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR)
CAMERA_B = CAMERA_FAR * CAMERA_NEAR / (CAMERA_NEAR - CAMERA_FAR)

CAMERA_DEPTH_NOISE = 0.001

CAMERA_F_PX = CAMERA_HEIGHT / (2*numpy.tan(CAMERA_FOV_Y*numpy.pi/360))

CAMERA_K = numpy.array([
    [ CAMERA_F_PX, 0, 0.5*CAMERA_WIDTH ],
    [ 0, CAMERA_F_PX, 0.5*CAMERA_HEIGHT ],
    [ 0, 0, 1 ]], dtype=numpy.float32)

MIN_CONTOUR_AREA_FRACTION = 50/(640*480)

OBJECT_SPLIT_AXIS = 0

OBJECT_SPLIT_RES = 0.02
OBJECT_SPLIT_THRESHOLD = 0.16
OBJECT_SPLIT_BINS = numpy.round(OBJECT_SPLIT_THRESHOLD / OBJECT_SPLIT_RES)

R_OPENGL_FROM_WORLD = numpy.array([
    [ 0, -1, 0, 0 ],
    [ 0,  0, 1, 0 ],
    [-1,  0, 0, 0 ],
    [ 0,  0, 0, 1 ]
], dtype=numpy.float32)

CAMERA_PERSPECTIVE = gfx.perspective_matrix(
    CAMERA_FOV_Y, CAMERA_ASPECT,
    CAMERA_NEAR, CAMERA_FAR)

SCAN_ANGLE_HALF_SWEEP = 30*numpy.pi/180
SCAN_ANGLE_COUNT = 60 # 1 degree

LOG_RENDER_TIME = 0
LOG_GRAB_TIME = 1
LOG_PROCESS_TIME = 2

LOG_GRAB_RGB_IMAGE_TIME = 3
LOG_GRAB_RGB_PROCESS_TIME = 4
LOG_GRAB_LABELS_IMAGE_TIME = 5
LOG_GRAB_LABELS_PROCESS_TIME = 6
LOG_GRAB_DEPTH_IMAGE_TIME = 7
LOG_GRAB_DEPTH_PROCESS_TIME = 8


LOG_DETECTIONS_START = 9

LOG_TIME_VARS = [
    'profiling.camera.render',
    'profiling.camera.grab',
    'profiling.camera.process',
    'pcgrab.rgb.image',
    'pcgrab.rgb.process',
    'pcgrab.labels.image',
    'pcgrab.labels.process',
    'pcgrab.depth.image',
    'pcgrab.depth.process',
]

USE_PBOS = False

assert len(LOG_TIME_VARS) == LOG_DETECTIONS_START

######################################################################

class LogTimer:

    def __init__(self, array, idx, denom, display=None):
        self.array = array
        self.idx = idx
        self.denom = denom
        self.display = display

    def __enter__(self):
        self.start = glfw.get_time()

    def __exit__(self, type, value, traceback):
        elapsed = glfw.get_time() - self.start
        self.array[self.idx] = elapsed / self.denom
        if self.display is not None:
            print('{}: {}'.format(self.display, self.array[self.idx]))
        

######################################################################

TextureInfo = namedtuple('TextureInfo',
                         'texname, width, height, channels, '
                         'tex_format, storage_type, data_size, '
                         'pbo, ctypes_type, numpy_type')

######################################################################

class SimCamera:

    DATA_TYPE_SIZE = {
        gl.FLOAT: 4,
        gl.UNSIGNED_BYTE: 1
    }

    CTYPES_TYPES = {
        gl.FLOAT: gl.float,
        gl.UNSIGNED_BYTE: gl.ubyte
    }

    NUMPY_TYPES = {
        gl.FLOAT: numpy.float32,
        gl.UNSIGNED_BYTE: numpy.uint8
    }

    def add_texture_to_grab(self, texname, tex_format, storage_type, channels):

        gl.BindTexture(gl.TEXTURE_2D, texname)
        width = gl.GetTexLevelParameteriv(gl.TEXTURE_2D, 0, gl.TEXTURE_WIDTH)
        height = gl.GetTexLevelParameteriv(gl.TEXTURE_2D, 0, gl.TEXTURE_HEIGHT)

        print('texture is {}x{}x{} with format {} and storage type {}'.format(
            width, height, channels,
            gfx.ENUM_LOOKUP[tex_format], gfx.ENUM_LOOKUP[storage_type]))

        data_size = width * height * channels * self.DATA_TYPE_SIZE[storage_type]

        pbo = gl.GenBuffers(1)
        gl.BindBuffer(gl.PIXEL_PACK_BUFFER, pbo)
        gl.BufferData(gl.PIXEL_PACK_BUFFER, data_size, gfx.c_void_p(0), gl.DYNAMIC_READ)
        gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)

        ctypes_type = self.CTYPES_TYPES[storage_type]
        numpy_type = self.NUMPY_TYPES[storage_type]

        tinfo = TextureInfo(texname, width, height, channels,
                            tex_format, storage_type, data_size,
                            pbo, ctypes_type, numpy_type)

        self.grab_textures[texname] = tinfo
        
    def __init__(self, robot, renderables, logger=None, render_labels=True, frame_budget=1.0):

        self.robot = robot
        self.renderables = renderables

        self.render_labels = render_labels

        u = numpy.arange(CAMERA_WIDTH, dtype=numpy.float32)
        v = numpy.arange(CAMERA_HEIGHT, dtype=numpy.float32)

        u += 0.5 - 0.5*CAMERA_WIDTH
        v += 0.5 - 0.5*CAMERA_HEIGHT

        self.detector = blob.ColorBlobDetector(mode='rgb')

        self.robot_y_per_camera_z = -u.reshape(1, -1) / CAMERA_F_PX
        self.robot_z_per_camera_z = -v.reshape(-1, 1) / CAMERA_F_PX

        robot_view_angles_x = numpy.arctan(self.robot_y_per_camera_z).flatten()
        sorter = numpy.arange(CAMERA_WIDTH)[::-1]

        self.scan_angles = numpy.linspace(SCAN_ANGLE_HALF_SWEEP,
                                          -SCAN_ANGLE_HALF_SWEEP,
                                          (SCAN_ANGLE_COUNT+1))

        assert self.scan_angles.min() >= robot_view_angles_x.min()
        assert self.scan_angles.max() <= robot_view_angles_x.max()

        idx = numpy.searchsorted(robot_view_angles_x, self.scan_angles, sorter=sorter)

        self.scan_angle_idx_hi = sorter[idx]
        self.scan_angle_idx_lo = self.scan_angle_idx_hi + 1
        
        assert numpy.all(self.scan_angles <= robot_view_angles_x[self.scan_angle_idx_hi])
        assert numpy.all(self.scan_angles > robot_view_angles_x[self.scan_angle_idx_lo])

        self.scan_ranges = numpy.zeros(len(self.scan_angles),
                                       dtype=numpy.float32)
        
        self.camera_rgb = numpy.zeros(
            (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=numpy.uint8)
        
        self.camera_labels = numpy.zeros(
            (CAMERA_HEIGHT, CAMERA_WIDTH), dtype=numpy.uint8)

        self.camera_points = numpy.zeros(
            (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=numpy.float32)

        self.camera_points_valid = numpy.empty_like(self.camera_labels)

        self.scratch = numpy.empty_like(self.camera_labels)

        self.framebuffer = gfx.Framebuffer(CAMERA_WIDTH, CAMERA_HEIGHT)


        self.framebuffer.add_aux_texture(gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE,
                                         gl.NEAREST, gl.NEAREST,
                                         gl.COLOR_ATTACHMENT1)

        self.framebuffer.add_aux_texture(gl.RGB32F, gl.RGB, gl.FLOAT,
                                         gl.LINEAR, gl.LINEAR,
                                         gl.COLOR_ATTACHMENT2)

        self.grab_textures = dict()

        self.add_texture_to_grab(self.framebuffer.rgb_texture,
                                 gl.RGB, gl.UNSIGNED_BYTE, 3)
        
        self.add_texture_to_grab(self.framebuffer.aux_textures[0],
                                 gl.RED_INTEGER, gl.UNSIGNED_BYTE, 1)
        
        self.add_texture_to_grab(self.framebuffer.aux_textures[1],
                                 gl.RGB, gl.FLOAT, 3)

        gl.DrawBuffers(3, [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2])

        self.framebuffer.deactivate()
        
        gfx.check_opengl_errors('add_aux_texture')

        self.detections = dict()

        self.image_file_number = 0

        self.frame_budget = frame_budget

        self.total_grab_time = 0
        self.total_grabbed_frames = 0

        if logger is None:

            self.log_vars = None

        else:

            lvars = LOG_TIME_VARS[:]

            for idx, color_name in enumerate(self.detector.color_names):
                prefix = 'blobfinder.' + color_name + '.'
                lvars.append(prefix + 'num_detections')
                lvars.append(prefix + 'max_area')
            
            self.log_vars = numpy.zeros(len(lvars), dtype=numpy.float32)

            logger.add_variables(lvars, self.log_vars)
            
    def render(self):
        
        self.framebuffer.activate()

        gl.Viewport(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
        #gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        gl.ClearBufferfv(gl.COLOR, 0, numpy.zeros(4, dtype=numpy.float32))
        gl.ClearBufferfv(gl.DEPTH, 0, [1.0])
        gl.ClearBufferfv(gl.COLOR, 1, [0.0, 0.0, 0.0, 0.0])
        gl.ClearBufferfv(gl.COLOR, 2, [CAMERA_FAR, 0.0, 0.0, 0.0])

        M = core.b2xform(self.robot.body.transform, 
                         core.ROBOT_CAMERA_LENS_Z)
    
        M = numpy.linalg.inv(M)
        
        M = numpy.dot(R_OPENGL_FROM_WORLD, M)

        gfx.IndexedPrimitives.set_view_matrix(M)

        gfx.IndexedPrimitives.set_world_matrix(R_OPENGL_FROM_WORLD.T)

        gfx.IndexedPrimitives.set_perspective_matrix(CAMERA_PERSPECTIVE)

        for r in self.renderables:
            r.render()

        self.framebuffer.deactivate()

        #gl.DrawBuffers(1, [gl.COLOR_ATTACHMENT0])

    def prepare_to_grab(self, texname):

        if not USE_PBOS:
            return

        tinfo = self.grab_textures[texname]

        gl.BindBuffer(gl.PIXEL_PACK_BUFFER, tinfo.pbo)
        gl.BindTexture(gl.TEXTURE_2D, texname)
        gl.GetTexImage(gl.TEXTURE_2D, 0, tinfo.tex_format, tinfo.storage_type, gfx.c_void_p(0))
        gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)

    def grab(self, texname):

        tinfo = self.grab_textures[texname]
        
        if USE_PBOS:

            gl.BindBuffer(gl.PIXEL_PACK_BUFFER, tinfo.pbo)
            address = gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY)
            buffer = (tinfo.ctypes_type * (tinfo.height*tinfo.width*tinfo.channels)).from_address(address)
            array = numpy.ctypeslib.as_array(buffer).reshape(tinfo.height, tinfo.width, tinfo.channels).squeeze()

        else:

            gl.BindTexture(gl.TEXTURE_2D, texname)
            buffer = gl.GetTexImage(gl.TEXTURE_2D, 0, tinfo.tex_format, tinfo.storage_type)
            array = numpy.frombuffer(buffer, dtype=tinfo.numpy_type).reshape(tinfo.height, tinfo.width, tinfo.channels).squeeze()

        return array
            
    def done_grabbing(self):
        
        if not USE_PBOS:
            return
        
        gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER)
        gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
        
    def grab_frame(self):

        with LogTimer(self.log_vars, LOG_GRAB_RGB_IMAGE_TIME, self.frame_budget):
            self.prepare_to_grab(self.framebuffer.rgb_texture)
        
        if self.render_labels:
            with LogTimer(self.log_vars, LOG_GRAB_LABELS_IMAGE_TIME, self.frame_budget):
                self.prepare_to_grab(self.framebuffer.aux_textures[0])

        with LogTimer(self.log_vars, LOG_GRAB_DEPTH_IMAGE_TIME, self.frame_budget):
            self.prepare_to_grab(self.framebuffer.aux_textures[1])

        gfx.check_opengl_errors('get tex image for PBOs')

        with LogTimer(self.log_vars, LOG_GRAB_LABELS_PROCESS_TIME, self.frame_budget):

            xyz_image_flipped = self.grab(self.framebuffer.aux_textures[1])
            self.camera_points = xyz_image_flipped[::-1].copy()
            self.done_grabbing()

            Z = self.camera_points[:,:,0]
            numpy.minimum(Z, CAMERA_MAX_POINT_DIST, out=Z)

            self.camera_points_valid = 255*((Z>=CAMERA_MIN_POINT_DIST).view(numpy.uint8))

        # depth scan
        row = CAMERA_HEIGHT//2-1
        zmid = Z[row:row+2].min(axis=0)

        self.scan_ranges[:] = numpy.minimum(zmid[self.scan_angle_idx_lo],
                                            zmid[self.scan_angle_idx_hi])

        near = self.scan_ranges < CAMERA_MIN_POINT_DIST
        self.scan_ranges[near] = numpy.nan
            
        if self.render_labels:

            with LogTimer(self.log_vars, LOG_GRAB_LABELS_PROCESS_TIME, self.frame_budget):
                labels_image_flipped = self.grab(self.framebuffer.aux_textures[0])
                self.camera_labels[:] = labels_image_flipped[::-1].copy()
                self.done_grabbing()

        else:

            rgb_image_flipped = self.grab(self.framebuffer.rgb_texture)
            self.camera_rgb[:] = rgb_image_flipped[::-1].copy()
            self.done_grabbing()


    def process_frame(self):

        if not self.render_labels:
            
            camera_ycrcb = self.detector.convert_to_ycrcb(self.camera_rgb)

            self.detector.label_image(camera_ycrcb,
                                      self.camera_labels,
                                      self.scratch)
        
        self.detections = self.detector.detect_blobs(
            self.camera_labels,
            MIN_CONTOUR_AREA_FRACTION,
            self.camera_points,
            self.camera_points_valid,
            self.scratch,
            OBJECT_SPLIT_AXIS,
            OBJECT_SPLIT_RES,
            OBJECT_SPLIT_BINS)

        if self.log_vars is not None:
            offset = LOG_DETECTIONS_START
            for idx, color_name in enumerate(self.detector.color_names):
                dlist = self.detections[color_name]
                self.log_vars[offset+0] = len(dlist)
                if not len(dlist):
                    self.log_vars[offset+1] = 0
                else:
                    biggest = dlist[0]
                    self.log_vars[offset+1] = biggest.area_fraction
                offset += 2
        
    def update(self):

        with LogTimer(self.log_vars, LOG_RENDER_TIME, self.frame_budget):
            self.render()
        
        with LogTimer(self.log_vars, LOG_GRAB_TIME, self.frame_budget):
            self.grab_frame()

        self.total_grab_time += self.log_vars[LOG_GRAB_TIME]
        self.total_grabbed_frames += 1

        print('average grab time: {}'.format(self.total_grab_time/self.total_grabbed_frames))
        

        with LogTimer(self.log_vars, LOG_PROCESS_TIME, self.frame_budget):
            self.process_frame()

    def save_images(self):

        files = [
            ('rgb', 'png'),
            ('labels', 'png'),
            ('detections', 'png')
        ]

        while True:

            filenames = dict()

            any_exists = False

            for ftype, extension in files:
                filename = 'camera_{}_{:04d}.{}'.format(
                    ftype, self.image_file_number, extension)
                if os.path.exists(filename):
                    any_exists = True
                filenames[ftype] = filename

            self.image_file_number += 1
                
            if not any_exists:
                break

        if self.render_labels:
            self.prepare_to_grab(self.framebuffer.rgb_texture)
            rgb_image_flipped = self.grab(self.framebuffer.rgb_texture)
            self.camera_rgb[:] = rgb_image_flipped[::-1].copy()
            self.done_grabbing()
            
        paletted_output = self.detector.colorize_labels(self.camera_labels)

        Image.fromarray(self.camera_points_valid).save('valid.png')
        Image.fromarray(paletted_output).save(filenames['labels'])
        Image.fromarray(self.camera_rgb).save(filenames['rgb'])

        print('unique labels:', numpy.unique(self.camera_labels))

        display = self.camera_rgb[:, :, ::-1].copy()
        palette = self.detector.palette[:, ::-1]

        ntheta = 32

        rvec = numpy.zeros(3, dtype=numpy.float32)
        tvec = rvec.copy()
        theta = numpy.linspace(0, 2*numpy.pi, ntheta, False).astype(numpy.float32)
        ctheta = numpy.cos(theta).reshape(-1, 1)
        stheta = numpy.sin(theta).reshape(-1, 1)
        dcoeffs = numpy.zeros(4, dtype=numpy.float32)
        opoints = numpy.zeros((ntheta, 3), dtype=numpy.float32)
        R = numpy.array([
            [ 0, -1, 0 ],
            [ 0, 0, -1 ],
            [ 1, 0, 0 ],
        ], dtype=numpy.float32)

        for color_name, color_detections in self.detections.items():
            color_index = self.detector.color_names.index(color_name)
            color_lite = tuple([int(c) for c in palette[color_index] // 2 + 127])
            for detection in color_detections:
                cv2.drawContours(display, [detection.contour], 0, color_lite, 2)
 
        for color_name, color_detections in self.detections.items():
            color_index = self.detector.color_names.index(color_name)
            color_dark = tuple([int(c) for c in palette[color_index] // 2])
            for detection in color_detections:
                mean = detection.xyz_mean
                axes = detection.axes
                pcs = detection.principal_components
                mean = numpy.dot(R, mean)
                pcs = numpy.array([numpy.dot(R, pc) for pc in pcs])
                opoints = (pcs[0].reshape(1, 3) * ctheta * axes[0] +
                           pcs[1].reshape(1, 3) * stheta * axes[1] +
                           mean.reshape(1, 3))
                ipoints, _ = cv2.projectPoints(opoints, rvec, tvec,
                                               CAMERA_K, dcoeffs)
                ipoints = numpy.round(ipoints*4).astype(int)
                cv2.polylines(display, [ipoints], True, color_dark,
                              1, cv2.LINE_AA, shift=2)
        
        cv2.imwrite(filenames['detections'], display)

        print('wrote', ', '.join(filenames.values()))
