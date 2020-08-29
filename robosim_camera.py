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

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
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

MIN_CONTOUR_AREA = 50

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

######################################################################

class SimCamera:

    def __init__(self, robot, renderables, logger=None, render_labels=True):

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
                                         gl.COLOR_ATTACHMENT1)

        gl.DrawBuffers(2, [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])

        self.framebuffer.deactivate()
        
        gfx.check_opengl_errors('add_aux_texture')

        self.detections = dict()

        self.image_file_number = 0


        if logger is None:

            self.log_vars = None

        else:

            lvars = []

            for idx, color_name in enumerate(self.detector.color_names):
                prefix = 'blobfinder.' + color_name + '.'
                lvars.append(prefix + 'num_detections')
                lvars.append(prefix + 'max_area')
            
            self.log_vars = numpy.zeros(len(lvars), dtype=numpy.float32)

            logger.add_variables(lvars, self.log_vars)
            
    def render(self):
        
        self.framebuffer.activate()


        gl.Viewport(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
        gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        M = core.b2xform(self.robot.body.transform, 
                         core.ROBOT_CAMERA_LENS_Z)
    
        M = numpy.linalg.inv(M)
        
        M = numpy.dot(R_OPENGL_FROM_WORLD, M)

        gfx.IndexedPrimitives.set_view_matrix(M)

        gfx.IndexedPrimitives.set_perspective_matrix(CAMERA_PERSPECTIVE)

        for r in self.renderables:
            r.render()

        self.framebuffer.deactivate()

        #gl.DrawBuffers(1, [gl.COLOR_ATTACHMENT0])
        
    def grab_frame(self):

        gl.BindTexture(gl.TEXTURE_2D, self.framebuffer.rgb_texture)
        
        rgb_buffer = gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGB, gl.UNSIGNED_BYTE)
        rgb_array = numpy.frombuffer(rgb_buffer, dtype=numpy.uint8)
        rgb_image_flipped = rgb_array.reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        
        self.camera_rgb[:] = rgb_image_flipped[::-1]

        gl.BindTexture(gl.TEXTURE_2D, self.framebuffer.aux_textures[0])

        if self.render_labels:

            labels_buffer = gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RED_INTEGER, gl.UNSIGNED_BYTE)
            labels_array = numpy.frombuffer(labels_buffer, dtype=numpy.uint8)
            labels_image_flipped = labels_array.reshape(CAMERA_HEIGHT, CAMERA_WIDTH)

            self.camera_labels[:] = labels_image_flipped[::-1]

        gl.BindTexture(gl.TEXTURE_2D, self.framebuffer.depth_texture)
        
        depth_buffer = gl.GetTexImage(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT, gl.FLOAT)
        depth_array = numpy.frombuffer(depth_buffer, dtype=numpy.float32)
        depth_image_flipped = depth_array.reshape(CAMERA_HEIGHT, CAMERA_WIDTH)

        depth_image = depth_image_flipped[::-1]

        nscl = CAMERA_DEPTH_NOISE
        depth_image_noisy = depth_image - 0.5*nscl + nscl*numpy.random.random(size=depth_image.shape)

        camera_z = (CAMERA_B / (depth_image_noisy + CAMERA_A))
        
        # camera Z = robot X
        self.camera_points[:,:,0] = camera_z

        Z = self.camera_points[:,:,0]
        
        Z[Z>CAMERA_MAX_POINT_DIST] = CAMERA_MAX_POINT_DIST

        self.camera_points_valid[:] = 255
        self.camera_points_valid[Z<CAMERA_MIN_POINT_DIST] = 0


        # camera X = negative robot Y
        self.camera_points[:,:,1] = Z * self.robot_y_per_camera_z

        # camera Y = negative robot Z
        self.camera_points[:,:,2] = Z * self.robot_z_per_camera_z

        # depth scan
        row = CAMERA_HEIGHT//2-1
        zmid = Z[row:row+2].min(axis=0)

        self.scan_ranges[:] = numpy.minimum(zmid[self.scan_angle_idx_lo],
                                            zmid[self.scan_angle_idx_hi])

        near = self.scan_ranges < CAMERA_MIN_POINT_DIST
        self.scan_ranges[near] = numpy.nan


    def process_frame(self):

        if not self.render_labels:
            
            camera_ycrcb = self.detector.convert_to_ycrcb(self.camera_rgb)

            self.detector.label_image(camera_ycrcb,
                                      self.camera_labels,
                                      self.scratch)
        
        self.detections = self.detector.detect_blobs(
            self.camera_labels,
            MIN_CONTOUR_AREA,
            self.camera_points,
            self.camera_points_valid,
            self.scratch,
            OBJECT_SPLIT_AXIS,
            OBJECT_SPLIT_RES,
            OBJECT_SPLIT_BINS)

        
        if self.log_vars is not None:
            offset = 0
            for idx, color_name in enumerate(self.detector.color_names):
                dlist = self.detections[color_name]
                self.log_vars[offset+0] = len(dlist)
                if not len(dlist):
                    self.log_vars[offset+1] = 0
                else:
                    biggest = dlist[0]
                    self.log_vars[offset+1] = biggest.area
                offset += 2
        
    def update(self):

        self.render()
        
        self.grab_frame()

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
                mean, axes, pcs = detection.ellipse_approx()
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
