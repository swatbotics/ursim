######################################################################
#
# robosim.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import numpy
import graphics as gfx
import svgelements as se
import re
import sys
import os

import cv2

import color_blob_detector as blob

import glfw
from CleanGL import gl

import Box2D as B2D

from PIL import Image

# DONE: teardown graphics
# DONE: teardown sim
# DONE: reset sim and env
# DONE: sim robot
# DONE: virtual bump sensors
# DONE: implement renderbuffers in graphics
# DONE: robot camera
# DONE: object detection
# TODO: odometry (EKF?)
# TODO: controller API

TAPE_COLOR = gfx.vec3(0.3, 0.3, 0.9)

CARDBOARD_COLOR = gfx.vec3(0.8, 0.7, 0.6)

LINE_COLORS = [
    TAPE_COLOR,
    CARDBOARD_COLOR
]

PYLON_COLORS = [
    gfx.vec3(1.0, 0.5, 0),
    gfx.vec3(0, 0.8, 0),
]

BALL_COLOR = gfx.vec3(0.5, 0, 1)

CIRCLE_COLORS = [ BALL_COLOR ] + PYLON_COLORS

PYLON_RADIUS = 0.05
PYLON_HEIGHT = 0.20

PYLON_MASS = 0.250
PYLON_I = PYLON_MASS * PYLON_RADIUS * PYLON_RADIUS

BALL_RADIUS = 0.1

TAPE_RADIUS = 0.025

TAPE_DASH_SIZE = 0.15

TAPE_POLYGON_OFFSET = 0.001

BALL_MASS = 0.05
BALL_AREA = numpy.pi*BALL_RADIUS**2
BALL_DENSITY = BALL_MASS / BALL_AREA

WALL_THICKNESS = 0.005
WALL_HEIGHT = 0.5
WALL_Z = 0.03

CARDBOARD_DENSITY_PER_M2 = 0.45

BLOCK_MASS = 0.5
BLOCK_SZ = 0.1
BLOCK_COLOR = gfx.vec3(0.6, 0.5, 0.3)

ROOM_HEIGHT = 1.5

ROOM_COLOR = gfx.vec3(1, 0.97, 0.93)

ROBOT_BASE_RADIUS = 0.5*0.36
ROBOT_BASE_HEIGHT = 0.12
ROBOT_BASE_Z = 0.01
ROBOT_BASE_MASS = 2.35
ROBOT_BASE_I = 0.5*ROBOT_BASE_MASS*ROBOT_BASE_RADIUS**2

ROBOT_BASE_COLOR = gfx.vec3(0.1, 0.1, 0.1)

ROBOT_CAMERA_DIMS = gfx.vec3(0.08, 0.25, 0.04)
ROBOT_CAMERA_Z = 0.18

ROBOT_WHEEL_OFFSET = 0.5*0.230
ROBOT_WHEEL_RADIUS = 0.035
ROBOT_WHEEL_WIDTH = 0.021

DEG = numpy.pi / 180

BUMP_ANGLE_RANGES = numpy.array([
    [ 20, 70 ],
    [ -25, 25 ],
    [ -70, -25 ]
], dtype=numpy.float32) * DEG

BUMP_DIST = 0.005

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_ASPECT = CAMERA_WIDTH / CAMERA_HEIGHT
CAMERA_FOV_Y = 49

CAMERA_NEAR = 0.15
CAMERA_FAR = 50.0

CAMERA_A = CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR)
CAMERA_B = CAMERA_FAR * CAMERA_NEAR / (CAMERA_NEAR - CAMERA_FAR)

CAMERA_F_PX = CAMERA_HEIGHT / (2*numpy.tan(CAMERA_FOV_Y*numpy.pi/360))

CAMERA_K = numpy.array([
    [ CAMERA_F_PX, 0, 0.5*CAMERA_WIDTH ],
    [ 0, CAMERA_F_PX, 0.5*CAMERA_HEIGHT ],
    [ 0, 0, 1 ]], dtype=numpy.float32)

MIN_CONTOUR_AREA = 50

OBJECT_X_RES = 0.02
OBJECT_X_SPLIT_THRESHOLD = 0.16
OBJECT_X_SPLIT_BINS = numpy.round(OBJECT_X_SPLIT_THRESHOLD / OBJECT_X_RES)


SQRT22 = 0.5*numpy.sqrt(2)

def vec_from_color(color):
    return gfx.vec3(color.red, color.green, color.blue) / 255.

######################################################################

class DetectedObject:

    def __init__(self, contour, area, xyz, is_split):

        self.contour = contour
        self.area = area
        self.xyz = xyz.copy()
        self.is_split = is_split

    def ellipse_approx(self):

        mean, V, evals = cv2.PCACompute2(self.xyz, mean=None)

        mean = mean.flatten()
        axes = 2*numpy.sqrt(evals.flatten())

        if numpy.abs(V[2,2]) > SQRT22:
            pivot = V[2,2]
        else:
            pivot = -V[2,0]

        if pivot < 0:
            V[2] = -V[2]

        detV = numpy.dot(V[0], numpy.cross(V[1], V[2]))

        if detV < 0:
            V[1] = -V[1]

        principal_components = V

        return mean, axes, principal_components

######################################################################

class SimObject:

    def __init__(self):
        
        self.gfx_objects = []
        
        self.body = None
        
        self.body_linear_mu = 0.0
        self.body_angular_mu = 0.0

    def init_render(self):
        pass

    def destroy_render(self):
        for obj in self.gfx_objects:
            obj.destroy()
        self.gfx_objects = []

    def render(self):
        
        for obj in self.gfx_objects:
            
            if self.body is not None and hasattr(obj, 'model_pose'):
                
                obj.model_pose = b2xform(self.body.transform)

            obj.render()

    def sim_update(self, world, time, dt):

        if self.body is not None:

            if self.body_linear_mu:
                self.body.ApplyForceToCenter(
                    -self.body_linear_mu * self.body.linearVelocity,
                    True)

            if self.body_angular_mu:
                self.body.ApplyTorque(
                    -self.body_angular_mu * self.body.angularVelocity,
                    True)

######################################################################

def b2ple(array):
    return tuple([float(ai) for ai in array])

def b2xform(transform, z=0.0):
    return gfx.rigid_2d_matrix(transform.position, transform.angle, z)

def tz(z):
    return gfx.translation_matrix(gfx.vec3(0, 0, z))
                                
######################################################################

class Pylon(SimObject):

    static_gfx_object = None

    def __init__(self, world, position, color):

        super().__init__()
        
        assert position.shape == (2,) and position.dtype == numpy.float32
        assert color.shape == (3,) and color.dtype == numpy.float32

        self.body = world.CreateDynamicBody(
            position = b2ple(position),
            fixtures = B2D.b2FixtureDef(
                shape = B2D.b2CircleShape(radius=PYLON_RADIUS),
                density = 1.0,
                restitution = 0.25,
                friction = 0.6
            ),
            userData = self
        )

        self.body.massData = B2D.b2MassData(mass=PYLON_MASS,
                                            I=PYLON_I)

        self.body_linear_mu = 0.9 * PYLON_MASS * 10.0

        self.color = color

    def init_render(self):

        if self.static_gfx_object is None:
            self.static_gfx_object = gfx.IndexedPrimitives.cylinder(
                PYLON_RADIUS, PYLON_HEIGHT, 32, 1,
                self.color,
                pre_transform=tz(0.5*PYLON_HEIGHT))

        self.gfx_objects = [self.static_gfx_object]

    def render(self):
        self.static_gfx_object.color = self.color
        super().render()

    def destroy_render(self):
        self.gfx_objects = []
    

######################################################################

class Ball(SimObject):

    static_gfx_object = None

    def __init__(self, world, position):

        super().__init__()
        
        assert position.shape == (2,) and position.dtype == numpy.float32

        self.body = world.CreateDynamicBody(
            position = b2ple(position),
            fixtures = B2D.b2FixtureDef(
                shape = B2D.b2CircleShape(radius=BALL_RADIUS),
                density = BALL_DENSITY,
                restitution = 0.98,
                friction = 0.95
            ),
            userData = self
        )

        self.body_linear_mu = 0.01 * BALL_MASS * 10.0
        
    def init_render(self):

        if self.static_gfx_object is None:
        
            self.static_gfx_object = gfx.IndexedPrimitives.sphere(
                BALL_RADIUS, 32, 24, 
                BALL_COLOR,
                pre_transform=tz(BALL_RADIUS),
                specular_exponent=60.0,
                specular_strength=0.125)
        
        self.gfx_objects = [ self.static_gfx_object ]

    def destroy_render(self):
        self.gfx_objects = []
        
######################################################################

class Wall(SimObject):

    def __init__(self, world, p0, p1):

        super().__init__()
        
        position = 0.5*(p0 + p1)
        
        delta = p1 - p0
        theta = numpy.arctan2(delta[1], delta[0])

        length = numpy.linalg.norm(delta)
        
        dims = gfx.vec3(length,
                        WALL_THICKNESS, WALL_HEIGHT)


        self.dims = dims

        r = 0.5*BLOCK_SZ
        bx = 0.5*float(length) - 1.5*BLOCK_SZ

        shapes = [
            B2D.b2PolygonShape(box=(b2ple(0.5*dims[:2]))),
            B2D.b2PolygonShape(box=(r, r, (bx, 0), 0)),
            B2D.b2PolygonShape(box=(r, r, (-bx, 0), 0)),
        ]
            
        
        self.body = world.CreateDynamicBody(
            position = b2ple(position),
            angle = float(theta),
            shapes = shapes,
            shapeFixture = B2D.b2FixtureDef(density=1,
                                            restitution=0.1,
                                            friction=0.95),
            userData = self
        )


        rho = CARDBOARD_DENSITY_PER_M2

        mx = rho * (dims[1] * dims[2])
        Ix = mx * dims[0]**2 / 12

        Ib = BLOCK_MASS*BLOCK_SZ**2/6

        mass = mx + 2*BLOCK_MASS 
        I = Ix + 2*(Ib + BLOCK_MASS*bx**2)

        self.body_linear_mu = 0.9 * mass * 10.0
        self.body_angular_mu = I * 10.0
        
        self.body.massData = B2D.b2MassData(
            mass = mass,
            I = I
        )

        self.bx = bx
        self.dims = dims

    def init_render(self):

        gfx_object = gfx.IndexedPrimitives.box(
            self.dims, CARDBOARD_COLOR,
            pre_transform=tz(WALL_Z + 0.5*self.dims[2]))

        self.gfx_objects = [gfx_object]

        for x in [-self.bx, self.bx]:

            block = gfx.IndexedPrimitives.box(
                gfx.vec3(BLOCK_SZ, BLOCK_SZ, BLOCK_SZ),
                BLOCK_COLOR,
                pre_transform=gfx.translation_matrix(
                    gfx.vec3(x, 0, 0.5*BLOCK_SZ)))

            self.gfx_objects.append(block)
        

######################################################################

class Box(SimObject):

    def __init__(self, world, dims, position, angle):

        super().__init__()
        
        assert dims.shape == (3,) and dims.dtype == numpy.float32
        assert position.shape == (2,) and position.dtype == numpy.float32

        self.body = world.CreateDynamicBody(
            position = b2ple(position),
            angle = float(angle),
            fixtures = B2D.b2FixtureDef(
                shape = B2D.b2PolygonShape(box=(b2ple(0.5*dims[:2]))),
                density = 1.0,
                restitution = 0.1,
                friction = 0.6
            ),
            userData = self
        )

        rho = CARDBOARD_DENSITY_PER_M2

        mx = rho * (dims[1] * dims[2])
        my = rho * (dims[0] * dims[2])
        mz = rho * (dims[0] * dims[1])

        Ix = mx * dims[0]**2 / 12
        Iy = my * dims[1]**2 / 12
        Iz = mz*(dims[0]**2 + dims[1]**2)/12

        mass = 2*(mx + my + mz)
        I = 2 * (Ix + Iy + mx * dims[0]**2/4 + my * dims[1]**2/4 + Iz)

        self.body_linear_mu = 0.9 * mass * 10.0
        self.body_angular_mu = I * 10.0
        
        self.body.massData = B2D.b2MassData(
            mass = mass,
            I = I
        )

        self.dims = dims

    def init_render(self):

        gfx_object = gfx.IndexedPrimitives.box(
            self.dims, CARDBOARD_COLOR,
            pre_transform=tz(0.5*self.dims[2]))

        self.gfx_objects = [gfx_object]
        
        
######################################################################

def line_intersect(l1, l2):

    l3 = numpy.cross(l1, l2)
    return l3[:2] / l3[2]

######################################################################

class Room(SimObject):

    def __init__(self, world, dims):

        super().__init__()
        
        self.dims = dims

        shapes = []

        thickness = 1.0

        w = float(dims[0])
        h = float(dims[1])


        shapes.append(
            B2D.b2PolygonShape(
                box=(
                    thickness, 0.5*h+thickness,
                    (-thickness, 0.5*h), 0.0
                )
            )
        )
        
        shapes.append(
            B2D.b2PolygonShape(
                box=(
                    thickness, 0.5*h+thickness,
                    (w+thickness, 0.5*h), 0.0
                )
            )
        )

        shapes.append(
            B2D.b2PolygonShape(
                box=(
                    0.5*w+thickness, thickness,
                    (0.5*w, -thickness), 0.0
                )
            )
        )

        shapes.append(
            B2D.b2PolygonShape(
                box=(
                    0.5*w+thickness, thickness,
                    (0.5*w, h+thickness), 0.0
                )
            )
        )
        
        self.body = world.CreateStaticBody(
            userData = self,
            shapes = shapes
        )


        self.floor_texture = None

    def init_render(self):

        if self.floor_texture is None:
            self.floor_texture = gfx.load_texture('textures/floor_texture.png')
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)


        w, h = self.dims

        vdata = numpy.array([
            [0, 0, 0, 0, 0, 1, 0, 0],
            [w, 0, 0, 0, 0, 1, w, 0],
            [w, h, 0, 0, 0, 1, w, h],
            [0, h, 0, 0, 0, 1, 0, h],
        ], dtype=numpy.float32)

        mode = gl.TRIANGLES

        indices = numpy.array([0, 1, 2, 0, 2, 3], dtype=numpy.uint8)

        floor_obj = gfx.IndexedPrimitives(
            vdata, mode, indices, 0.8*gfx.vec3(1, 1, 1),
            texture=self.floor_texture,
            specular_exponent = 40.0,
            specular_strength = 0.5)

        w, h = self.dims
        z = ROOM_HEIGHT

        verts = numpy.array([
            [ 0, 0, 0 ],
            [ w, 0, 0 ],
            [ 0, h, 0 ],
            [ w, h, 0 ],
            [ 0, 0, z ],
            [ w, 0, z ],
            [ 0, h, z ],
            [ w, h, z ],
        ], dtype=numpy.float32)

        indices = numpy.array([
            [ 0, 5, 1 ], 
            [ 0, 4, 5 ],
            [ 1, 7, 3 ], 
            [ 1, 5, 7 ],
            [ 3, 6, 2 ],
            [ 3, 7, 6 ],
            [ 2, 4, 0 ],
            [ 2, 6, 4 ],
        ], dtype=numpy.uint8)

        room_obj = gfx.IndexedPrimitives.faceted_triangles(
            verts, indices, ROOM_COLOR)

        room_obj.specular_strength = 0.25

        self.gfx_objects = [ floor_obj, room_obj ]

######################################################################

class TapeStrips(SimObject):

    def __init__(self, point_lists):

        super().__init__()

        self.point_lists = point_lists

    def init_render(self):

        r = TAPE_RADIUS
        offset = gfx.vec3(0, 0, r)

        self.gfx_objects = []


        dashes = []

        for points in self.point_lists:

            points = points.copy() # don't modify original points

            deltas = points[1:] - points[:-1]
            segment_lengths = numpy.linalg.norm(deltas, axis=1)
            tangents = deltas / segment_lengths.reshape(-1, 1)

            segment_lengths[0] += TAPE_RADIUS
            points[0] -= TAPE_RADIUS * tangents[0]
            deltas[0] = points[1] - points[0]

            segment_lengths[-1] += TAPE_RADIUS
            points[-1] += TAPE_RADIUS * tangents[-1]
            deltas[-1] = points[-1] - points[-2]

            total_length = segment_lengths.sum()

            num_dashes = int(numpy.ceil(total_length / TAPE_DASH_SIZE))
            if num_dashes % 2 == 0:
                num_dashes -= 1

            u = numpy.hstack(([numpy.float32(0)], numpy.cumsum(segment_lengths)))
            u /= u[-1]

            cur_dash = [ points[0] ]
            cur_u = 0.0
            cur_idx = 0
            
            emit_dash = True

            for dash_idx in range(num_dashes):

                target_u = (dash_idx + 1) / num_dashes

                segment_end_u = u[cur_idx+1]

                while segment_end_u < target_u:
                    cur_idx += 1
                    cur_dash.append(points[cur_idx])
                    segment_end_u = u[cur_idx+1]

                segment_start_u = u[cur_idx]

                assert segment_start_u < target_u
                assert segment_end_u >= target_u
                
                segment_alpha = ( (target_u - segment_start_u) /
                                  (segment_end_u - segment_start_u) )

                cur_dash.append( gfx.mix(points[cur_idx],
                                         points[cur_idx+1],
                                         segment_alpha) )

                if emit_dash:
                    dashes.append(numpy.array(cur_dash, dtype=numpy.float32))

                emit_dash = not emit_dash

                cur_dash = [ cur_dash[-1] ]


        npoints_total = sum([len(points) for points in dashes])

        vdata = numpy.zeros((2*npoints_total, 8), dtype=numpy.float32)
        
        vdata[:, 2] = TAPE_POLYGON_OFFSET
        vdata[:, 5]= 1

        vdata_offset = 0

        indices = []
                

        for points in dashes:

            prev_line_l = None
            prev_line_r = None

            points_l = []
            points_r = []

            for i, p0 in enumerate(points[:-1]):

                p1 = points[i+1]

                tangent = gfx.normalize(p1 - p0)
                normal = numpy.array([-tangent[1], tangent[0]], dtype=numpy.float32)

                line = gfx.vec3(normal[0], normal[1], -numpy.dot(normal, p0))

                line_l = line - offset
                line_r = line + offset

                if i == 0:

                    points_l.append( p0 + r * (normal) )
                    points_r.append( p0 + r * (-normal) )

                else:

                    if abs(numpy.dot(line_l[:2], prev_line_l[:2])) > 0.999:

                        points_l.append( p0 + r * normal )
                        points_r.append( p0 - r * normal )

                    else:

                        points_l.append( line_intersect(line_l, prev_line_l) )
                        points_r.append( line_intersect(line_r, prev_line_r) )

                if i == len(points) - 2:

                    points_l.append( p1 + r * (normal) )
                    points_r.append( p1 + r * (-normal) )

                prev_line_l = line_l
                prev_line_r = line_r

                

            for i in range(len(points)-1):
                a = vdata_offset+2*i
                b = a + 1
                c = a + 2
                d = a + 3
                indices.extend([a, b, c])
                indices.extend([c, b, d])


            points_l = numpy.array(points_l)
            points_r = numpy.array(points_r)

            next_vdata_offset = vdata_offset + 2*len(points)

            vdata[vdata_offset:next_vdata_offset:2, :2] = points_l
            vdata[vdata_offset+1:next_vdata_offset:2, :2] = points_r
            vdata[vdata_offset:next_vdata_offset, 6:8] = vdata[vdata_offset:next_vdata_offset, 0:2]

            vdata_offset = next_vdata_offset

        indices = numpy.array(indices, dtype=numpy.uint32)

        gfx_object = gfx.IndexedPrimitives(vdata, gl.TRIANGLES,
                                           indices=indices,
                                           color=TAPE_COLOR)

        gfx_object.specular_exponent = 100.0
        gfx_object.specular_strength = 0.05

        self.gfx_objects.append(gfx_object)


######################################################################

def linear_angular_from_wheel_lr(wheel_lr_vel):

    l, r = wheel_lr_vel

    linear = (r+l)/2
    angular = (r-l)/(2*ROBOT_WHEEL_OFFSET)

    return linear, angular

def wheel_lr_from_linear_angular(linear_angular):

    linear, angular = linear_angular

    l = linear - angular*ROBOT_WHEEL_OFFSET
    r = linear + angular*ROBOT_WHEEL_OFFSET

    return numpy.array([l, r])

######################################################################

def clamp_abs(quantity, limit):
    return numpy.clip(quantity, -limit, limit)
    

######################################################################

class Robot(SimObject):

    def __init__(self, world, position, angle):

        super().__init__()

        self.body = world.CreateDynamicBody(
            position = b2ple(position),
            angle = float(angle),
            fixtures = B2D.b2FixtureDef(
                shape = B2D.b2CircleShape(radius=ROBOT_BASE_RADIUS),
                density = 1.0,
                restitution = 0.25,
                friction = 0.1,
            ),
            userData = self
        )

        self.body.massData = B2D.b2MassData(
            mass = ROBOT_BASE_MASS,
            I = ROBOT_BASE_I
        )

        # left and then right
        self.wheel_vel_cmd = numpy.array([0, 0], dtype=float)

        self.wheel_offsets = numpy.array([
            [ ROBOT_WHEEL_OFFSET, 0],
            [-ROBOT_WHEEL_OFFSET, 0]
        ], dtype=float)
        
        self.max_lateral_impulse = 0.05 # m/(s*kg)
        self.max_forward_impulse = 0.05 # m/(s*kg)

        self.wheel_velocity_fitler_accel = 2.0 # m/s^2

        self.desired_linear_angular_velocity = numpy.array(
            [0.0, 0.0], dtype=numpy.float32)

        self.desired_wheel_velocity_filtered = numpy.array(
            [0.0, 0.0], dtype=numpy.float32)

        self.rolling_mu = 4.0
        
        self.motors_enabled = True

        self.bump = numpy.zeros(len(BUMP_ANGLE_RANGES), dtype=numpy.uint8)

        self.colliders = set()

    def init_render(self):

        self.gfx_objects.append(
            gfx.IndexedPrimitives.cylinder(
                ROBOT_BASE_RADIUS, ROBOT_BASE_HEIGHT, 64, 1,
                ROBOT_BASE_COLOR,
                pre_transform=tz(0.5*ROBOT_BASE_HEIGHT + ROBOT_BASE_Z),
                specular_exponent=40.0,
                specular_strength=0.75
            )
        )

        tx = -0.5*ROBOT_CAMERA_DIMS[0]
        
        self.gfx_objects.append(
            gfx.IndexedPrimitives.box(
                ROBOT_CAMERA_DIMS,
                ROBOT_BASE_COLOR,
                pre_transform=gfx.translation_matrix(
                    gfx.vec3(tx, 0, 0.5*ROBOT_CAMERA_DIMS[2] + ROBOT_CAMERA_Z)),
                specular_exponent=40.0,
                specular_strength=0.75
            )
        )

        btop = ROBOT_BASE_Z + ROBOT_BASE_HEIGHT
        cbottom = ROBOT_CAMERA_Z

        pheight = cbottom - btop

        for y in [-0.1, 0.1]:

            self.gfx_objects.append(
                gfx.IndexedPrimitives.cylinder(
                    0.01, pheight, 32, 1,
                    gfx.vec3(0.75, 0.75, 0.75),
                    pre_transform=gfx.translation_matrix(gfx.vec3(tx, y, 0.5*pheight + btop)),
                    specular_exponent=20.0,
                    specular_strength=0.75
                )
            )
        
    def sim_update(self, world, time, dt):

        body = self.body

        current_normal = body.GetWorldVector((1, 0))
        current_tangent = body.GetWorldVector((0, 1))

        lateral_velocity = body.linearVelocity.dot(current_tangent)

        lateral_impulse = clamp_abs(-body.mass * lateral_velocity,
                                    self.max_lateral_impulse)

        body.ApplyLinearImpulse(lateral_impulse * current_tangent,
                                body.position, True)

        desired_wheel_velocity = wheel_lr_from_linear_angular(
            self.desired_linear_angular_velocity
        )

        self.desired_wheel_velocity_filtered += clamp_abs(
            desired_wheel_velocity - self.desired_wheel_velocity_filtered,
            self.wheel_velocity_fitler_accel * dt)

        for idx, side in enumerate([1.0, -1.0]):

            offset = B2D.b2Vec2(0, side * ROBOT_WHEEL_OFFSET)

            world_point = body.GetWorldPoint(offset)

            wheel_velocity = body.GetLinearVelocityFromWorldPoint(world_point)

            wheel_fwd_velocity = wheel_velocity.dot(current_normal)

            if self.motors_enabled:

                wheel_velocity_error = (
                    self.desired_wheel_velocity_filtered[idx] - wheel_fwd_velocity
                )

                forward_impulse = clamp_abs(
                    wheel_velocity_error * body.mass,
                    self.max_forward_impulse)

                body.ApplyLinearImpulse(0.5 * forward_impulse * current_normal,
                                        world_point, True)
                
            else:

                body.ApplyForce(-self.rolling_mu*wheel_fwd_velocity * current_normal,
                                world_point, True)

        self.bump[:] = 0

        transformA = self.body.transform

        finished_colliders = set()

        for collider in self.colliders:
            
            transformB = collider.body.transform

            collider_did_hit = False
            
            for fixtureA in self.body.fixtures:
                shapeA = fixtureA.shape

                for fixtureB in collider.body.fixtures:
                    shapeB = fixtureB.shape

                    pointA, _, distance, _ = B2D.b2Distance(
                        shapeA = shapeA,
                        shapeB = shapeB,
                        transformA = transformA,
                        transformB = transformB
                    )

                    if distance < BUMP_DIST:

                        collider_did_hit = True

                        lx, ly = self.body.GetLocalPoint(pointA)

                        theta = numpy.arctan2(ly, lx)

                        in_range = ( (theta >= BUMP_ANGLE_RANGES[:,0]) &
                                     (theta <= BUMP_ANGLE_RANGES[:,1]) )

                        self.bump |= in_range
                        
            if not collider_did_hit:
                finished_colliders.add(collider)

        #print('bump:', self.bump)
        self.colliders -= finished_colliders

                                
######################################################################

def match_color(color, carray):

    if not isinstance(carray, numpy.ndarray):
        carray = numpy.array(carray)

    assert len(carray.shape) == 2 and carray.shape[1] == 3

    dists = numpy.linalg.norm(carray - color, axis=1)

    i = dists.argmin()

    return i, carray[i]

    
######################################################################

class SvgTransformer:

    def __init__(self, width, height, scl):

        self.scl = scl
        self.dims = numpy.array([width, height], dtype=numpy.float32) * scl

        shift = numpy.array([[1, 0, 0],
                             [0, 1, self.dims[1]],
                             [0, 0, 1]])
        
        flip = numpy.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]])

        S = numpy.array([[scl, 0, 0],
                         [0, scl, 0],
                         [0, 0, 1]])

        


        self.global_transform = numpy.dot(numpy.dot(shift, flip), S)
        

        self.local_transform = numpy.eye(3)

    def set_local_transform(self, xx, yx, xy, yy, x0, y0):

        self.local_transform = numpy.array([[xx, xy, x0],
                                            [yx, yy, y0],
                                            [0, 0, 1]])

    def transform(self, x, y):

        point = numpy.array([x, y, 1])
        point = numpy.dot(self.local_transform, point)
        point = numpy.dot(self.global_transform, point)

        return point[:2].astype(numpy.float32)

    def scale_dims(self, x, y):
        return self.scl * x, self.scl*y

######################################################################

    
class RoboSim(B2D.b2ContactListener):

    def __init__(self):

        super().__init__()

        self.detector = blob.ColorBlobDetector(mode='rgb')

        self.world = B2D.b2World(gravity=(0, 0), doSleep=True)
        self.world.contactListener = self

        self.dims = numpy.array([-1, -1], dtype=numpy.float32)

        self.objects = []
        self.robot = None

        self.dt = 0.01 # 100 HZ
        self.ticks_per_camera_frame = 4

        self.velocity_iterations = 6
        self.position_iterations = 2
        
        self.remaining_sim_time = 0.0
        self.sim_time = 0.0
        self.sim_ticks = 0

        self.svg_filename = None

        self.framebuffer = None
        
        self.camera_perspective = gfx.perspective_matrix(
            CAMERA_FOV_Y, CAMERA_ASPECT,
            CAMERA_NEAR, CAMERA_FAR)

        self.camera_rotation = numpy.array([
            [ 0, -1, 0, 0 ],
            [ 0,  0, 1, 0 ],
            [-1,  0, 0, 0 ],
            [ 0,  0, 0, 1 ]
        ], dtype=numpy.float32)
        

        u = numpy.arange(CAMERA_WIDTH, dtype=numpy.float32)
        v = numpy.arange(CAMERA_HEIGHT, dtype=numpy.float32)

        u += 0.5 - 0.5*CAMERA_WIDTH
        v += 0.5 - 0.5*CAMERA_HEIGHT

        self.robot_y_per_camera_z = -u.reshape(1, -1) / CAMERA_F_PX
        self.robot_z_per_camera_z = -v.reshape(-1, 1) / CAMERA_F_PX

        self.camera_rgb = numpy.zeros(
            (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=numpy.uint8)
        
        self.camera_labels = numpy.zeros(
            (CAMERA_HEIGHT, CAMERA_WIDTH), dtype=numpy.uint8)

        self.camera_points = numpy.zeros(
            (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=numpy.float32)

        self.detections = None
        
        print('created the world!')

    def reload(self):

        self.clear()

        if self.svg_filename is not None:
            self.load_svg(self.svg_filename)

    def clear(self):

        self.remaining_sim_time = 0.0
        self.sim_time = 0.0
        self.sim_ticks = 0

        for obj in self.objects:
            obj.destroy_render()
            if obj.body is not None:
                self.world.DestroyBody(obj.body)
            
        self.objects = []
        
        self.robot = None

    def load_svg(self, svgfile):

        svg = se.SVG.parse(svgfile, color='none')
        print('parsed', svgfile)

        scl = 1e-2
        
        xform = SvgTransformer(svg.viewbox.viewbox_width,
                               svg.viewbox.viewbox_height, scl)
        
        self.dims = xform.dims

        self.objects.append(Room(self.world, self.dims))

        robot_init_position = 0.5*self.dims
        robot_init_angle = 0.0

        tape_point_lists = []

        for item in svg:

            xx, yx, xy, yy, x0, y0 = [getattr(item.transform, letter)
                                      for letter in 'abcdef']

            det = xx*yy - yx*xy
            is_rigid = (abs(det - 1) < 1e-4)
            
            assert(is_rigid)

            xform.set_local_transform(xx, yx, xy, yy, x0, y0)
            
            fcolor = None
            scolor = None

            if item.fill.value is not None:
                fcolor = vec_from_color(item.fill)

            if item.stroke.value is not None:
                scolor = vec_from_color(item.stroke)

            if isinstance(item, se.Rect):

                w, h = xform.scale_dims(item.width, item.height)

                cx = item.x + 0.5*item.width
                cy = item.y + 0.5*item.height

                dims = gfx.vec3(w, h, min(w, h))
                pctr = xform.transform(cx, cy)
                pfwd = xform.transform(cx+1, cy)
                delta = pfwd-pctr
                theta = numpy.arctan2(delta[1], delta[0])
                
                if numpy.all(fcolor == 1):

                    # room rectangle
                    continue

                else:

                    self.objects.append( Box(self.world, dims,
                                             pctr, theta) )
                
            elif isinstance(item, se.Circle):
                
                cidx, color = match_color(fcolor, CIRCLE_COLORS)

                position = xform.transform(item.cx, item.cy)

                if cidx == 0:
                    self.objects.append(Ball(self.world, position))
                else:
                    self.objects.append(Pylon(self.world, position, color))
                                        
            elif isinstance(item, se.SimpleLine):

                p0 = xform.transform(item.x1, item.y1)
                p1 = xform.transform(item.x2, item.y2)

                cidx, color = match_color(scolor, LINE_COLORS)

                if cidx == 0:
                    
                    points = numpy.array([p0, p1])
                    tape_point_lists.append(points)
                    
                else:

                    self.objects.append( Wall(self.world, p0, p1) )
                
            elif isinstance(item, se.Polyline):
                
                points = numpy.array(
                    [xform.transform(p.x, p.y) for p in item.points])

                tape_point_lists.append(points)

            elif isinstance(item, se.Polygon):

                points = numpy.array(
                    [xform.transform(p.x, p.y) for p in item.points])

                assert len(points) == 3

                pairs = numpy.array([
                    [0, 1],
                    [1, 2],
                    [2, 0]
                ])

                diffs = points[pairs[:,0]] - points[pairs[:,1]]

                dists = numpy.linalg.norm(diffs, axis=1)
                a = dists.argmin()

                i, j = pairs[a]
                k = 3-i-j

                tangent = diffs[a]
                ctr = 0.5*(points[i] + points[j])

                normal = gfx.normalize(gfx.vec2(-tangent[1], tangent[0]))

                dist = numpy.dot(normal, points[k]-ctr)

                if dist < 0:
                    normal = -normal
                    dist = -dist

                print('dist:', dist)
                robot_init_position = ctr + 0.5 * dist * normal
                robot_init_angle = numpy.arctan2(normal[1], normal[0])

            else:
                
                print('*** warning: ignoring SVG item:', item, '***')
                continue

        if len(tape_point_lists):
            self.objects.append(TapeStrips(tape_point_lists))


        print('robot will start at {} {}'.format(
            robot_init_position, robot_init_angle))

        self.robot = Robot(self.world,
                           robot_init_position,
                           robot_init_angle)

        self.objects.append(self.robot)

            
        self.svg_filename = os.path.abspath(svgfile)

    def init_render(self):
        
        for obj in self.objects:
            obj.init_render()

        if self.framebuffer is None:
            self.framebuffer = gfx.Framebuffer(CAMERA_WIDTH, CAMERA_HEIGHT)

        self.render_framebuffer()
        self.process_camera()

    def render(self):
        for obj in self.objects:
            obj.render()

    def render_framebuffer(self):

        #now = glfw.get_time()

        self.framebuffer.activate()

        gl.Viewport(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
        gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        # set up modelview and perspective matrices

        M = b2xform(self.robot.body.transform,
                    ROBOT_CAMERA_Z + 0.5*ROBOT_CAMERA_DIMS[2])

        M = numpy.linalg.inv(M)
        M = numpy.dot(self.camera_rotation, M)

        gfx.IndexedPrimitives.set_view_matrix(M)

        gfx.IndexedPrimitives.set_perspective_matrix(self.camera_perspective)

        self.render()

        self.framebuffer.deactivate()

        gl.BindTexture(gl.TEXTURE_2D, self.framebuffer.rgb_texture)
        
        rgb_buffer = gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGB, gl.UNSIGNED_BYTE)
        rgb_array = numpy.frombuffer(rgb_buffer, dtype=numpy.uint8)
        rgb_image_flipped = rgb_array.reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        
        self.camera_rgb[:] = rgb_image_flipped[::-1]

        camera_ycbcr = self.detector.convert_to_ycrcb(self.camera_rgb)
        self.detector.label_image(camera_ycbcr, self.camera_labels)

        gl.BindTexture(gl.TEXTURE_2D, self.framebuffer.depth_texture)
        
        depth_buffer = gl.GetTexImage(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT, gl.FLOAT)
        depth_array = numpy.frombuffer(depth_buffer, dtype=numpy.float32)
        depth_image_flipped = depth_array.reshape(CAMERA_HEIGHT, CAMERA_WIDTH)

        depth_image = depth_image_flipped[::-1]

        camera_z = (CAMERA_B / (depth_image + CAMERA_A))
        

        
        # camera Z = robot X
        self.camera_points[:,:,0] = camera_z

        Z = self.camera_points[:,:,0]

        # camera X = negative robot Y
        self.camera_points[:,:,1] = Z * self.robot_y_per_camera_z

        # camera Y = negative robot Z
        self.camera_points[:,:,2] = Z * self.robot_z_per_camera_z

    def process_camera(self):
    
        single_object_mask = numpy.empty((CAMERA_HEIGHT, CAMERA_WIDTH),
                                         dtype=numpy.uint8)

        self.detections = dict()

        for color_index in range(self.detector.num_colors):

            color_name = self.detector.color_names[color_index]

            mask = self.camera_labels & (1 << color_index)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            color_detections = []

            for contour_index in range(len(contours)):

                contour = contours[contour_index]

                x0, y0, w, h = cv2.boundingRect(contour)
                
                if w*h < MIN_CONTOUR_AREA:
                    continue

                topleft = (x0, y0)
                
                draw_mask = single_object_mask[:h, :w]
                draw_mask[:] = 0

                shifted = contour - topleft

                cv2.drawContours(draw_mask, [shifted], 0,
                                 (255, 255, 255), -1)

                area = numpy.count_nonzero(draw_mask)
                if area < MIN_CONTOUR_AREA:
                    continue
                
                xyz_subrect = self.camera_points[y0:y0+h, x0:x0+w]

                mask_i, mask_j = numpy.nonzero(draw_mask)

                object_xyz = xyz_subrect[mask_i, mask_j]

                object_x = object_xyz[:, 0]
                ox0 = object_x.min()
                bin_idx = ((object_x - ox0) / OBJECT_X_RES).astype(numpy.int32)

                unique_bins = numpy.unique(bin_idx)
                diffs = unique_bins[1:] - unique_bins[:-1]

                toobig = (diffs > OBJECT_X_SPLIT_BINS)

                if not numpy.any(toobig):
                    
                    detection = DetectedObject(contour, area,
                                               object_xyz, is_split=False)
                    
                    color_detections.append(detection)

                else:

                    uidx0 = 0

                    while uidx0 < len(unique_bins):
                        
                        uidx1 = uidx0
                        
                        while uidx1 < len(toobig) and not toobig[uidx1]:
                            uidx1 += 1
                            
                        first_ok_bin = unique_bins[uidx0]
                        last_ok_bin = unique_bins[uidx1]

                        xidx, = numpy.nonzero((bin_idx >= first_ok_bin) &
                                              (bin_idx <= last_ok_bin))

                        area = len(xidx)

                        if area > MIN_CONTOUR_AREA:
                        
                            xi = mask_i[xidx]
                            xj = mask_j[xidx]

                            draw_mask[:] = 0
                            draw_mask[xi, xj] = 255

                            detection = DetectedObject(contour, area,
                                                       xyz_subrect[xi, xj],
                                                       is_split=True)

                            color_detections.append(detection)

                        uidx0 = uidx1 + 1

            color_detections.sort(key=lambda d: (-d.area, -d.contour[0,0,1]))

            self.detections[color_name] = color_detections
                                                   

    def update(self, time_since_last_update):

        self.remaining_sim_time += time_since_last_update

        #print('simulating {} worth of real time'.format(
        #    self.remaining_sim_time))
        
        while self.remaining_sim_time >= self.dt:
            
            self.sim_time += self.dt
            self.remaining_sim_time -= self.dt
            self.sim_ticks += 1

            if self.sim_ticks % self.ticks_per_camera_frame == 0:
                self.render_framebuffer()
                self.process_camera()

                # TODO: call controller
            
            #now = glfw.get_time()
            
            for obj in self.objects:
                obj.sim_update(self.world, self.sim_time, self.dt)

            self.world.Step(self.dt,
                            self.velocity_iterations,
                            self.position_iterations)

            self.world.ClearForces()

            #elapsed = glfw.get_time() - now
            #print('  sim step took', elapsed, 'seconds')


    def PreSolve(self, contact, old_manifold):

        other = None

        if contact.fixtureA.body == self.robot.body:
            other = contact.fixtureB.body
        elif contact.fixtureB.body == self.robot.body:
            other = contact.fixtureA.body

        if other is not None:
            self.robot.colliders.add(other.userData)
        
######################################################################

class RoboSimApp(gfx.GlfwApp):

    def __init__(self, sim):

        super().__init__()

        self.create_window('Robot simulator', 640, 480, units='window')

        gfx.IndexedPrimitives.DEFAULT_SPECULAR_EXPONENT = 100.0
        gfx.IndexedPrimitives.DEFAULT_SPECULAR_STRENGTH = 0.1

        
        gl.Enable(gl.FRAMEBUFFER_SRGB)
        gl.Enable(gl.DEPTH_TEST)
        gl.Enable(gl.CULL_FACE)

        sim.init_render()

        self.sim = sim

        self.perspective = None
        self.view = None

        self.xrot = 0
        self.yrot = 0

        self.mouse_pos = numpy.array(self.framebuffer_size/2, dtype=numpy.float32)

        self.handle_mouse_rot()

        self.animating = True
        self.was_animating = False

        self.image_file_number = 0

    def set_animating(self, a):

        if not a:
            self.was_animating = False
            self.prev_update = None

        self.animating = a

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

            self.sim.remaining_sim_time = 0
            self.sim.update(self.sim.dt)
            self.need_render = True

        elif key == glfw.KEY_R:

            self.sim.reload()
            self.sim.init_render()
            self.need_render = True

        elif key == glfw.KEY_M:

            self.sim.robot.motors_enabled = not self.sim.robot.motors_enabled

        elif key == glfw.KEY_C:

            self.save_camera_images()

        elif key == glfw.KEY_B:
            for obj in self.sim.objects:
                if isinstance(obj, Ball):
                    kick_impulse = B2D.b2Vec2(1, 0)
                    wdist = None
                    for other in self.sim.objects:
                        if isinstance(other, Robot):
                            diff = other.body.position - obj.body.position
                            dist = diff.length
                            if wdist is None or dist < wdist:
                                wdist = dist
                                desired_vel = diff * 4 / dist
                                actual_vel = obj.body.linearVelocity
                                kick_impulse = (desired_vel - actual_vel)*BALL_MASS
                    obj.body.ApplyLinearImpulse(kick_impulse, obj.body.position, True)
                    obj.body.bullet = True
                    print('kicked the ball')


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

    def save_camera_images(self):

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
                    #any_exists = True
                    pass
                filenames[ftype] = filename

            #self.image_file_number += 1
                
            if not any_exists:
                break

        paletted_output = self.sim.detector.colorize_labels(self.sim.camera_labels)
        Image.fromarray(paletted_output).save(filenames['labels'])
        Image.fromarray(self.sim.camera_rgb).save(filenames['rgb'])

        display = self.sim.camera_rgb[:, :, ::-1].copy()
        palette = self.sim.detector.palette[:, ::-1] 

        rvec = numpy.zeros(3, dtype=numpy.float32)
        tvec = rvec.copy()
        theta = numpy.linspace(0, 2*numpy.pi, 32, False).astype(numpy.float32)
        ctheta = numpy.cos(theta).reshape(-1, 1)
        stheta = numpy.sin(theta).reshape(-1, 1)
        dcoeffs = numpy.zeros(4, dtype=numpy.float32)
        opoints = numpy.zeros((32, 3), dtype=numpy.float32)
        R = numpy.array([
            [ 0, -1, 0 ],
            [ 0, 0, -1 ],
            [ 1, 0, 0 ],
        ], dtype=numpy.float32)

        for color_name, color_detections in self.sim.detections.items():
            color_index = self.sim.detector.color_names.index(color_name)
            color_lite = tuple([int(c) for c in palette[color_index] // 2 + 127])
            for detection in color_detections:
                cv2.drawContours(display, [detection.contour], 0, color_lite, 2)
 
        for color_name, color_detections in self.sim.detections.items():
            color_index = self.sim.detector.color_names.index(color_name)
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
                ipoints = numpy.round(ipoints).astype(int)
                cv2.drawContours(display, [ipoints], 0, color_dark, 1, cv2.LINE_AA)
        
        cv2.imwrite(filenames['detections'], display)

        print('wrote', ', '.join(filenames.values()))
        

    ############################################################
        
    def update(self):

        if self.animating:
            now = glfw.get_time()
            if self.was_animating:
                since_last_update = now - self.prev_update
                self.sim.update(since_last_update)                
                #print('seconds per update:', seconds_per_update)
            self.was_animating = True
            self.prev_update = now

        la = numpy.zeros(2)

        if self.key_is_down(glfw.KEY_I):
            la += (0.5, 0)
            
        if self.key_is_down(glfw.KEY_K):
            la += (-0.5, 0)
            
        if self.key_is_down(glfw.KEY_J):
            la += (0, 2.0)
            
        if self.key_is_down(glfw.KEY_L):
            la += (0, -2.0)
            
        if self.key_is_down(glfw.KEY_U):
            la += (0.5, 1.0)
            
        if self.key_is_down(glfw.KEY_O): 
            la += (0.5, -1.0)

        self.sim.robot.desired_linear_angular_velocity[:] = la
        
        self.sim.robot.motors_enabled = numpy.any(la)
        
    def render(self):

        gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
        gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        gl.Viewport(0, 0,
                   self.framebuffer_size[0],
                   self.framebuffer_size[1])
        
        if self.perspective is None:
            
            w, h = self.framebuffer_size
            aspect = w / max(h, 1)

            self.perspective = gfx.perspective_matrix(
                45, aspect, CAMERA_NEAR, CAMERA_FAR)

            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['perspective'],
                            self.perspective)

        if self.view is None:

            Rx = gfx.rotation_matrix(self.xrot, gfx.vec3(1, 0, 0))
            Ry = gfx.rotation_matrix(self.yrot, gfx.vec3(0, 1, 0))

            R_mouse = numpy.dot(Rx, Ry)

            w, h = self.sim.dims
            m = numpy.linalg.norm([w, h])

            self.view = gfx.look_at(eye=gfx.vec3(0.5*w, 0.5*h - 0.5*m, 0.75*ROOM_HEIGHT),
                                    center=gfx.vec3(0.5*w, 0.5*h, 0.75*ROOM_HEIGHT),
                                    up=gfx.vec3(0, 0, 1),
                                    Rextra=R_mouse)

        gfx.IndexedPrimitives.set_perspective_matrix(self.perspective)
        gfx.IndexedPrimitives.set_view_matrix(self.view)

        self.sim.render()

        w = min(self.framebuffer_size[0], self.sim.framebuffer.width)
        h = min(self.framebuffer_size[1], self.sim.framebuffer.height)

        gl.BindFramebuffer(gl.READ_FRAMEBUFFER, self.sim.framebuffer.fbo)
        gl.BindFramebuffer(gl.DRAW_FRAMEBUFFER, 0)
        
        gl.BlitFramebuffer(0, 0, w, h,
                           0, self.framebuffer_size[1]-h//2,
                           w//2, self.framebuffer_size[1],
                           gl.COLOR_BUFFER_BIT, gl.NEAREST)

        gl.BindFramebuffer(gl.FRAMEBUFFER, 0)

######################################################################
            
def _test_load_environment():


    sim = RoboSim()
    sim.load_svg('environments/first_environment.svg')

    #return

    app = RoboSimApp(sim)

    app.run()

if __name__ == '__main__':
    
    _test_load_environment()
