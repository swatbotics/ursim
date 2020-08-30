######################################################################
#
# zoombot/core.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################
#
# Core 2D physics and graphics for robot simulation. Not for direct
# use by users.
#
######################################################################

import os
from datetime import timedelta

import numpy
import Box2D as B2D
import svgelements 

from . import gfx
from .clean_gl import gl
from .find_path import find_path
from .datalog import Logger
from .transform2d import Transform2D

TAPE_COLOR = gfx.vec3(0.3, 0.3, 0.9)

CARDBOARD_COLOR = gfx.vec3(0.8, 0.7, 0.6)

LINE_COLORS = [
    TAPE_COLOR,
    CARDBOARD_COLOR
]

PYLON_COLOR_NAMES = [
    'orange', 'green'
]

PYLON_COLORS = [
    gfx.vec3(1.0, 0.5, 0),
    gfx.vec3(0, 0.8, 0),
]

BALL_COLOR = gfx.vec3(0.5, 0, 1)

CIRCLE_COLORS = [ BALL_COLOR ] + PYLON_COLORS

PYLON_RADIUS = 0.05
PYLON_HEIGHT = 0.23

PYLON_MASS = 0.250
PYLON_I = PYLON_MASS * PYLON_RADIUS * PYLON_RADIUS


TAPE_RADIUS = 0.025

TAPE_DASH_SIZE = 0.15

TAPE_POLYGON_OFFSET = 0.001

BALL_RADIUS = 0.15

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
ROBOT_CAMERA_BOTTOM_Z = 0.18
ROBOT_CAMERA_LENS_Z = ROBOT_CAMERA_BOTTOM_Z + 0.5*ROBOT_CAMERA_DIMS[2]

ROBOT_WHEEL_OFFSET = 0.5*0.230
ROBOT_WHEEL_RADIUS = 0.035
ROBOT_WHEEL_WIDTH = 0.021

DEG = numpy.pi / 180

BUMP_ANGLE_RANGES = numpy.array([
    [ 20, 70 ],
    [ -25, 25 ],
    [ -70, -25 ]
], dtype=numpy.float32) * DEG

BUMP_DIST = 0.001

GRAVITY = 9.8

WHEEL_MAX_LATERAL_IMPULSE = 0.5 # m/(s*kg)
WHEEL_MAX_FORWARD_FORCE = 10.0 # N
WHEEL_VEL_KP = 0.3 # dimensionless
WHEEL_VEL_KI = 0.3 # 1/s - higher means more overshoot
WHEEL_VEL_INTEGRATOR_MAX = 0.1 # m - affects both overshoot and size of steady-state error

ODOM_FREQUENCY = 4

ODOM_NOISE_STDDEV = 0.015

# Wn = 0.1
ODOM_FILTER_B = numpy.array([0.13672874, 0.13672874])
ODOM_FILTER_A = numpy.array([-0.72654253])

# Wn = 0.05
SETPOINT_FILTER_B = numpy.array([0.07295966, 0.07295966])
SETPOINT_FILTER_A = numpy.array([-0.85408069])

LOG_ROBOT_POS_X =           0
LOG_ROBOT_POS_Y =           1
LOG_ROBOT_POS_ANGLE =       2
LOG_ROBOT_VEL_X =           3
LOG_ROBOT_VEL_Y =           4
LOG_ROBOT_VEL_ANGLE =       5
LOG_ROBOT_CMD_VEL_FORWARD = 6
LOG_ROBOT_CMD_VEL_ANGULAR = 7
LOG_ROBOT_CMD_VEL_LWHEEL  = 8
LOG_ROBOT_CMD_VEL_RWHEEL  = 9
LOG_ROBOT_VEL_FORWARD =     10
LOG_ROBOT_VEL_LWHEEL =      11
LOG_ROBOT_VEL_RWHEEL =      12
LOG_ROBOT_MOTORS_ENABLED  = 13
LOG_ROBOT_BUMP_LEFT       = 14
LOG_ROBOT_BUMP_CENTER     = 15
LOG_ROBOT_BUMP_RIGHT      = 16
LOG_ODOM_VEL_RAW_LWHEEL   = 17
LOG_ODOM_VEL_RAW_RWHEEL   = 18
LOG_ODOM_VEL_RAW_FORWARD  = 19
LOG_ODOM_VEL_RAW_ANGLE    = 20
LOG_ODOM_VEL_FILT_LWHEEL  = 21
LOG_ODOM_VEL_FILT_RWHEEL  = 22
LOG_ODOM_VEL_FILT_FORWARD = 23
LOG_ODOM_VEL_FILT_ANGLE   = 24
LOG_ODOM_POS_X            = 25
LOG_ODOM_POS_Y            = 26
LOG_ODOM_POS_ANGLE        = 27

LOG_NUM_VARS        = 28


LOG_NAMES = [
    'robot.pos.x',
    'robot.pos.y',
    'robot.pos.angle',
    'robot.vel.x',
    'robot.vel.y',
    'robot.vel.angle',
    'robot.cmd_vel.forward',
    'robot.cmd_vel.angle',
    'robot.cmd_wheel_vel.l',
    'robot.cmd_wheel_vel.r',
    'robot.vel.forward',
    'robot.wheel_vel.l',
    'robot.wheel_vel.r',
    'robot.motors_enabled',
    'robot.bump.left',
    'robot.bump.center',
    'robot.bump.right',
    'odom.wheel_vel.raw.l',
    'odom.wheel_vel.raw.r',
    'odom.vel.raw.forward',
    'odom.vel.raw.angle',
    'odom.wheel_vel.filtered.l',
    'odom.wheel_vel.filtered.r',
    'odom.vel.filtered.forward',
    'odom.vel.filtered.angle',
    'odom.pos.x',
    'odom.pos.y',
    'odom.pos.angle'
]



assert len(LOG_NAMES) == LOG_NUM_VARS


######################################################################

def vec_from_svg_color(color):
    return gfx.vec3(color.red, color.green, color.blue) / 255.

######################################################################

def match_svg_color(color, carray):

    if not isinstance(carray, numpy.ndarray):
        carray = numpy.array(carray)

    assert len(carray.shape) == 2 and carray.shape[1] == 3

    dists = numpy.linalg.norm(carray - color, axis=1)

    i = dists.argmin()

    return i, carray[i]

######################################################################

def b2ple(array):
    return tuple([float(ai) for ai in array])

def b2xform(transform, z=0.0):
    return gfx.rigid_2d_matrix(transform.position, transform.angle, z)


######################################################################

class SimObject:

    def __init__(self, world=None):
        
        self.gfx_objects = []

        self.world = world
        self.body = None
        self.body_linear_mu = None
        self.body_angular_mu = None

        self.gfx_objects = []

    def sim_update(self, time, dt):

        if self.body is None:
            return
        
        mg = self.body.mass * GRAVITY
        
        if self.body_linear_mu is not None:
            self.body.ApplyForceToCenter(
                (-self.body_linear_mu * mg ) * self.body.linearVelocity,
                True)

        if self.body_angular_mu is not None:
            self.body.ApplyTorque(
                (-self.body_angular_mu * mg) * self.body.angularVelocity,
                True)

    def destroy(self):
        
        if self.world is not None and self.body is not None:
            self.world.DestroyBody(self.body)
            self.body = None
            
        if self.world is None and self.body is not None:
            raise RuntimeError(str(self) + 'has body but no world!')

        for obj in self.gfx_objects:
            obj.destroy()

        self.gfx_objects = []

    def reset(self):
        pass

    def render(self):

        myxform = None
        
        for obj in self.gfx_objects:
            if self.body is not None and hasattr(obj, 'model_pose'):
                if myxform is None:
                    myxform = b2xform(self.body.transform)                    
                obj.model_pose = myxform
            obj.render()
        
######################################################################                
                
class Pylon(SimObject):

    static_gfx_object = None

    def __init__(self, world, position, cname):

        super().__init__(world=world)
        
        assert cname in PYLON_COLOR_NAMES

        self.body_linear_mu = 0.9

        cidx = PYLON_COLOR_NAMES.index(cname)
        
        self.color = PYLON_COLORS[cidx]
        self.material_id = (1 << (cidx+1))

        self.initialize(position)
        
    def initialize(self, position):

        self.destroy()

        self.body = self.world.CreateDynamicBody(
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

        self.orig_position = position

        if self.static_gfx_object is None:
            self.static_gfx_object = gfx.IndexedPrimitives.cylinder(
                PYLON_RADIUS, PYLON_HEIGHT, 32, 1,
                self.color,
                pre_transform=gfx.tz(0.5*PYLON_HEIGHT))
            
        self.gfx_objects = [self.static_gfx_object]
        
    def reset(self):
        self.initialize(self.orig_position)

    def render(self):
        self.static_gfx_object.color = self.color
        self.static_gfx_object.material_id = self.material_id
        super().render()

    def destroy(self):
        self.gfx_objects = []
        super().destroy()
        
######################################################################

class Ball(SimObject):

    static_gfx_object = None

    def __init__(self, world, position):

        super().__init__(world=world)
        
        self.body_linear_mu = 0.01
        
        self.initialize(position)

    def initialize(self, position):

        self.destroy()
        
        self.orig_position = position
        
        self.body = self.world.CreateDynamicBody(
            position = b2ple(position),
            fixtures = B2D.b2FixtureDef(
                shape = B2D.b2CircleShape(radius=BALL_RADIUS),
                density = BALL_DENSITY,
                restitution = 0.98,
                friction = 0.95
            ),
            userData = self
        )

        if self.static_gfx_object is None:
        
            self.static_gfx_object = gfx.IndexedPrimitives.sphere(
                BALL_RADIUS, 32, 24, 
                BALL_COLOR,
                pre_transform=gfx.tz(BALL_RADIUS),
                specular_exponent=60.0,
                specular_strength=0.125,
                material_id=int(1 << 3))
        
        self.gfx_objects = [ self.static_gfx_object ]

    def reset(self):
        self.initialize(self.orig_position)

    def destroy(self):
        self.gfx_objects = []
        super().destroy()
        
######################################################################

class Wall(SimObject):

    def __init__(self, world, p0, p1):

        super().__init__(world=world)

        self.body_linear_mu = 0.9 

        self.initialize(p0, p1)

    def initialize(self, p0, p1):

        self.destroy()
        
        self.orig_p0 = p0
        self.orig_p1 = p1
        
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
            B2D.b2PolygonShape(box=(b2ple(0.5*numpy.array(dims[:2])))),
            B2D.b2PolygonShape(box=(r, r, (bx, 0), 0)),
            B2D.b2PolygonShape(box=(r, r, (-bx, 0), 0)),
        ]
            
        self.body = self.world.CreateDynamicBody(
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

        self.body_angular_mu = I-Ix
        
        self.body.massData = B2D.b2MassData(
            mass = mass,
            I = I
        )

        self.bx = bx
        self.dims = dims

        gfx_object = gfx.IndexedPrimitives.box(
            self.dims, CARDBOARD_COLOR,
            pre_transform=gfx.tz(WALL_Z + 0.5*self.dims[2]))

        self.gfx_objects = [gfx_object]

        for x in [-self.bx, self.bx]:

            block = gfx.IndexedPrimitives.box(
                gfx.vec3(BLOCK_SZ, BLOCK_SZ, BLOCK_SZ),
                BLOCK_COLOR,
                pre_transform=gfx.translation_matrix(
                    gfx.vec3(x, 0, 0.5*BLOCK_SZ)))

            self.gfx_objects.append(block)

    def reset(self):
        self.initialize(self.orig_p0, self.orig_p1)

######################################################################

class Box(SimObject):

    def __init__(self, world, dims, position, angle):

        super().__init__(world=world)
        
        self.initialize(dims, position, angle)

    def initialize(self, dims, position, angle):

        self.destroy()
        
        self.orig_position = position
        self.orig_angle = angle

        self.body = self.world.CreateDynamicBody(
            position = b2ple(position),
            angle = float(angle),
            fixtures = B2D.b2FixtureDef(
                shape = B2D.b2PolygonShape(box=(b2ple(0.5*numpy.array(dims[:2])))),
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

        self.body_linear_mu = 0.9
        self.body_angular_mu = I
        
        self.body.massData = B2D.b2MassData(
            mass = mass,
            I = I
        )

        self.dims = dims

        gfx_object = gfx.IndexedPrimitives.box(
            self.dims, CARDBOARD_COLOR,
            pre_transform=gfx.tz(0.5*self.dims[2]))

        self.gfx_objects = [gfx_object]
        
    def reset(self):
        self.initialize(self.dims, self.orig_position, self.orig_angle)

######################################################################

class Room(SimObject):

    floor_texture = None
    wall_texture = None
    
    def __init__(self, world, dims):

        super().__init__(world=world)

        self.initialize(dims)

    def initialize(self, dims):

        self.destroy()
        
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
        
        self.body = self.world.CreateStaticBody(
            userData = self,
            shapes = shapes
        )

        if self.floor_texture is None:
            self.floor_texture = gfx.load_texture(find_path('textures/floor_texture.png'))
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)

        if self.wall_texture is None:
            self.wall_texture = gfx.load_texture(find_path('textures/wall_texture.png'))
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
            verts, indices, ROOM_COLOR, texture=self.wall_texture)

        room_obj.specular_strength = 0.25

        self.gfx_objects = [ floor_obj, room_obj ]
        
    def reset(self):
        self.initialize(self.dims)


######################################################################

class TapeStrips(SimObject):

    def __init__(self, point_lists):
        super().__init__()
        self.point_lists = point_lists

    def initialize(self, point_lists):
        
        self.point_lists = point_lists

        r = TAPE_RADIUS
        offset = gfx.vec3(0, 0, r)

        self.gfx_objects = []

        dashes = []

        for points in point_lists:

            points = points.copy() # don't modify original points

            deltas = points[1:] - points[:-1]
            segment_lengths = numpy.linalg.norm(deltas, axis=1)
            tangents = deltas / segment_lengths.reshape(-1, 1)

            is_loop = (numpy.linalg.norm(points[-1] - points[0]) < TAPE_RADIUS)

            if not is_loop:

                segment_lengths[0] += TAPE_RADIUS
                points[0] -= TAPE_RADIUS * tangents[0]
                deltas[0] = points[1] - points[0]

                segment_lengths[-1] += TAPE_RADIUS
                points[-1] += TAPE_RADIUS * tangents[-1]
                deltas[-1] = points[-1] - points[-2]

                desired_parity = 1

            else:

                desired_parity = 0

            total_length = segment_lengths.sum()

            num_dashes = int(numpy.ceil(total_length / TAPE_DASH_SIZE))
            if num_dashes % 2 != desired_parity:
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

            # merge very close points in this
            deltas = points[1:] - points[:-1]
            norms = numpy.linalg.norm(deltas, axis=1)
            keep = numpy.hstack( ([ True ], norms > 1e-3) )

            points = points[keep]
            
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

                        linter, ldenom = gfx.line_intersect_2d(line_l, prev_line_l)
                        rinter, rdenom = gfx.line_intersect_2d(line_r, prev_line_r)

                        # TODO: parallel lines?

                        points_l.append( linter )
                        points_r.append( rinter ) 

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
                                           color=TAPE_COLOR,
                                           specular_exponent = 100.0,
                                           specular_strength = 0.05,
                                           material_id=int(1 << 0))

        self.gfx_objects.append(gfx_object)

    def reset(self):
        self.initialize(self.point_lists)
        

######################################################################

def linear_angular_from_wheel_lr(wheel_lr_vel):

    l, r = wheel_lr_vel

    linear = (r+l)/2
    angular = (r-l)/(2*ROBOT_WHEEL_OFFSET)

    return linear, angular

######################################################################

def wheel_lr_from_linear_angular(linear_angular):

    linear, angular = linear_angular

    l = linear - angular*ROBOT_WHEEL_OFFSET
    r = linear + angular*ROBOT_WHEEL_OFFSET

    return numpy.array([l, r])

######################################################################

def clamp_abs(quantity, limit):
    return numpy.clip(quantity, -limit, limit)
    
######################################################################

def iir_filter(meas, inputs, outputs, B, A):
    
    assert len(inputs) == len(B)
    assert len(outputs) == len(A)

    inputs[1:] = inputs[:-1]
    inputs[0] = meas

    output = numpy.dot(B, inputs) - numpy.dot(A, outputs)

    outputs[1:] = outputs[:-1]
    outputs[0] = output

    return output

######################################################################

class Robot(SimObject):

    def __init__(self, world):

        super().__init__(world=world)

        self.body = None

        # left and then right
        self.wheel_vel_cmd = numpy.array([0, 0], dtype=float)

        self.wheel_offsets = numpy.array([
            [ ROBOT_WHEEL_OFFSET, 0],
            [-ROBOT_WHEEL_OFFSET, 0]
        ], dtype=float)
        

        self.odom_linear_angular_vel_raw = numpy.zeros(2, dtype=numpy.float32)
        self.odom_linear_angular_vel_filtered = numpy.zeros(2, dtype=numpy.float32)
        
        self.odom_wheel_vel_raw = numpy.zeros((2, len(ODOM_FILTER_B)),
                                              dtype=numpy.float64)
        
        self.odom_wheel_vel_filtered = numpy.zeros((2, len(ODOM_FILTER_A)),
                                                   dtype=numpy.float64)
        
        self.odom_pose = Transform2D()
        self.initial_pose = Transform2D()

        self.desired_linear_angular_vel = numpy.zeros(2, dtype=numpy.float64)
        
        self.desired_linear_angular_vel_raw = numpy.zeros((2, len(SETPOINT_FILTER_B)),
                                                          dtype=numpy.float64)
        
        self.desired_linear_angular_vel_filtered = numpy.zeros((2, len(SETPOINT_FILTER_A)),
                                                               dtype=numpy.float64)
        
        self.desired_wheel_vel = numpy.zeros(2, dtype=numpy.float32)
        self.wheel_vel_integrator = numpy.zeros(2, dtype=numpy.float32)

        self.wheel_vel = numpy.zeros(2, dtype=numpy.float32)

        self.forward_vel = 0.0

        self.rolling_mu = 0.15
        
        self.motors_enabled = True

        self.bump = numpy.zeros(len(BUMP_ANGLE_RANGES), dtype=numpy.uint8)

        self.log_vars = numpy.zeros(LOG_NUM_VARS, dtype=numpy.float32)

        self.filter_setpoints = False

        self.initialize()
        
    def initialize(self, position=None, angle=None):

        self.destroy()

        if position is None:
            position = (0.0, 0.0)

        if angle is None:
            angle = 0.0

        self.colliders = set()

        self.bump[:] = 0
        
        self.orig_position = position
        self.orig_angle = angle

        self.odom_pose = Transform2D()
        self.initial_pose = Transform2D(position, angle)

        self.odom_tick = 0

        self.forward_vel = 0.0

        self.desired_linear_angular_vel[:] = 0
        self.desired_linear_angular_vel_raw[:] = 0
        self.desired_linear_angular_vel_filtered[:] = 0

        self.odom_wheel_vel_raw[:] = 0
        self.odom_wheel_vel_filtered[:] = 0
        self.wheel_vel_integrator[:] = 0
        self.wheel_vel[:] = 0

        self.body = self.world.CreateDynamicBody(
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

        ##################################################

        self.gfx_objects.append(
            gfx.IndexedPrimitives.cylinder(
                ROBOT_BASE_RADIUS, ROBOT_BASE_HEIGHT, 64, 1,
                ROBOT_BASE_COLOR,
                pre_transform=gfx.tz(0.5*ROBOT_BASE_HEIGHT + ROBOT_BASE_Z),
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
                    gfx.vec3(tx, 0, ROBOT_CAMERA_LENS_Z)),
                specular_exponent=40.0,
                specular_strength=0.75
            )
        )

        btop = ROBOT_BASE_Z + ROBOT_BASE_HEIGHT
        cbottom = ROBOT_CAMERA_BOTTOM_Z

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

            
        vdata = numpy.zeros((8, 3), dtype=numpy.float32)
        xy = vdata[:, :2]
        vdata[:, 2] = btop + 0.001

        a = 0.08
        b = 0.04
        c = -0.15
        d = 0.025
        cd = -0.11
        e = 0.105

        xy[0] = (c, b)
        xy[1] = (cd, 0)
        xy[2] = (c, -b)
        xy[3] = (d, a)
        xy[4] = (d, b)
        xy[5] = (d, -b)
        xy[6] = (d, -a)
        xy[7] = (e, 0)

        indices = [ [ 0, 1, 4 ],
                    [ 1, 5, 4 ],
                    [ 2, 5, 1 ],
                    [ 3, 4, 7 ],
                    [ 4, 5, 7 ],
                    [ 5, 6, 7 ] ]

        indices = numpy.array(indices, dtype=numpy.uint8)
        
        arrow = gfx.IndexedPrimitives.faceted_triangles(
            vdata, indices, gfx.vec3(0.8, 0.7, 0.0),
            specular_exponent=40.0,
            specular_strength=0.75
        )

        self.gfx_objects.append(arrow)

        self.bump_lites = []

        for theta_deg in [ 45, 0, -45 ]:

            theta_rad = theta_deg*numpy.pi/180
            r = 0.14
            tx = r*numpy.cos(theta_rad)
            ty = r*numpy.sin(theta_rad)

            lite = gfx.IndexedPrimitives.sphere(
                0.011, 16, 12,
                gfx.vec3(0.25, 0, 0),
                pre_transform=gfx.translation_matrix(gfx.vec3(tx, ty, btop)),
                specular_exponent=50.0,
                specular_strength=0.25,
                enable_lighting=True)

            self.gfx_objects.append(lite)
            self.bump_lites.append(lite)
        

    def reset(self):
        self.initialize(self.orig_position, self.orig_angle)

    def setup_log(self, logger):
        logger.add_variables(LOG_NAMES, self.log_vars)

    def update_log(self):

        l = self.log_vars

        l[LOG_ROBOT_POS_X] = self.body.position.x
        l[LOG_ROBOT_POS_Y] = self.body.position.y
        l[LOG_ROBOT_POS_ANGLE] = self.body.angle
        l[LOG_ROBOT_VEL_X] = self.body.linearVelocity.x
        l[LOG_ROBOT_VEL_Y] = self.body.linearVelocity.y
        l[LOG_ROBOT_VEL_ANGLE] = self.body.angularVelocity
        l[LOG_ROBOT_CMD_VEL_FORWARD] = self.desired_linear_angular_vel[0]
        l[LOG_ROBOT_CMD_VEL_ANGULAR] = self.desired_linear_angular_vel[1]
        l[LOG_ROBOT_CMD_VEL_LWHEEL] = self.desired_wheel_vel[0]
        l[LOG_ROBOT_CMD_VEL_RWHEEL] = self.desired_wheel_vel[1]
        l[LOG_ROBOT_VEL_FORWARD] = self.forward_vel
        l[LOG_ROBOT_VEL_LWHEEL] = self.wheel_vel[0]
        l[LOG_ROBOT_VEL_RWHEEL] = self.wheel_vel[1]
        l[LOG_ROBOT_MOTORS_ENABLED] = self.motors_enabled
        l[LOG_ROBOT_BUMP_LEFT:LOG_ROBOT_BUMP_LEFT+3] = self.bump

        rel_odom = self.initial_pose * self.odom_pose
        
        l[LOG_ODOM_POS_X] = rel_odom.position[0]
        l[LOG_ODOM_POS_Y] = rel_odom.position[1]
        l[LOG_ODOM_POS_ANGLE] = rel_odom.angle
        l[LOG_ODOM_VEL_RAW_LWHEEL] = self.odom_wheel_vel_raw[0,0]
        l[LOG_ODOM_VEL_RAW_RWHEEL] = self.odom_wheel_vel_raw[1,0]
        l[LOG_ODOM_VEL_RAW_FORWARD] = self.odom_linear_angular_vel_raw[0]
        l[LOG_ODOM_VEL_RAW_ANGLE] = self.odom_linear_angular_vel_raw[1]
        l[LOG_ODOM_VEL_FILT_LWHEEL] = self.odom_wheel_vel_filtered[0,0]
        l[LOG_ODOM_VEL_FILT_RWHEEL] = self.odom_wheel_vel_filtered[1,0]
        l[LOG_ODOM_VEL_FILT_FORWARD] = self.odom_linear_angular_vel_filtered[0]
        l[LOG_ODOM_VEL_FILT_ANGLE] = self.odom_linear_angular_vel_filtered[1]

    def sim_update(self, time, dt):

        dt_sec = dt.total_seconds()

        body = self.body

        current_tangent = body.GetWorldVector((1, 0))
        current_normal = body.GetWorldVector((0, 1))

        self.forward_vel = body.linearVelocity.dot(current_tangent)

        lateral_vel = body.linearVelocity.dot(current_normal)

        lateral_impulse = clamp_abs(-body.mass * lateral_vel,
                                    WHEEL_MAX_LATERAL_IMPULSE)

        body.ApplyLinearImpulse(lateral_impulse * current_normal,
                                body.position, True)

        for idx in range(2):
            if self.filter_setpoints:
                b, a = SETPOINT_FILTER_B, SETPOINT_FILTER_A
            else:
                b = [1, 0]
                a = [0]
            iir_filter(self.desired_linear_angular_vel[idx],
                       self.desired_linear_angular_vel_raw[idx],
                       self.desired_linear_angular_vel_filtered[idx],
                       b, a)
                                                              
        self.desired_wheel_vel = wheel_lr_from_linear_angular(
            self.desired_linear_angular_vel_filtered[:, 0]
        )

        for idx, side in enumerate([1.0, -1.0]):

            offset = B2D.b2Vec2(0, side * ROBOT_WHEEL_OFFSET)

            world_point = body.GetWorldPoint(offset)

            wheel_vel = body.GetLinearVelocityFromWorldPoint(world_point)

            wheel_fwd_vel = wheel_vel.dot(current_tangent)

            if numpy.abs(wheel_fwd_vel) > 1e-5:
                wheel_noise = numpy.random.normal(scale=ODOM_NOISE_STDDEV)
            else:
                wheel_noise = 0.0
            
            self.wheel_vel[idx] = wheel_fwd_vel
            
            meas_vel = wheel_fwd_vel + wheel_noise

            filtered_vel = iir_filter(meas_vel,
                                      self.odom_wheel_vel_raw[idx],
                                      self.odom_wheel_vel_filtered[idx],
                                      ODOM_FILTER_B,
                                      ODOM_FILTER_A)
            
            applied_force = 0.0
            
            if self.motors_enabled:

                wheel_vel_error = (
                    self.desired_wheel_vel[idx] - filtered_vel
                )

                self.wheel_vel_integrator[idx] = clamp_abs(
                    self.wheel_vel_integrator[idx] + wheel_vel_error * dt_sec,
                    WHEEL_VEL_INTEGRATOR_MAX)

                wheel_delta_vel_cmd = (WHEEL_VEL_KP * wheel_vel_error +
                                       WHEEL_VEL_KI * self.wheel_vel_integrator[idx])
                
                applied_force = clamp_abs(
                    wheel_delta_vel_cmd * body.mass / dt_sec,
                    WHEEL_MAX_FORWARD_FORCE)

            friction_force = -self.rolling_mu * wheel_fwd_vel * body.mass * GRAVITY

            total_force = (0.5*applied_force + friction_force)

            body.ApplyForce(total_force * current_tangent,
                            world_point, True)

        ##################################################

        self.odom_linear_angular_vel_raw[:] = linear_angular_from_wheel_lr(
            self.odom_wheel_vel_raw[:,0])
        
        self.odom_linear_angular_vel_filtered[:] = linear_angular_from_wheel_lr(
            self.odom_wheel_vel_filtered[:,0])

        self.odom_tick += 1
        
        if self.odom_tick % ODOM_FREQUENCY == 0:

            odt = dt_sec * ODOM_FREQUENCY
            odom_fwd = odt * self.odom_linear_angular_vel_raw[0]
            odom_spin = odt * self.odom_linear_angular_vel_raw[1]

            self.odom_pose.position += odom_fwd * self.odom_pose.matrix[:2, 0]
            self.odom_pose.angle += odom_spin

        ##################################################
        
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

    def render(self):

        for i in range(3):
            lite = self.bump_lites[i]
            bump = self.bump[i]
            if bump:
                lite.enable_lighting = False
                lite.color = gfx.vec3(1, 0, 0)
            else:
                lite.enable_lighting = True
                lite.color = gfx.vec3(0.25, 0, 0)
        
        super().render()
        
        
                                
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

        self.world = B2D.b2World(gravity=(0, 0), doSleep=True)
        self.world.contactListener = self

        self.dims = numpy.array([-1, -1], dtype=numpy.float32)

        self.robot = Robot(self.world)

        self.room = Room(self.world, [4.0, 4.0])

        self.modification_counter = 0

        self.tape_strips = None
        
        self.objects = [ self.robot, self.room ]

        self.dt = timedelta(milliseconds=10)
        
        self.physics_ticks_per_update = 4

        self.logger = Logger(self.dt * self.physics_ticks_per_update)
        self.robot.setup_log(self.logger)

        self.velocity_iterations = 6
        self.position_iterations = 2
        
        self.sim_time = timedelta()
        self.sim_ticks = 0

        self.svg_filename = None

        self.framebuffer = None

        self.detections = None

        print('created the world!')

    def set_dims(self, room_width, room_height):
        self.dims = numpy.array([room_width, room_height],
                                dtype=numpy.float32)
        self.room.initialize(self.dims)
        self.svg_filename = None
        self.modification_counter += 1

    def add_object(self, obj):
        self.objects.append(obj)
        self.svg_filename = None
        self.modification_counter += 1
        return obj

    def add_box(self, dims, pctr, theta):
        self.add_object(Box(self.world, dims, pctr, theta))

    def add_pylon(self, position, cname):
        self.add_object(Pylon(self.world, position, cname))

    def add_pylon(self, position, cname):
        self.add_object(Pylon(self.world, position, cname))

    def add_ball(self, position):
        self.add_object(Ball(self.world, position))

    def add_tape_strip(self, points):
        if self.tape_strips is None:
            self.tape_strips = self.add_object(TapeStrips([points]))
        else:
            self.tape_strips.point_lists.append(points)
            self.modification_counter += 1

    def add_wall(self, p0, p1):
        self.add_object(Wall(self.world, numpy.array(p0), numpy.array(p1)))

    def reset(self, reload_svg=True):
        self.logger.finish()
        if reload_svg and self.svg_filename is not None:
            self.clear()
            self.load_svg(self.svg_filename)
        else:
            self.sim_time = timedelta(0)
            self.sim_ticks = 0
            for obj in self.objects:
                obj.reset()
        self.modification_counter += 1

    def clear(self):

        self.sim_time = timedelta(0)
        self.sim_ticks = 0
        self.modification_counter += 1

        assert self.robot == self.objects[0]

        for obj in self.objects[2:]:
            obj.destroy()
            
        self.objects[:] = [self.robot, self.room]

    def load_svg(self, svgfile):

        svg = svgelements.SVG.parse(svgfile, color='none')
        print('parsed', svgfile)

        scl = 1e-2
        
        xform = SvgTransformer(svg.viewbox.viewbox_width,
                               svg.viewbox.viewbox_height, scl)

        self.set_dims(*xform.dims)

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
                fcolor = vec_from_svg_color(item.fill)

            if item.stroke.value is not None:
                scolor = vec_from_svg_color(item.stroke)

            if isinstance(item, svgelements.Rect):

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

                    self.add_box(dims, pctr, theta)
                
            elif isinstance(item, svgelements.Circle):
                
                cidx, color = match_svg_color(fcolor, CIRCLE_COLORS)

                position = xform.transform(item.cx, item.cy)

                if cidx == 0:
                    self.add_ball(position)
                else:
                    self.add_pylon(position,
                                   PYLON_COLOR_NAMES[cidx-1])
                                        
            elif isinstance(item, svgelements.SimpleLine):

                p0 = xform.transform(item.x1, item.y1)
                p1 = xform.transform(item.x2, item.y2)

                cidx, color = match_svg_color(scolor, LINE_COLORS)

                if cidx == 0:
                    
                    points = numpy.array([p0, p1])
                    self.add_tape_strip(points)
                    
                else:

                    self.add_wall(p0, p1)

            elif isinstance(item, svgelements.Polyline):
                
                points = numpy.array(
                    [xform.transform(p.x, p.y) for p in item.points])

                self.add_tape_strip(points)

            elif isinstance(item, svgelements.Polygon):

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

                robot_init_position = ctr + 0.5 * dist * normal
                robot_init_angle = numpy.arctan2(normal[1], normal[0])

            else:
                
                print('*** warning: ignoring SVG item:', item, '***')
                continue

        assert self.robot == self.objects[0]
        assert self.room == self.objects[1]
        
        self.initialize_robot(robot_init_position,
                              robot_init_angle)

        if self.tape_strips is not None:
            self.tape_strips.reset()

        self.svg_filename = os.path.abspath(svgfile)

    def initialize_robot(self, pos, angle):
        self.robot.initialize(pos, angle)

    def update(self):

        for i in range(self.physics_ticks_per_update):

            for obj in self.objects:
                obj.sim_update(self.sim_time, self.dt)

            self.world.Step(self.dt.total_seconds(),
                            self.velocity_iterations,
                            self.position_iterations)

            self.world.ClearForces()

            self.sim_time += self.dt
            self.sim_ticks += 1

        self.robot.update_log()
        self.logger.append_log_row()


    def kick_ball(self):

        for obj in self.objects:
            if isinstance(obj, Ball):
                kick_impulse = B2D.b2Vec2(1, 0)
                wdist = None
                for other in self.objects:
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
        
    def PreSolve(self, contact, old_manifold):

        other = None

        if contact.fixtureA.body == self.robot.body:
            other = contact.fixtureB.body
        elif contact.fixtureB.body == self.robot.body:
            other = contact.fixtureA.body

        if other is not None:
            self.robot.colliders.add(other.userData)
    
            
