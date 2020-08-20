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

import glfw
from CleanGL import gl

import Box2D as B2D

# DONE: teardown graphics
# DONE: teardown sim
# DONE: reset sim and env
# DONE: sim robot
# DONE: virtual bump sensors
# TODO: implement renderbuffers in graphics
# TODO: robot camera

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

TAPE_POLYGON_OFFSET = 0.001

BALL_MASS = 0.05
BALL_AREA = 2*numpy.pi*BALL_RADIUS**2
BALL_DENSITY = BALL_MASS / BALL_AREA

WALL_THICKNESS = 0.005
WALL_HEIGHT = 0.5
WALL_Z = 0.03

CARDBOARD_DENSITY_PER_M2 = 0.45

BLOCK_MASS = 0.5
BLOCK_SZ = 0.1
BLOCK_COLOR = gfx.vec3(0.5, 0.25, 0)

ROOM_HEIGHT = 1.5

ROOM_COLOR = gfx.vec3(1, 0.97, 0.93)

ROBOT_BASE_RADIUS = 0.5*0.36
ROBOT_BASE_HEIGHT = 0.12
ROBOT_BASE_Z = 0.01
ROBOT_BASE_MASS = 2.35
ROBOT_BASE_I = 0.5*ROBOT_BASE_MASS*ROBOT_BASE_RADIUS**2

ROBOT_BASE_COLOR = gfx.vec3(0.1, 0.1, 0.1)

ROBOT_CAMERA_DIMS = gfx.vec3(0.08, 0.25, 0.04)
ROBOT_CAMERA_Z = 0.27

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


def vec_from_color(color):
    return gfx.vec3(color.red, color.green, color.blue) / 255.

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
                specular_exponent=40.0,
                specular_strength=0.5)
        
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

class TapeStrip(SimObject):

    def __init__(self, points):

        super().__init__()

        self.points = points

    def init_render(self):


        r = TAPE_RADIUS
        offset = gfx.vec3(0, 0, r)

        prev_line_l = None
        prev_line_r = None

        points_l = []
        points_r = []

        for i, p0 in enumerate(self.points[:-1]):

            p1 = self.points[i+1]

            tangent = gfx.normalize(p1 - p0)
            normal = numpy.array([-tangent[1], tangent[0]], dtype=numpy.float32)

            line = gfx.vec3(normal[0], normal[1], -numpy.dot(normal, p0))

            line_l = line - offset
            line_r = line + offset

            if i == 0:

                points_l.append( p0 + r * (normal - tangent) )
                points_r.append( p0 + r * (-normal - tangent) )

            else:

                if abs(numpy.dot(line_l[:2], prev_line_l[:2])) > 0.999:

                    points_l.append( p0 + r * normal )
                    points_r.append( p0 - r * normal )

                else:

                    points_l.append( line_intersect(line_l, prev_line_l) )
                    points_r.append( line_intersect(line_r, prev_line_r) )

            if i == len(self.points) - 2:

                points_l.append( p1 + r * (normal + tangent) )
                points_r.append( p1 + r * (-normal + tangent) )

            prev_line_l = line_l
            prev_line_r = line_r

        points_l = numpy.array(points_l)
        points_r = numpy.array(points_r)

        vdata = numpy.zeros((2*len(self.points), 8), dtype=numpy.float32)

        vdata[0::2, :2] = points_l
        vdata[1::2, :2] = points_r
        vdata[:, 2] = TAPE_POLYGON_OFFSET
        vdata[:, 5] = 1
        vdata[:, 6:8] = vdata[:, 0:2]

        gfx_object = gfx.IndexedPrimitives(vdata, gl.TRIANGLE_STRIP,
                                           indices=None,
                                           color=TAPE_COLOR)

        gfx_object.specular_exponent = 100.0
        gfx_object.specular_strength = 0.05
            
        self.gfx_objects = [ gfx_object ]


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

        self.rolling_mu = 0.5
        
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

                self.desired_wheel_velocity_filtered[idx] = 0.0
                
                body.ApplyForce(-self.rolling_mu*wheel_fwd_velocity * current_normal,
                                world_point, True)

        self.bump[:] = 0

        transformA = self.body.transform

        finished_colliders = set()

        for collider in self.colliders:
            
            transformB = collider.body.transform

            min_dist = None
            min_pointA = None
            
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

                    if min_dist is None or distance < min_dist:
                        min_dist = distance
                        min_pointA = pointA

            if min_dist > BUMP_DIST:
                
                finished_colliders.add(collider)

            else:

                lx, ly = self.body.GetLocalPoint(min_pointA)

                theta = numpy.arctan2(ly, lx)

                in_range = ( (theta >= BUMP_ANGLE_RANGES[:,0]) &
                             (theta <= BUMP_ANGLE_RANGES[:,1]) )

                self.bump |= in_range
                    
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

        self.world = B2D.b2World(gravity=(0, 0), doSleep=True)
        self.world.contactListener = self

        self.dims = numpy.array([-1, -1], dtype=numpy.float32)

        self.objects = []
        self.robot = None

        self.dt = 0.01

        self.velocity_iterations = 6
        self.position_iterations = 2

        
        self.remaining_sim_time = 0.0
        self.sim_time = 0.0

        self.svg_filename = None
        
        print('created the world!')

    def reload(self):

        self.clear()

        if self.svg_filename is not None:
            self.load_svg(self.svg_filename)

    def clear(self):

        self.remaining_sim_time = 0.0
        self.sim_time = 0.0
        
        for obj in self.objects:
            
            if obj.body is not None:
                self.world.DestroyBody(obj.body)
                
            obj.destroy_render()
            
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
                    self.objects.append(TapeStrip(points))
                    
                else:

                    self.objects.append( Wall(self.world, p0, p1) )
                
            elif isinstance(item, se.Polyline):
                
                points = numpy.array(
                    [xform.transform(p.x, p.y) for p in item.points])
                
                self.objects.append(TapeStrip(points))

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

    def destroy_render(self):
        for obj in self.objects:
            obj.destroy_render()
            
    def render(self):
        for obj in self.objects:
            obj.render()

    def update(self, time_since_last_update):


        self.remaining_sim_time += time_since_last_update

        #print('simulating {} worth of real time'.format(
        #    self.remaining_sim_time))
        
        while self.remaining_sim_time >= self.dt:
            
            self.sim_time += self.dt
            self.remaining_sim_time -= self.dt

            now = glfw.get_time()
            
            for obj in self.objects:
                obj.sim_update(self.world, self.sim_time, self.dt)

            self.world.Step(self.dt,
                            self.velocity_iterations,
                            self.position_iterations)

            self.world.ClearForces()

            elapsed = glfw.get_time() - now
            #print('  sim step took', elapsed, 'seconds')


    def PreSolve(self, contact, old_manifold):

        other = None

        if contact.fixtureA.body == self.robot.body:
            other = contact.fixtureB.body
        elif contact.fixtureB.body == self.robot.body:
            other = contact.fixtureA.body

        if other is None:
            return

        self.robot.colliders.add(other.userData)
        
######################################################################

class RoboSimApp(gfx.GlfwApp):

    def __init__(self, sim):

        super().__init__()

        self.create_window('Robot simulator', 640, 480, units='window')

        gfx.IndexedPrimitives.DEFAULT_SPECULAR_EXPONENT = 100.0
        gfx.IndexedPrimitives.DEFAULT_SPECULAR_STRENGTH = 0.1

        sim.init_render()

        gl.Enable(gl.CULL_FACE)
        gl.Enable(gl.FRAMEBUFFER_SRGB)

        self.sim = sim

        self.perspective = None
        self.view = None

        self.camR = numpy.array([
            [ 0, -1, 0, 0 ],
            [ 0,  0, 1, 0 ],
            [-1,  0, 0, 0 ],
            [ 0,  0, 0, 1 ]
        ], dtype=numpy.float32)

        self.xrot = 0
        self.yrot = 0

        self.mouse_pos = numpy.array(self.framebuffer_size/2, dtype=numpy.float32)

        self.handle_mouse_rot()

        self.animating = True
        self.was_animating = False

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
                                desired_vel = 4.0*diff/dist
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
        #self.xrot = gfx.mix(numpy.pi/2, 0, numpy.clip(foo[1], 0, 1))
        self.xrot = gfx.mix(numpy.pi/2, -numpy.pi/2, numpy.clip(foo[1], 0, 1))

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
            
        
        if numpy.any(la):
            self.sim.robot.motors_enabled = True

        
    def render(self):

        gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
        gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        gl.Viewport(0, 0,
                   self.framebuffer_size[0],
                   self.framebuffer_size[1])

        gl.UseProgram(gfx.IndexedPrimitives.program)
        
        if self.perspective is None:
            
            w, h = self.framebuffer_size
            aspect = w / max(h, 1)

            self.perspective = gfx.perspective_matrix(
                45, aspect, 0.005, 25.0)

            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['perspective'],
                            self.perspective)

        view_robot = True

        if self.view is None or view_robot:

            if view_robot:

                M = b2xform(self.sim.robot.body.transform,
                            ROBOT_CAMERA_Z + 0.5*ROBOT_CAMERA_DIMS[2])

                #M = numpy.dot(M, gfx.translation_matrix(
                #    gfx.vec3(-0.95*ROBOT_CAMERA_DIMS[0], 0, 0)))

                M = numpy.linalg.inv(M)
                M = numpy.dot(self.camR, M)

                self.view = M

            else:

                Rx = gfx.rotation_matrix(self.xrot, gfx.vec3(1, 0, 0))
                Ry = gfx.rotation_matrix(self.yrot, gfx.vec3(0, 1, 0))

                R_mouse = numpy.dot(Rx, Ry)

                w, h = self.sim.dims
                m = numpy.linalg.norm([w, h])

                self.view = gfx.look_at(eye=gfx.vec3(0.5*w, 0.5*h - 0.5*m, 0.75*ROOM_HEIGHT),
                                        center=gfx.vec3(0.5*w, 0.5*h, 0.75*ROOM_HEIGHT),
                                        up=gfx.vec3(0, 0, 1),
                                        Rextra=R_mouse)


            '''
            rx, ry = self.sim.robot.body.position

            self.view = gfx.look_at(eye=gfx.vec3(rx, ry-1.5, 0),
                                    center=gfx.vec3(rx, ry, 0),
                                    up=gfx.vec3(0, 0, 1),
                                    Rextra=R_mouse)
            '''

            view_pos = -numpy.dot(numpy.linalg.inv(self.view[:3, :3]),
                                  self.view[:3, 3])

            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['viewPos'], view_pos)
            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['view'], self.view)

        gl.Enable(gl.DEPTH_TEST)

        self.sim.render()
        #self.sim.robot.render()


        
        
        

######################################################################
            
def _test_load_environment():


    sim = RoboSim()
    sim.load_svg('environments/first_environment.svg')

    #return

    app = RoboSimApp(sim)

    app.run()

        
if __name__ == '__main__':
    
    _test_load_environment()
