######################################################################
#
# robosim_core.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import numpy
import graphics as gfx
import Box2D as B2D
import svgelements as se
import os
import robosim_logging as rlog
from transform2d import Transform2D

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

GRAVITY = 9.8

WHEEL_MAX_LATERAL_IMPULSE = 0.5 # m/(s*kg)
WHEEL_MAX_FORWARD_FORCE = 10.0 # N
WHEEL_VEL_KP = 0.95 # dimensionless
WHEEL_VEL_KI = 4.0 # 1/s - higher means more overshoot
WHEEL_VEL_INTEGRATOR_MAX = 0.005 # m - affects both overshoot and size of steady-state error

ODOM_WHEEL_VEL_STDDEV = 0.01

ODOM_FILTER_A = numpy.array([-0.41421356],
                            dtype=numpy.float64)

ODOM_FILTER_B = numpy.array([0.29289322, 0.29289322],
                            dtype=numpy.float64)

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

    def __init__(self):
        
        self.gfx_objects = []
        
        self.body = None
        self.body_linear_mu = None
        self.body_angular_mu = None

    def sim_update(self, world, time, dt):

        if self.body_linear_mu is not None:
            self.body.ApplyForceToCenter(
                -self.body_linear_mu * self.body.linearVelocity * self.body.mass * GRAVITY,
                True)

        if self.body_angular_mu is not None:
            self.body.ApplyTorque(
                -self.body_angular_mu * self.body.angularVelocity * self.body.mass * GRAVITY,
                True)

######################################################################                
                
class Pylon(SimObject):

    static_gfx_object = None

    def __init__(self, world, position, color, material_id):

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

        self.body_linear_mu = 0.9

        self.color = color

        self.material_id = material_id

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

        self.body_linear_mu = 0.01
        
        
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

        self.body_linear_mu = 0.9 
        self.body_angular_mu = I
        
        self.body.massData = B2D.b2MassData(
            mass = mass,
            I = I
        )

        self.bx = bx
        self.dims = dims

        

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

        self.body_linear_mu = 0.9
        self.body_angular_mu = I
        
        self.body.massData = B2D.b2MassData(
            mass = mass,
            I = I
        )

        self.dims = dims

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


######################################################################

class TapeStrips(SimObject):

    def __init__(self, point_lists):

        super().__init__()

        self.point_lists = point_lists

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

class Robot(SimObject):

    def __init__(self, world):

        super().__init__()

        self.world = world
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

        self.desired_linear_angular_vel = numpy.zeros(2, dtype=numpy.float32)
        self.desired_wheel_vel = numpy.zeros(2, dtype=numpy.float32)
        self.wheel_vel_integrator = numpy.zeros(2, dtype=numpy.float32)

        self.wheel_vel = numpy.zeros(2, dtype=numpy.float32)

        self.forward_vel = 0.0

        self.rolling_mu = 0.15
        
        self.motors_enabled = True

        self.bump = numpy.zeros(len(BUMP_ANGLE_RANGES), dtype=numpy.uint8)

        self.colliders = set()

        self.log_vars = numpy.zeros(LOG_NUM_VARS, dtype=numpy.float32)

        self.initialize()
        
    def initialize(self, position=None, angle=None):

        if self.body is not None:
            self.world.DestroyBody(self.body)
            self.body = None

        if position is None:
            position = (0.0, 0.0)

        if angle is None:
            angle = 0.0

        self.odom_pose = Transform2D()
        self.initial_pose = Transform2D(position, angle)

        self.odom_wheel_vel_raw[:] = 0
        self.odom_wheel_vel_filtered[:] = 0
        self.wheel_vel_integrator[:] = 0

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

    def sim_update(self, world, time, dt):

        body = self.body

        current_tangent = body.GetWorldVector((1, 0))
        current_normal = body.GetWorldVector((0, 1))

        self.forward_vel = body.linearVelocity.dot(current_tangent)

        lateral_vel = body.linearVelocity.dot(current_normal)

        lateral_impulse = clamp_abs(-body.mass * lateral_vel,
                                    WHEEL_MAX_LATERAL_IMPULSE)

        body.ApplyLinearImpulse(lateral_impulse * current_normal,
                                body.position, True)

        self.desired_wheel_vel = wheel_lr_from_linear_angular(
            self.desired_linear_angular_vel
        )
        
        wheel_noise = numpy.random.normal(size=2, scale=ODOM_WHEEL_VEL_STDDEV)

        # shift past wheel velocities down one
        self.odom_wheel_vel_raw[:, 1:] = self.odom_wheel_vel_raw[:, :-1]


        for idx, side in enumerate([1.0, -1.0]):

            offset = B2D.b2Vec2(0, side * ROBOT_WHEEL_OFFSET)

            world_point = body.GetWorldPoint(offset)

            wheel_vel = body.GetLinearVelocityFromWorldPoint(world_point)

            wheel_fwd_vel = wheel_vel.dot(current_tangent)

            self.wheel_vel[idx] = wheel_fwd_vel
            
            meas_vel = wheel_fwd_vel + wheel_noise[idx]
                                
            self.odom_wheel_vel_raw[idx, 0] = meas_vel


            filtered_vel = ( numpy.dot(ODOM_FILTER_B, self.odom_wheel_vel_raw[idx]) -
                             numpy.dot(ODOM_FILTER_A, self.odom_wheel_vel_filtered[idx]) )

            if idx == 0:
                print('got meas', meas_vel)
                print('x:', self.odom_wheel_vel_raw[idx])
                print('y:', self.odom_wheel_vel_filtered[idx])
                print('output:', filtered_vel)
                print()
            

            self.odom_wheel_vel_filtered[idx, 1:] = self.odom_wheel_vel_filtered[idx, :-1]
            self.odom_wheel_vel_filtered[idx, 0] = filtered_vel
            
            applied_force = 0.0
            
            if self.motors_enabled:

                wheel_vel_error = (
                    self.desired_wheel_vel[idx] - filtered_vel
                )

                print('wheel_vel_error[{}] = {}'.format(
                    idx, wheel_vel_error))
                

                self.wheel_vel_integrator[idx] = clamp_abs(
                    self.wheel_vel_integrator[idx] + wheel_vel_error * dt,
                    WHEEL_VEL_INTEGRATOR_MAX)

                wheel_delta_vel_cmd = (WHEEL_VEL_KP * wheel_vel_error +
                                       WHEEL_VEL_KI * self.wheel_vel_integrator[idx])
                
                applied_force = clamp_abs(
                    wheel_delta_vel_cmd * body.mass / dt,
                    WHEEL_MAX_FORWARD_FORCE)

            friction_force = -self.rolling_mu * wheel_fwd_vel * body.mass * GRAVITY
            
            body.ApplyForce((0.5*applied_force + friction_force) * current_tangent,
                            world_point, True)

            #body.ApplyLinearImpulse((0.5*applied_force + friction_force)*dt * current_tangent,
            #world_point, True)
            
        ##################################################

        self.odom_linear_angular_vel_raw[:] = linear_angular_from_wheel_lr(self.odom_wheel_vel_raw[:,0])
        self.odom_linear_angular_vel_filtered[:] = linear_angular_from_wheel_lr(self.odom_wheel_vel_filtered[:,0])

        odom_fwd = dt * self.odom_linear_angular_vel_raw[0]
        odom_spin = dt * self.odom_linear_angular_vel_raw[1]

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
        self.objects = [ self.robot ]

        self.dt = 0.01 # 100 HZ
        self.physics_ticks_per_update = 4

        self.logger = rlog.Logger(self.dt)
        self.robot.setup_log(self.logger)

        self.velocity_iterations = 6
        self.position_iterations = 2
        
        self.remaining_sim_time = 0.0
        self.sim_time = 0.0
        self.sim_ticks = 0

        self.svg_filename = None

        self.framebuffer = None

        self.detections = None

        print('created the world!')

    def reload(self):

        self.logger.finish()

        self.clear()

        if self.svg_filename is not None:
            self.load_svg(self.svg_filename)

    def clear(self):

        self.remaining_sim_time = 0.0
        self.sim_time = 0.0
        self.sim_ticks = 0

        assert self.robot == self.objects[0]

        for obj in self.objects[1:]:
            if obj.body is not None:
                self.world.DestroyBody(obj.body)
            
        self.objects = [self.robot]
        

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
                fcolor = vec_from_svg_color(item.fill)

            if item.stroke.value is not None:
                scolor = vec_from_svg_color(item.stroke)

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
                
                cidx, color = match_svg_color(fcolor, CIRCLE_COLORS)

                position = xform.transform(item.cx, item.cy)

                if cidx == 0:
                    self.objects.append(Ball(self.world, position))
                else:
                    self.objects.append(Pylon(self.world, position,
                                              color, int(1 << cidx)))
                                        
            elif isinstance(item, se.SimpleLine):

                p0 = xform.transform(item.x1, item.y1)
                p1 = xform.transform(item.x2, item.y2)

                cidx, color = match_svg_color(scolor, LINE_COLORS)

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

                robot_init_position = ctr + 0.5 * dist * normal
                robot_init_angle = numpy.arctan2(normal[1], normal[0])

            else:
                
                print('*** warning: ignoring SVG item:', item, '***')
                continue

        if len(tape_point_lists):
            self.objects.append(TapeStrips(tape_point_lists))

        assert self.robot == self.objects[0]

        self.robot.initialize(robot_init_position,
                              robot_init_angle)
        

        self.svg_filename = os.path.abspath(svgfile)

    def update(self):

        for i in range(self.physics_ticks_per_update):

            for obj in self.objects:
                obj.sim_update(self.world, self.sim_time, self.dt)

            self.world.Step(self.dt,
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

######################################################################

def _test_load_environment():


    sim = RoboSim()
    sim.load_svg('environments/first_environment.svg')

    print('sim objects:')

    for sim_object in sim.objects:
        print('  ' + type(sim_object).__name__)
        if sim_object.body is not None:
            print('    transform:',
                  sim_object.body.position, sim_object.body.angle)

######################################################################

if __name__ == '__main__':

    _test_load_environment()
    
            
