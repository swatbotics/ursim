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

import glfw
from CleanGL import gl

import Box2D as B2D

# TODO: teardown graphics
# TODO: teardown sim
# TODO: reset sim and env

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

BALL_MASS = 0.1
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

    def render(self):
        
        for obj in self.gfx_objects:
            
            if self.body is not None and hasattr(obj, 'model_pose'):
                
                obj.model_pose = b2xform(self.body.transform)

            obj.render()

    def setup_sim(self, world):
        pass

    def sim_update(self, world, time):

        if self.body is not None:

            if self.body_linear_mu:
                self.body.ApplyForce(
                    -self.body_linear_mu * self.body.linearVelocity,
                    self.body.worldCenter, True)

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
                friction = 0.95,
            )
        )

        self.body.massData = B2D.b2MassData(mass=PYLON_MASS,
                                            I=PYLON_I)

        self.body_linear_mu = 0.9 * PYLON_MASS * 10.0

        print('pylon has radius={}, mass={}, I={}'.format(
            PYLON_RADIUS, PYLON_MASS, PYLON_I))

        self.color = color

    def init_render(self):

        gfx_object = gfx.IndexedPrimitives.cylinder(
            PYLON_RADIUS, PYLON_HEIGHT, 32, 1,
            self.color,
            pre_transform=tz(0.5*PYLON_HEIGHT))

        self.gfx_objects = [gfx_object]

######################################################################

class Ball(SimObject):

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
            )
        )

        self.body_linear_mu = 0.05 * BALL_MASS * 10.0
        
        print('ball has radius={}, mass={} ({}), I={}'.format(
            BALL_RADIUS, BALL_MASS, self.body.mass, self.body.inertia))
        

    def init_render(self):
        
        gfx_object = gfx.IndexedPrimitives.sphere(
            BALL_RADIUS, 32, 24, 
            BALL_COLOR,
            pre_transform=tz(BALL_RADIUS),
            specular_exponent=40.0,
            specular_strength=0.5)
        
        self.gfx_objects = [ gfx_object ]
        
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
                                            friction=0.95)
        )


        rho = CARDBOARD_DENSITY_PER_M2

        mx = rho * (dims[1] * dims[2])
        Ix = mx * dims[0]**2 / 12

        Ib = BLOCK_MASS*BLOCK_SZ**2/6

        mass = mx + 2*BLOCK_MASS 
        I = Ix + 2*(Ib + BLOCK_MASS*bx**2)

        print('this wall has dims={}, mass={}, I={}'.format(dims, mass, I))

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
                friction = 0.95
            )
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

        print('this box has dims={}, mass={}, I={}'.format(dims, mass, I))

        self.body_linear_mu = 0.9 * mass * 10.0
        self.body_angular_mu = I * 10.0
        
        self.body.massData = B2D.b2MassData(
            mass = mass,
            I = I
        )

        self.dims = dims

    def init_render(self):

        gfx_object = gfx.IndexedPrimitives.box(
            self.dims, CARDBOARD_COLOR)

        self.gfx_objects = [gfx_object]
        
        
######################################################################

def line_intersect(l1, l2):

    l3 = numpy.cross(l1, l2)
    return l3[:2] / l3[2]

######################################################################

class Room(SimObject):

    floor_texture = None
    
    def __init__(self, world, dims):

        super().__init__()
        
        self.dims = dims

        self.body = world.CreateStaticBody()

        # clockwise is inwards
        w, h = b2ple(dims)
        
        verts = [
            (0., 0.),
            (0., h),
            (w, h),
            (w, 0.)
        ]
        
        self.body.CreateLoopFixture(vertices=verts)

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

class RoboSim:

    def __init__(self):

        self.world = B2D.b2World(gravity=(0, 0), doSleep=True)

        self.dims = numpy.array([-1, -1], dtype=numpy.float32)

        self.objects = []

        self.dt = 0.01

        self.velocity_iterations = 6
        self.position_iterations = 2

        
        self.remaining_sim_time = 0.0
        self.sim_time = 0.0


        
        print('created the world!')

    def load_svg(self, svgfile):

        svg = se.SVG.parse(svgfile, color='none')
        print('parsed', svgfile)

        scl = 1e-2
        
        xform = SvgTransformer(svg.viewbox.viewbox_width,
                               svg.viewbox.viewbox_height, scl)
        
        self.dims = xform.dims

        self.objects.append(Room(self.world, self.dims))

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

                if numpy.all(fcolor == 1):
                    continue

                dims = gfx.vec3(w, h, min(w, h))
                pctr = xform.transform(cx, cy)
                pfwd = xform.transform(cx+1, cy)
                delta = pfwd-pctr
                theta = numpy.arctan2(delta[1], delta[0])

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
                
            else:
                
                print('*** warning: ignoring SVG item:', item, '***')
                continue
        

    def render(self):

        for obj in self.objects:
            obj.render()

    def update(self, time_since_last_update):

        self.remaining_sim_time += time_since_last_update

        while self.remaining_sim_time >= self.dt:
            
            self.sim_time += self.dt
            self.remaining_sim_time -= self.dt

            print('sim update, t={}'.format(self.sim_time))

            for obj in self.objects:
                obj.sim_update(self.world, self.sim_time)

            self.world.Step(self.dt,
                            self.velocity_iterations,
                            self.position_iterations)

            self.world.ClearForces()

            print('clearing forces')

                
        
######################################################################

class RoboSimApp(gfx.GlfwApp):

    def __init__(self, sim):

        super().__init__()

        self.create_window('Robot simulator', 1200, 1080)

        gfx.IndexedPrimitives.DEFAULT_SPECULAR_EXPONENT = 100.0
        gfx.IndexedPrimitives.DEFAULT_SPECULAR_STRENGTH = 0.1

        for obj in sim.objects:
            obj.init_render()

        gl.Enable(gl.CULL_FACE)
        gl.Enable(gl.FRAMEBUFFER_SRGB)

        self.sim = sim

        self.perspective = None
        self.view = None

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

        if key == glfw.KEY_B:
            for obj in self.sim.objects:
                if isinstance(obj, Ball):
                    kick_impulse = B2D.b2Vec2(1, 0)
                    wdist = None
                    for other in self.sim.objects:
                        if isinstance(other, Wall):
                            diff = other.body.position - obj.body.position
                            dist = diff.length
                            if wdist is None or dist < wdist:
                                wdist = dist
                                desired_vel = 20*diff/dist
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

        pass

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

            self.perspective = gfx.perspective_matrix(45.0, aspect, 0.1, 25.0)

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

            view_pos = -numpy.dot(numpy.linalg.inv(self.view[:3, :3]),
                                  self.view[:3, 3])

            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['viewPos'], view_pos)
            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['view'], self.view)

        gl.Enable(gl.DEPTH_TEST)

        self.sim.render()


        
        
        

######################################################################
            
def _test_load_environment():


    sim = RoboSim()
    sim.load_svg('environments/first_environment.svg')

    #return

    app = RoboSimApp(sim)

    app.run()

        
if __name__ == '__main__':
    
    _test_load_environment()
