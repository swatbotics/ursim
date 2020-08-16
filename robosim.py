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

import matplotlib.pyplot as plt

import glfw
from CleanGL import gl

TAPE_COLOR = gfx.vec3(0.3, 0.3, 0.9)

CARDBOARD_COLOR = gfx.vec3(0.8, 0.7, 0.6)

PYLON_COLORS = [
    gfx.vec3(1.0, 0.5, 0),
    gfx.vec3(0, 0.8, 0)
]

PYLON_RADIUS = 0.05
PYLON_HEIGHT = 0.20

TAPE_RADIUS = 0.025

TAPE_POLYGON_OFFSET = 0.001


WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.5


def vec_from_color(color):
    return gfx.vec3(color.red, color.green, color.blue) / 255.

######################################################################

class Pylon:

    gfx_object = None

    def __init__(self, position, color):
        
        assert position.shape == (2,) and position.dtype == numpy.float32
        assert color.shape == (3,) and color.dtype == numpy.float32

        self.position = position
        self.color = color

    def render_setup(self):

        self.model_pose = gfx.translation_matrix(
            gfx.vec3(self.position[0], self.position[1], 0.5*PYLON_HEIGHT))

        if self.gfx_object is None:
            
            self.gfx_object = gfx.IndexedPrimitives.cylinder(
                PYLON_RADIUS, PYLON_HEIGHT, 32, 1,
                self.color, None, self.model_pose)

        else:

            self.gfx_object.color = self.color
            self.gfx_object.model_pose = self.model_pose
        
    def render(self):

        self.render_setup()

        self.gfx_object.render()

######################################################################

class Box:

    def __init__(self, dims, position, angle, color):

        assert dims.shape == (3,) and dims.dtype == numpy.float32
        assert position.shape == (2,) and position.dtype == numpy.float32

        self.dims = dims
        self.position = position
        self.angle = angle
        self.color = color

        self.gfx_object = None
        
    def render_setup(self):

        self.model_pose = gfx.rigid_2d_matrix(self.position,
                                              self.angle,
                                              0.5*self.dims[2])


        if self.gfx_object is None:

            self.gfx_object = gfx.IndexedPrimitives.box(
                self.dims, self.color, None, self.model_pose)

            self.gfx_object.specular_exponent = 100.0
            self.gfx_object.specular_strength = 0.1

        else:

            self.gfx_object.model_pose = self.model_pose

    def render(self):

        self.render_setup()
        self.gfx_object.render()

        
######################################################################

def line_intersect(l1, l2):

    l3 = numpy.cross(l1, l2)
    return l3[:2] / l3[2]

######################################################################

class TapeStrip:

    def __init__(self, points):

        self.points = points

        self.gfx_object = None

    def render_setup(self):


        if self.gfx_object is None:

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

            self.gfx_object = gfx.IndexedPrimitives(vdata, gl.TRIANGLE_STRIP,
                                                    indices=None,
                                                    color=TAPE_COLOR,
                                                    texture=None, model_pose=None)

            self.gfx_object.specular_exponent = 100.0
            self.gfx_object.specular_strength = 0.05
            


    def render(self):

        self.render_setup()
        self.gfx_object.render()
        

######################################################################

def match_color(color, carray):

    if not isinstance(carray, numpy.ndarray):
        carray = numpy.array(carray)

    assert len(carray.shape) == 2 and carray.shape[1] == 3

    dists = numpy.linalg.norm(carray - color, axis=1)

    i = dists.argmin()

    return i, carray[i]

    
######################################################################

class Transformer:

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

class Environment:

    def __init__(self):

        self.dims = numpy.array([-1, -1], dtype=numpy.float32)

        self.floor_texture = None
        self.floor_gfx_obj = None
        
        self.pylons = []
        self.tape_strips = []
        self.boxes = []
        self.walls = []

    def render_setup(self):

        if self.floor_gfx_obj is None:
        
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

            self.floor_gfx_obj = gfx.IndexedPrimitives(
                vdata, mode, indices, 0.8*gfx.vec3(1, 1, 1),
                self.floor_texture)

            self.floor_gfx_obj.specular_strength = 0.25
        
    def render(self):

        self.render_setup()

        self.floor_gfx_obj.render()

        for obj in self.pylons + self.tape_strips + self.boxes + self.walls:
            obj.render()
            
    @classmethod
    def load_svg(cls, svgfile):

        env = Environment()

        svg = se.SVG.parse(svgfile, color='none')

        scl = 1e-2
        
        xform = Transformer(svg.viewbox.viewbox_width,
                            svg.viewbox.viewbox_height, scl)

        env.dims = xform.dims

        for item in svg:

            xx, yx, xy, yy, x0, y0 = [getattr(item.transform, letter) for letter in 'abcdef']

            det = xx*yy - yx*xy
            is_rigid = (abs(det - 1) < 1e-4)
            
            print('  transform:', xx, yx, xy, yy, x0, y0)
            print('  determinant:', det)
            print('  is_rigid:', is_rigid)

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
                    print('skipping arena')
                    continue

                dims = gfx.vec3(w, h, min(w, h))
                pctr = xform.transform(cx, cy)
                pfwd = xform.transform(cx+1, cy)
                delta = pfwd-pctr
                theta = numpy.arctan2(delta[1], delta[0])

                print('pctr is', pctr)

                env.boxes.append( Box(dims, pctr, theta, CARDBOARD_COLOR) )
                
                print('rect', item.x, item.y, item.width, item.height)

            elif isinstance(item, se.Circle):
                
                print('circle', item.cx, item.cy, item.rx)

                _, color = match_color(fcolor, PYLON_COLORS)

                position = xform.transform(item.cx, item.cy)
                
                env.pylons.append(Pylon(position, color))
                                        
            elif isinstance(item, se.SimpleLine):

                p0 = xform.transform(item.x1, item.y1)
                p1 = xform.transform(item.x2, item.y2)

                pctr = 0.5*(p0 + p1)


                delta = p1 - p0
                theta = numpy.arctan2(delta[1], delta[0])

                dims = gfx.vec3(numpy.linalg.norm(delta),
                                WALL_THICKNESS, WALL_HEIGHT)
                
                env.walls.append( Box(dims, pctr, theta, CARDBOARD_COLOR) )
                
                print('line', item.x1, item.y1, item.x2, item.y2)
                
            elif isinstance(item, se.Polyline):
                
                points = numpy.array(
                    [xform.transform(p.x, p.y) for p in item.points])
                
                print('polyline', points)

                env.tape_strips.append(TapeStrip(points))
                
            else:
                
                print('*** warning: ignoring SVG item:', item, '***')
                continue

        return env

######################################################################

class RoboSimApp(gfx.GlfwApp):

    def __init__(self, env):

        super().__init__()

        self.create_window('Robot simulator', 1200, 1080)

        gl.Enable(gl.CULL_FACE)
        gl.Enable(gl.FRAMEBUFFER_SRGB)

        self.env = env

        self.env.render_setup()
        
        self.perspective = None
        self.view = None

        self.xrot = 0
        self.yrot = 0

        self.mouse_pos = numpy.array(self.framebuffer_size/2, dtype=numpy.float32)

        self.handle_mouse_rot()

    def key(self, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(self.window, gl.TRUE)


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
        self.xrot = gfx.mix(numpy.pi/2, -numpy.pi/2, numpy.clip(foo[1], 0, 1))

        self.need_render = True
        self.view = None
            
    def framebuffer_resized(self):
        self.perspective = None

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

            w, h = self.env.dims
        
            self.view = gfx.look_at(eye=gfx.vec3(0.5*w, 0.5*h, 5.0),
                                    center=gfx.vec3(0.5*w, 0.5*h, 0),
                                    up=gfx.vec3(0, 1, 0),
                                    Rextra=R_mouse)

            view_pos = -numpy.dot(numpy.linalg.inv(self.view[:3, :3]),
                                  self.view[:3, 3])

            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['viewPos'], view_pos)
            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['view'], self.view)

        gl.Enable(gl.DEPTH_TEST)

        self.env.render()


        
        
        

######################################################################
            
def _test_load_environment():

    env = Environment.load_svg('environments/first_environment.svg')

    app = RoboSimApp(env)

    app.run()

        
if __name__ == '__main__':
    
    _test_load_environment()
