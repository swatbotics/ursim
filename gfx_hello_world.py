######################################################################
#
# gfx_hello_world.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import numpy as np

import glfw
from CleanGL import gl
import graphics as gfx
import transformations as tf

######################################################################

class HelloWorldApp(gfx.GlfwApp):

    def __init__(self):

        super().__init__()

        self.create_window('Hello World', 800, 600)

        gl.Enable(gl.CULL_FACE)
        gl.Enable(gl.FRAMEBUFFER_SRGB)

        texture = gfx.load_texture('textures/monalisa.jpg', 'RGB')
        
        self.fsquad = gfx.FullscreenQuad(texture)

        ballpos = np.eye(4, dtype=np.float32)
        ballpos[:3, 3] = gfx.vec3(1.5, 0, 0)

        cylpos = np.eye(4, dtype=np.float32)
        cylpos[:3, 3] = gfx.vec3(0, 1.5, 0)
        
        self.objects = [
            
            gfx.IndexedPrimitives.box(gfx.vec3(1, 1, 1),
                                      gfx.vec3(0.5, 0.75, 1.0),
                                      None),
            
            gfx.IndexedPrimitives.sphere(0.5, 32, 24,
                                         gfx.vec3(1, 0, 0),
                                         None, ballpos),
            
            gfx.IndexedPrimitives.cylinder(0.5, 1, 32, 1,
                                           gfx.vec3(1, 0, 1),
                                           None,
                                           cylpos)
            
        ]
        
        self.mouse_pos = np.array(self.framebuffer_size/2, dtype=np.float32)

        self.handle_mouse_rot()

    ############################################################
        
    def key(self, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(self.window, gl.TRUE)
            
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

        self.yrot = gfx.mix(-2*np.pi, 2*np.pi, foo[0]) - np.pi*3/4
        self.xrot = gfx.mix(np.pi/2, -np.pi/4, np.clip(foo[1], 0, 1))

        self.need_render = True
        self.view = None
                
    ############################################################

    def framebuffer_resized(self):
        self.perspective = None

    ############################################################
        
    def render(self):
            
        gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
        gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        gl.Viewport(0, 0,
                   self.framebuffer_size[0],
                   self.framebuffer_size[1])

        gl.Disable(gl.DEPTH_TEST)
        self.fsquad.render()

        gl.UseProgram(gfx.IndexedPrimitives.program)
        
        if self.perspective is None:
            
            w, h = self.framebuffer_size
            aspect = w / max(h, 1)

            self.perspective = gfx.perspective_matrix(45.0, aspect, 0.1, 10.0)

            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['perspective'],
                            self.perspective)

        if self.view is None:

            Rx = tf.rotation_matrix(self.xrot, gfx.vec3(1, 0, 0))
            Ry = tf.rotation_matrix(self.yrot, gfx.vec3(0, 1, 0))

            R_mouse = np.dot(Rx, Ry).astype(np.float32)
        
            self.view = gfx.look_at(eye=gfx.vec3(0, -3, 0),
                                    center=gfx.vec3(0, 0, 0),
                                    up=gfx.vec3(0, 0, 1),
                                    Rextra=R_mouse)

            view_pos = -np.dot(np.linalg.inv(self.view[:3, :3]), self.view[:3, 3])

            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['viewPos'], view_pos)
            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['view'], self.view)

        gl.Enable(gl.DEPTH_TEST)

        for obj in self.objects:
            obj.render()


def main():

    app = HelloWorldApp()
    app.run()

if __name__ == "__main__":
    main()

