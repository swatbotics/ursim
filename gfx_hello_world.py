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

######################################################################

class HelloWorldApp(gfx.GlfwApp):

    def __init__(self):

        super().__init__()

        self.create_window('Hello World', 800, 600)

        gl.Enable(gl.CULL_FACE)
        gl.Enable(gl.FRAMEBUFFER_SRGB)

        self.framebuffer = gfx.Framebuffer(512, 512)

        self.texture = gfx.load_texture('textures/monalisa.jpg', 'RGB')
        
        self.fsquad = gfx.FullscreenQuad(self.texture)

        ballpos = gfx.translation_matrix(gfx.vec3(1.5, 0, 0))

        cylpos = gfx.translation_matrix(gfx.vec3(0, 1.5, 0))
        
        self.objects = [
            
            gfx.IndexedPrimitives.box(gfx.vec3(1, 1, 1),
                                      gfx.vec3(0.5, 0.75, 1.0),
                                      texture=None),
            
            gfx.IndexedPrimitives.sphere(0.5, 32, 24,
                                         gfx.vec3(1, 0, 0),
                                         model_pose=ballpos),
            
            gfx.IndexedPrimitives.cylinder(0.5, 1, 32, 1,
                                           gfx.vec3(1, 0, 1),
                                           model_pose=cylpos)
            
        ]

        self.box = self.objects[0]
        
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

        self.yrot = gfx.mix(-2*np.pi, 2*np.pi, foo[0])
        self.xrot = gfx.mix(np.pi/2, -np.pi/4, np.clip(foo[1], 0, 1))

        self.need_render = True
        self.view = None
                
    ############################################################

    def framebuffer_resized(self):
        self.perspective = None

    ############################################################
        
    def render(self):
            
        self.framebuffer.activate()

        self.box.texture = None

        self.render_step()

        self.framebuffer.deactivate()

        gfx.check_opengl_errors('after framebuffer')

        gl.Viewport(0, 0,
                   self.framebuffer_size[0],
                   self.framebuffer_size[1])

        self.box.texture = self.framebuffer.rgb_textures[0]
        
        self.render_step()

    def render_step(self):
        
        gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        gl.Disable(gl.DEPTH_TEST)
        self.fsquad.render()

        
        if self.perspective is None:
            
            w, h = self.framebuffer_size
            aspect = w / max(h, 1)

            self.perspective = gfx.perspective_matrix(45.0, aspect, 0.1, 10.0)

            gfx.IndexedPrimitives.set_perspective_matrix(self.perspective)


        if self.view is None:

            Rx = gfx.rotation_matrix(self.xrot, gfx.vec3(1, 0, 0))
            Ry = gfx.rotation_matrix(self.yrot, gfx.vec3(0, 1, 0))

            R_mouse = np.dot(Rx, Ry)
        
            self.view = gfx.look_at(eye=gfx.vec3(2, 2, 0),
                                    center=gfx.vec3(0, 0, 0),
                                    up=gfx.vec3(0, 0, 1),
                                    Rextra=R_mouse)

            gfx.IndexedPrimitives.set_view_matrix(self.view)

        gl.Enable(gl.DEPTH_TEST)

        for obj in self.objects:
            obj.render()

    def destroy(self):
        
        print('hello world app is going away!')

        gl.DeleteTextures(1, [self.texture])

        for obj in self.objects:
            obj.destroy(destroy_static=True)
        

def main():

    app = HelloWorldApp()
    app.run()


if __name__ == "__main__":
    main()

