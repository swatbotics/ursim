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
import robosim_controller as ctrl
import robosim_core as core
import robosim_camera as scam
import glfw
import time
from CleanGL import gl

# DONE: teardown graphics
# DONE: teardown sim
# DONE: reset sim and env
# DONE: sim robot
# DONE: virtual bump sensors
# DONE: implement renderbuffers in graphics
# DONE: robot camera
# DONE: object detection
# DONE: clean up code to split into separate files with cleaner dependencies
# DONE: logging
# DONE: deal with slower-than-realtime
# DONE: implement 2D rigid xform class
# DONE: odometry (Euler integrator + Gaussian noise)
# DONE: log odometry pose relative to initial position
# DONE: put transform2d in its own python file with documentation
# DONE: controller API in its own python file with documentation
# TODO: laser scan interface
# TODO: nicer GUI/camera interface?
# TODO: more sophisticated frame rate control?

LOG_PROFILING_DELTA = 0
LOG_PROFILING_PHYSICS = 1
LOG_PROFILING_CAMERA = 2
LOG_PROFILING_RENDERCALL = 3
LOG_PROFILING_COUNT = 4

LOG_PROFILING_NAMES = [
    'profiling.overall',
    'profiling.physics',
    'profiling.camera',
    'profiling.rendercalls'
]

######################################################################

class SimRenderable:

    factory_lookup = None

    def __init__(self, sim_object):
        
        self.sim_object = sim_object
        self.gfx_objects = []

    def render(self):

        for obj in self.gfx_objects:
            if (self.sim_object.body is not None and
                hasattr(obj, 'model_pose')):
                obj.model_pose = core.b2xform(self.sim_object.body.transform)
            obj.render()

    def destroy(self):

        for obj in self.gfx_objects:
            obj.destroy()

        self.gfx_objects = []

    @classmethod
    def create_for_object(cls, sim_object):

        if cls.factory_lookup is None:
            cls.factory_lookup = {
                core.Pylon: PylonRenderable,
                core.Ball: BallRenderable,
                core.Wall: WallRenderable,
                core.Box: BoxRenderable,
                core.Room: RoomRenderable,
                core.TapeStrips: TapeStripsRenderable,
                core.Robot: RobotRenderable
            }

        subclass = cls.factory_lookup[type(sim_object)]
        return subclass(sim_object)

######################################################################

class PylonRenderable(SimRenderable):

    static_gfx_object = None

    def __init__(self, sim_object):
        
        super().__init__(sim_object)

        if self.static_gfx_object is None:
            self.static_gfx_object = gfx.IndexedPrimitives.cylinder(
                core.PYLON_RADIUS, core.PYLON_HEIGHT, 32, 1,
                self.sim_object.color,
                pre_transform=gfx.tz(0.5*core.PYLON_HEIGHT))
            
        self.gfx_objects = [self.static_gfx_object]
        
    def render(self):
        self.static_gfx_object.color = self.sim_object.color
        self.static_gfx_object.material_id = self.sim_object.material_id
        super().render()

    def destroy(self):
        self.gfx_objects = []
        
######################################################################

class BallRenderable(SimRenderable):

    static_gfx_object = None

    def __init__(self, sim_object):
        
        super().__init__(sim_object)
    
        if self.static_gfx_object is None:
        
            self.static_gfx_object = gfx.IndexedPrimitives.sphere(
                core.BALL_RADIUS, 32, 24, 
                core.BALL_COLOR,
                pre_transform=gfx.tz(core.BALL_RADIUS),
                specular_exponent=60.0,
                specular_strength=0.125,
                material_id=int(1 << 3))
        
        self.gfx_objects = [ self.static_gfx_object ]

    def destroy(self):
        self.gfx_objects = []
        
######################################################################

class WallRenderable(SimRenderable):


    def __init__(self, sim_object):

        super().__init__(sim_object)

        gfx_object = gfx.IndexedPrimitives.box(
            sim_object.dims, core.CARDBOARD_COLOR,
            pre_transform=gfx.tz(core.WALL_Z + 0.5*self.sim_object.dims[2]))

        self.gfx_objects = [gfx_object]

        for x in [-self.sim_object.bx, self.sim_object.bx]:

            block = gfx.IndexedPrimitives.box(
                gfx.vec3(core.BLOCK_SZ, core.BLOCK_SZ, core.BLOCK_SZ),
                core.BLOCK_COLOR,
                pre_transform=gfx.translation_matrix(
                    gfx.vec3(x, 0, 0.5*core.BLOCK_SZ)))

            self.gfx_objects.append(block)

######################################################################

class BoxRenderable(SimRenderable):


    def __init__(self, sim_object):

        super().__init__(sim_object)
        
        gfx_object = gfx.IndexedPrimitives.box(
            sim_object.dims, core.CARDBOARD_COLOR,
            pre_transform=gfx.tz(0.5*sim_object.dims[2]))

        self.gfx_objects = [gfx_object]

######################################################################

class RoomRenderable(SimRenderable):

    floor_texture = None

    def __init__(self, sim_object):

        super().__init__(sim_object)

        if self.floor_texture is None:
            self.floor_texture = gfx.load_texture('textures/floor_texture.png')
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)

        w, h = sim_object.dims

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

        z = core.ROOM_HEIGHT

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
            verts, indices, core.ROOM_COLOR)

        room_obj.specular_strength = 0.25

        self.gfx_objects = [ floor_obj, room_obj ]
        
######################################################################

class TapeStripsRenderable(SimRenderable):

    def __init__(self, sim_object):

        super().__init__(sim_object)

        r = core.TAPE_RADIUS
        offset = gfx.vec3(0, 0, r)

        self.gfx_objects = []

        dashes = []

        for points in sim_object.point_lists:

            points = points.copy() # don't modify original points

            deltas = points[1:] - points[:-1]
            segment_lengths = numpy.linalg.norm(deltas, axis=1)
            tangents = deltas / segment_lengths.reshape(-1, 1)

            is_loop = (numpy.linalg.norm(points[-1] - points[0]) < core.TAPE_RADIUS)

            if not is_loop:

                segment_lengths[0] += core.TAPE_RADIUS
                points[0] -= core.TAPE_RADIUS * tangents[0]
                deltas[0] = points[1] - points[0]

                segment_lengths[-1] += core.TAPE_RADIUS
                points[-1] += core.TAPE_RADIUS * tangents[-1]
                deltas[-1] = points[-1] - points[-2]

                desired_parity = 1

            else:

                desired_parity = 0

            total_length = segment_lengths.sum()

            num_dashes = int(numpy.ceil(total_length / core.TAPE_DASH_SIZE))
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
        
        vdata[:, 2] = core.TAPE_POLYGON_OFFSET
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
                                           color=core.TAPE_COLOR,
                                           specular_exponent = 100.0,
                                           specular_strength = 0.05,
                                           material_id=int(1 << 0))

        self.gfx_objects.append(gfx_object)

######################################################################

class RobotRenderable(SimRenderable):

    def __init__(self, sim_object):
        
        super().__init__(sim_object)

        self.gfx_objects.append(
            gfx.IndexedPrimitives.cylinder(
                core.ROBOT_BASE_RADIUS, core.ROBOT_BASE_HEIGHT, 64, 1,
                core.ROBOT_BASE_COLOR,
                pre_transform=gfx.tz(0.5*core.ROBOT_BASE_HEIGHT + core.ROBOT_BASE_Z),
                specular_exponent=40.0,
                specular_strength=0.75
            )
        )

        tx = -0.5*core.ROBOT_CAMERA_DIMS[0]
        
        self.gfx_objects.append(
            gfx.IndexedPrimitives.box(
                core.ROBOT_CAMERA_DIMS,
                core.ROBOT_BASE_COLOR,
                pre_transform=gfx.translation_matrix(
                    gfx.vec3(tx, 0, 0.5*core.ROBOT_CAMERA_DIMS[2] + core.ROBOT_CAMERA_Z)),
                specular_exponent=40.0,
                specular_strength=0.75
            )
        )

        btop = core.ROBOT_BASE_Z + core.ROBOT_BASE_HEIGHT
        cbottom = core.ROBOT_CAMERA_Z

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

######################################################################

class KeyboardController(ctrl.Controller):

    def __init__(self, app=None):
        super().__init__()
        self.app = app

    def update(self, time, dt, robot_state, detections):
        
        la = numpy.zeros(2)

        app = self.app

        if app.key_is_down(glfw.KEY_I):
            la += (0.5, 0)
            
        if app.key_is_down(glfw.KEY_K):
            la += (-0.5, 0)
            
        if app.key_is_down(glfw.KEY_J):
            la += (0, 2.0)
            
        if app.key_is_down(glfw.KEY_L):
            la += (0, -2.0)
            
        if app.key_is_down(glfw.KEY_U):
            la += (0.5, 1.0)
            
        if app.key_is_down(glfw.KEY_O): 
            la += (0.5, -1.0)

        if numpy.any(la):
            return ctrl.ControllerOutput(
                forward_vel=la[0],
                angular_vel=la[1])
        else:
            return None
            
######################################################################

class RoboSimApp(gfx.GlfwApp):

    def __init__(self, controller=None, filter_setpoints=False):

        super().__init__()

        self.create_window('Robot simulator', 640, 480, units='window')

        gfx.IndexedPrimitives.DEFAULT_SPECULAR_EXPONENT = 100.0
        gfx.IndexedPrimitives.DEFAULT_SPECULAR_STRENGTH = 0.1

        gl.Enable(gl.FRAMEBUFFER_SRGB)
        gl.Enable(gl.DEPTH_TEST)
        gl.Enable(gl.CULL_FACE)

        self.sim = core.RoboSim()

        self.perspective = None
        self.view = None

        self.xrot = 0
        self.yrot = 0

        self.mouse_pos = numpy.array(self.framebuffer_size/2, dtype=numpy.float32)

        self.handle_mouse_rot()

        self.animating = True
        self.was_animating = False

        self.renderables = []

        self.sim_camera = scam.SimCamera(self.sim.robot,
                                         self.renderables,
                                         self.sim.logger)

        assert self.sim.dt == 0.01
        assert self.sim.physics_ticks_per_update == 4

        self.frame_budget = self.sim.dt * self.sim.physics_ticks_per_update

        self.last_update_time = None

        self.log_time = numpy.zeros(LOG_PROFILING_COUNT, dtype=numpy.float32)
        self.sim.logger.add_variables(LOG_PROFILING_NAMES, self.log_time)

        self.last_sim_modification = self.sim.modification_counter - 1
        self.controller_initialized = False

        if controller is None:
            controller = KeyboardController(self)
            filter_setpoints = True

        self.controller = controller
        self.sim.robot.filter_setpoints = filter_setpoints

    def update_sim(self):

        start = glfw.get_time()
        self.sim_camera.update()
        self.log_time[LOG_PROFILING_CAMERA] = (glfw.get_time() - start)/self.frame_budget

        if not self.controller_initialized:
            
            self.controller.initialize(self.sim.sim_time,
                                       self.sim.robot.odom_pose.copy())
            
            self.controller_initialized = True

        robot = self.sim.robot

        robot_state = ctrl.RobotState(
            robot.odom_pose.copy(),
            bool(robot.bump[0]),
            bool(robot.bump[1]),
            bool(robot.bump[2]),
            robot.odom_wheel_vel_filtered[0],
            robot.odom_wheel_vel_filtered[1],
            robot.odom_linear_angular_vel_filtered[0],
            robot.odom_linear_angular_vel_filtered[1])
            
        result = self.controller.update(self.sim.sim_time,
                                        self.frame_budget,
                                        robot_state,
                                        self.sim_camera.detections)

        if result is None:
            self.sim.robot.motors_enabled = False
            self.sim.robot.desired_linear_angular_vel[:] = 0
        else:
            self.sim.robot.motors_enabled = True
            self.sim.robot.desired_linear_angular_vel[:] = (
                result.forward_vel, result.angular_vel)

        start = glfw.get_time()
        self.sim.update()
        self.log_time[LOG_PROFILING_PHYSICS] = (glfw.get_time() - start)/self.frame_budget

    def set_animating(self, a):

        if not a:
            self.was_animating = False
            self.prev_update = None

        self.animating = a

    def destroy(self):
        self.sim.logger.finish()

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

            self.update_sim()
            self.need_render = True

        elif key == glfw.KEY_R:

            self.sim.reset(reload_svg=True)
            self.last_update_time = None
            self.controller_initialized = False

        elif key == glfw.KEY_M:

            self.sim.robot.motors_enabled = not self.sim.robot.motors_enabled

        elif key == glfw.KEY_C:

            self.sim_camera.save_images()

        elif key == glfw.KEY_B:
            self.sim.kick_ball()

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
        
    def update(self):

        if self.last_sim_modification != self.sim.modification_counter:

            print('**** CREATING GRAPHICS OBJECTS ****')
            
            for r in self.renderables:
                r.destroy()

            self.renderables[:] = []
            
            for o in self.sim.objects:
                r = SimRenderable.create_for_object(o)
                self.renderables.append(r)

            self.last_sim_modification = self.sim.modification_counter
            self.need_render = True

            self.sim_camera.update()
        
        if self.animating:
            now = glfw.get_time()
            if self.was_animating:
                delta_t = now - self.prev_update
                self.log_time[LOG_PROFILING_DELTA] = delta_t/self.frame_budget
                if delta_t < self.frame_budget:
                    extra = self.frame_budget - delta_t
                    time.sleep(extra)
            self.prev_update = now
            self.was_animating = True
            self.update_sim()
        
    def render(self):

        start = glfw.get_time()

        gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
        gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        gl.Viewport(0, 0,
                   self.framebuffer_size[0],
                   self.framebuffer_size[1])
        
        if self.perspective is None:
            
            w, h = self.framebuffer_size
            aspect = w / max(h, 1)

            self.perspective = gfx.perspective_matrix(
                45, aspect, 0.15, 50.0)

            gfx.set_uniform(gfx.IndexedPrimitives.uniforms['perspective'],
                            self.perspective)

        if self.view is None:

            Rx = gfx.rotation_matrix(self.xrot, gfx.vec3(1, 0, 0))
            Ry = gfx.rotation_matrix(self.yrot, gfx.vec3(0, 1, 0))

            R_mouse = numpy.dot(Rx, Ry)

            w, h = self.sim.dims
            m = max(numpy.linalg.norm([w, h]), 8.0)

            self.view = gfx.look_at(
                eye=gfx.vec3(0.5*w, 0.5*h - 0.5*m, 0.25*core.ROOM_HEIGHT),
                center=gfx.vec3(0.5*w, 0.5*h, 0.25*core.ROOM_HEIGHT),
                up=gfx.vec3(0, 0, 1),
                Rextra=R_mouse)

        gfx.IndexedPrimitives.set_perspective_matrix(self.perspective)
        gfx.IndexedPrimitives.set_view_matrix(self.view)

        for r in self.renderables:
            r.render()

        w = min(self.framebuffer_size[0], self.sim_camera.framebuffer.width)
        h = min(self.framebuffer_size[1], self.sim_camera.framebuffer.height)

        gl.BindFramebuffer(gl.READ_FRAMEBUFFER, self.sim_camera.framebuffer.fbo)

        gl.BindFramebuffer(gl.DRAW_FRAMEBUFFER, 0)

        gl.BlitFramebuffer(0, 0, w, h,
                           0, self.framebuffer_size[1]-h//2,
                           w//2, self.framebuffer_size[1],
                           gl.COLOR_BUFFER_BIT, gl.NEAREST)

        gl.BindFramebuffer(gl.FRAMEBUFFER, 0)

        self.log_time[LOG_PROFILING_RENDERCALL] = (glfw.get_time() - start)/self.frame_budget
        
######################################################################

def keyboard_demo():
    
    app = RoboSimApp()

    app.sim.load_svg('environments/first_environment.svg')

    app.run()

if __name__ == '__main__':
    
    keyboard_demo()
