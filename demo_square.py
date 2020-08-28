import robosim
import robosim_controller as ctrl
import numpy

######################################################################

class SimpleSquareController(ctrl.Controller):

    def __init__(self):
        super().__init__()

    def initialize(self, time, odom_pose):
        self.set_state(time, 'straight', odom_pose)

    def set_state(self, time, state, odom_pose):
        print('set state to {} at time {:.2f}'.format(state, time))
        self.init_time = time
        self.state = state
        self.init_odom_pose = odom_pose.copy()

    def update(self, time, dt, robot_state, detections):
        
        elapsed = time - self.init_time

        # note rounding to nearest update period
        is_done = (elapsed >= 2.0 - 0.5*dt)

        if is_done:
            rel_pose = self.init_odom_pose.transform_inv(robot_state.odom_pose)
            print('done, rel_pose =', rel_pose)
            if self.state == 'straight':
                new_state = 'turn'
            else:
                new_state = 'straight'
            print('I can see:', detections)
            self.set_state(time, new_state, robot_state.odom_pose)

        if self.state == 'straight':

            return ctrl.ControllerOutput(
                forward_vel=0.5, angular_vel=0.0)

        else: # self.state == 'turn'

            return ctrl.ControllerOutput(
                forward_vel=0.0, angular_vel=numpy.pi/4)

######################################################################

if __name__ == '__main__':

    controller = SimpleSquareController()
    
    app = robosim.RoboSimApp(controller)

    app.sim.set_dims(5.0, 5.0)
    app.sim.add_pylon((1.5, 1.5), 'green')
    app.sim.add_pylon((3.5, 3.5), 'green')
    app.sim.add_pylon((1.5, 3.5), 'orange')
    app.sim.add_pylon((3.5, 1.5), 'orange')
    app.sim.add_ball((2.5, 2.5))
    app.sim.initialize_robot((2.0, 2.0), 0.0)

    app.run()

