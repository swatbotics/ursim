import robosim
import robosim_controller as ctrl
import numpy

######################################################################

class BadSquareController(ctrl.Controller):

    def __init__(self):
        super().__init__()

    def initialize(self, time, odom_pose):
        self.set_state(time, 'straight')

    def set_state(self, time, state):
        print('set state to {} at time {:.2f}'.format(state, time))
        self.init_time = time
        self.state = state

    def update(self, time, dt, bump, detections, odom_pose):

        elapsed = time - self.init_time

        if self.state == 'straight':

            if elapsed >= 2.0 - 0.5*dt:
                self.set_state(time, 'turn')

            return ctrl.ControllerOutput(
                vel_forward=0.5, vel_angle=0.0)

        else: # self.state == 'turn'

            if elapsed >= 2.0 - 0.5*dt:
                self.set_state(time, 'straight')

            return ctrl.ControllerOutput(
                vel_forward=0.0, vel_angle=numpy.pi/4)

######################################################################

if __name__ == '__main__':

    controller = BadSquareController()
    
    app = robosim.RoboSimApp(controller)

    app.sim.set_dims(5.0, 5.0)
    app.sim.initialize_robot((2.0, 2.0), 0.0)

    app.run()

