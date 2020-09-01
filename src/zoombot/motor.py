import numpy
import datetime
import matplotlib.pyplot as plt
import scipy.integrate

RPS_PER_RPM = numpy.pi / 30
RPM_PER_RPS = 30 / numpy.pi

def linear_ode(x0, U, lambdas, Uinv, b, t):

    was_scalar = numpy.isscalar(t)
    
    if was_scalar:
        t = numpy.array([t])

    n, = x0.shape
    k, = t.shape
    
    assert lambdas.shape == (n,) and b.shape == (n,)
    assert U.shape == (n, n)
    assert Uinv.shape == U.shape

    # orig system is
    #
    #    x'(t) = A x(t) + b
    #    x'(t) = U D Uinv x(t) + b
    #
    # transform to system
    #
    #    Uinv x'(t) = D Uinv x(t) + Uinv b
    #    alpha'(t) = D alpha(t) + b
    #
    # transform again to system
    #
    #    q'(t) = D q(t) where q(t) = (alpha(t) + Dinv b)

    W = numpy.hstack((x0.reshape(-1, 1), b.reshape(-1, 1)))
    M = numpy.linalg.solve(U, W)

    alpha0 = numpy.dot(Uinv, x0)
    beta0 = numpy.dot(Uinv, b)

    q0 = alpha0 + beta0 / lambdas

    lambdas_t = t.reshape(k, 1) * lambdas.reshape(1, 2)
    q_of_t = q0 * numpy.exp(lambdas_t)

    alpha_of_t = q_of_t - beta0 / lambdas

    x_of_t = numpy.dot(alpha_of_t, U.T)

    # x0 should have shape (n,)
    # Uinv and U should have shape (n,n)
    # lambdas should have shape (n,)
    # b should have shape (n,)
    # t should have shape (k,)

    # returns (k,n) outputs of ODE
    
    if was_scalar:
        x_of_t = x_of_t.flatten()
    
    return x_of_t
    

class Motor:

    def steady_speed_for_torque(self, T, V=None):
        K = self.K
        R = self.R
        b = self.b
        if V is None:
            V = self.V_nominal
        return (K*V - R*T)/(K**2 + R*b)

    def steady_current_for_torque(self, T, V=None):
        K = self.K
        R = self.R
        b = self.b
        if V is None:
            V = self.V_nominal
        return (K*T + V*b)/(K**2 + R*b)

    def steady_torque_for_current(self, i, V=None):
        K = self.K
        R = self.R
        b = self.b
        if V is None:
            V = self.V_nominal
        return (i*(K**2 + R*b) - V*b)/K

    def steady_speed_for_current(self, i, V=None):
        K = self.K
        R = self.R
        if V is None:
            V = self.V_nominal
        return (-R*i + V)/K
    
    def stall_torque(self, V=None):
        if V is None:
            V = self.V_nominal
        return self.K*V/self.R

    def stall_current(self, V=None):
        if V is None:
            V = self.V_nominal
        return V/self.R

    def state_deriv(self, state, control):
        return numpy.dot(self.A, state) + numpy.dot(self.B, control)

    def simulate_dynamics(self, init_state, control, dt):
        b = numpy.dot(self.B, control)
        return linear_ode(init_state, self.U, self.lambdas, self.Uinv, b, dt)

    def electrical_time_constant(self):
        return self.L/self.R

    def mechanical_time_constant(self):
        return self.R*self.J/self.K**2

    def motor_torque_from_wheel_tgt_force(self, wheel_tgt_force):

        wheel_torque = wheel_tgt_force*self.wheel_radius
        motor_torque = wheel_torque / self.gear_ratio

        return motor_torque

    def wheel_tgt_force_from_motor_torque(self, motor_torque):

        wheel_torque = motor_torque * self.gear_ratio
        wheel_tgt_force = wheel_torque / self.wheel_radius

        return wheel_tgt_force
    
    def wheel_tgt_speed_from_motor_speed(self, motor_speed):

        wheel_angular_speed = motor_speed / self.gear_ratio
        wheel_tgt_speed = wheel_angular_speed * self.wheel_radius

        return wheel_tgt_speed

    def __init__(self):
        
        # inspirations for constants:
        # http://kobuki.yujinrobot.com/wiki/online-user-guide/
        # https://www.portescap.com/en/products/brush-dc-motors/16n78-athlonix-brush-dc-motor
        
        #J = 0.0002  # way too big to be physically plausible but if it gets any smaller motor basically stops instantly with no output
        J = 3e-5
        V = 12

        if 1:
            
            # kobuki
            noload_current = 0.210
            noload_speed = 9960*RPS_PER_RPM
            stall_current = 6.1

            rated_current = 0.750

            L = 0.0015
            
            K = -V*(noload_current - stall_current)/(noload_speed*stall_current)
            b = -V*noload_current*(noload_current - stall_current)/(noload_speed**2*stall_current)
            R = V/stall_current
        else:

            # portescap
            noload_current = 0.005
            noload_speed = 8380*RPS_PER_RPM
            stall_torque = 0.0124

            rated_current = 0.490

            L = 0.0005
            
            K = V*stall_torque/(V*noload_current + noload_speed*stall_torque)
            b = V*noload_current*stall_torque/(noload_speed*(V*noload_current + noload_speed*stall_torque))
            R = V**2/(V*noload_current + noload_speed*stall_torque)

        self.K = K
        self.R = R
        self.b = b
        self.L = L
        self.J = J

        self.V_nominal = V

        self.rated_current = rated_current

        self.gear_ratio = 6545/132 # about 50:1
        self.wheel_radius = 0.035  # m
        
        self.A = numpy.array([
            [ -b/J, K/J ],
            [ -K/L, -R/L ]
        ])

        self.B = numpy.array([
            [ 1/J, 0 ],
            [ 0, 1/L ]
        ])

        self.lambdas, self.U = numpy.linalg.eig(self.A)

        self.Uinv = numpy.linalg.inv(self.U)

######################################################################        

def _test_motor_basics():

    M = Motor()

    noload_speed = M.steady_speed_for_torque(0)
    noload_current = M.steady_current_for_torque(0)

    rated_current = M.rated_current
    rated_torque = M.steady_torque_for_current(rated_current)
    rated_speed = M.steady_speed_for_current(rated_current)

    rated_torque2 = (M.K**2*rated_current + M.b*(M.R*rated_current - M.V_nominal))/M.K
    rated_speed2 = (-M.R*rated_current + M.V_nominal)/M.K

    print('got motor K={}, R={}, b={}'.format(M.K, M.R, M.b))
    print('no-load speed:  ', noload_speed*RPM_PER_RPS, 'RPM')
    print('should be same: ', M.steady_speed_for_current(noload_current)*RPM_PER_RPS, 'RPM')
    print('no-load current:', noload_current, 'A')
    print('stall torque:   ', M.stall_torque(), 'Nm')
    print('stall current:  ', M.stall_current(), 'A')
    print('rated current:  ', M.rated_current, 'A')
    print('rated torque:   ', rated_torque, 'Nm')
    print('should be same: ', rated_torque2)
    print('rated speed:    ', rated_speed*RPM_PER_RPS, 'RPM')
    print('should be same: ', rated_speed2*RPM_PER_RPS)
    print('electrical t.c.:', M.electrical_time_constant(), 's')
    print('mechanical t.c.:', M.mechanical_time_constant(), 's')
    print('rated wheel tgt speed:', M.wheel_tgt_speed_from_motor_speed(rated_speed), 'm/s')
    print('rated wheel force:', M.wheel_tgt_force_from_motor_torque(rated_torque), 'N')

    for T in numpy.linspace(0, M.stall_torque(), 10):

        speed = M.steady_speed_for_torque(T)
        current = M.steady_current_for_torque(T)

        state = numpy.array([speed, current])

        control = numpy.array([-T, M.V_nominal])

        dstate = M.state_deriv(state, control)

        print('should be [0, 0]:', dstate)
                               
    dt = 0.01
    t = numpy.arange(2501)*dt

    init_state = numpy.array([noload_speed, noload_current])
    control = numpy.array([-rated_torque, M.V_nominal])

    def f(y, t):
        return numpy.dot(M.A, y) + numpy.dot(M.B, control)

    start = datetime.datetime.now()
    x_numeric = scipy.integrate.odeint(f, init_state, t).T
    print('got scipy solution in {} seconds'.format(
        (datetime.datetime.now()-start).total_seconds()))

    start = datetime.datetime.now()
    x_analytic = M.simulate_dynamics(init_state, control, t).T
    print('got my solution in {} seconds'.format(
        (datetime.datetime.now()-start).total_seconds()))

    plt.subplot(2, 1, 1)
    plt.plot(t, x_analytic[0], label='speed (mine)')
    plt.plot(t, x_numeric[0], label='speed (numpy)')
    plt.plot(0, 0, color='none')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, x_analytic[1], label='current (mine)')
    plt.plot(t, x_numeric[1], label='current (numpy)')
    plt.plot(0, 0, color='none')
    plt.legend()

    plt.show()

######################################################################    

def _test_motor_drive_mass():


    robot_mass = 2.5 # kg

    # robot body state is [position, velocity]
    robot_state = numpy.zeros(2)

    # motor state is [speed, current]
    motor_state = numpy.zeros(2)

    # motor control is [torque, voltage]
    # but really we only control voltage, torque is going to be
    # a function of velocity mismatch

    frequency = 100
    duration = 5.0

    dt = 1.0 / frequency

    nticks = int(duration * frequency) + 1

    all_vars = numpy.zeros((nticks, 7), dtype=float)

    all_vars[:, 0] = dt*numpy.arange(len(all_vars))
    all_vars[:, 1] = 0
    all_vars[0, 2:4] = motor_state
    all_vars[:, 4] = 0
    all_vars[0, 5:7] = robot_state

    # OK so 10, 20 settles to .25 m/s with no clipping in about 2-3 seconds
    # also starting from here and setting kp=c*10 and ki=sqrt(c)*10 roughly works
    #kp = 10.0
    #ki = 20.0

    kp = 200
    ki = 200

    vel_error_int_max = 10/ki
    
    motor_torque = 0.0

    desired_robot_vel = 0.5
    vel_error_int = 0.0

    motor = Motor()

    for idx in range(1, len(all_vars)):

        # get the wheel tangential speed of the motors
        wheel_tgt_speed = motor.wheel_tgt_speed_from_motor_speed(motor_state[0])
        
        # step 1) drive motors
        vel_error = (desired_robot_vel - wheel_tgt_speed)

        if idx < len(all_vars)//2:

            V_cmd = kp*vel_error + ki*vel_error_int

            vel_error_int += dt * vel_error

            vel_error_int = numpy.clip(vel_error_int,
                                       -vel_error_int_max,
                                       vel_error_int_max)

            V_cmd = numpy.clip(V_cmd, -motor.V_nominal, motor.V_nominal)

        else:

            V_cmd = 0
        
        motor_control = numpy.array([motor_torque, V_cmd])

        motor_state = motor.simulate_dynamics(motor_state, motor_control, dt)

        # step 2) tie to world physics

        # get the wheel tangential speed of the motors
        wheel_tgt_speed = motor.wheel_tgt_speed_from_motor_speed(motor_state[0])

        vel_mismatch = wheel_tgt_speed - robot_state[1]
        vel_impulse = vel_mismatch * robot_mass
        F = vel_impulse/dt

        wheel_torque = motor.motor_torque_from_wheel_tgt_force(-F)

        robot_accel = F / robot_mass

        vel_new = robot_state[1] + robot_accel*dt
        
        pos_new = robot_state[0] + 0.5*(vel_new + robot_state[1])*dt

        robot_state[:] = [pos_new, vel_new]

        all_vars[idx, 1] = V_cmd
        all_vars[idx, 2:4] = motor_state
        all_vars[idx, 4] = F
        all_vars[idx, 5:7] = robot_state

    avt = all_vars.T

    plt.subplot(5, 1, 1)
    plt.plot(avt[0], avt[2], label='motor speed')
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(avt[0], avt[3], label='motor current')
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(avt[0], avt[1], label='cmd voltage')
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(avt[0], avt[4], label='ground reaction force (min={:.2f}, max={:.2f})'.format(avt[4].min(), avt[4].max()))
    plt.legend()
    
    plt.subplot(5, 1, 5)
    plt.plot(avt[0], numpy.ones_like(avt[0])*desired_robot_vel, '--', color=[0.7, 0.7, 0.7])
    plt.plot(avt[0], avt[6], label='robot speed')
    plt.plot(avt[0], motor.wheel_tgt_speed_from_motor_speed(avt[2]), label='wheel tgt speed')
    plt.legend()

    
    plt.show()
    

if __name__ == '__main__':

    #_test_motor_basics()
    _test_motor_drive_mass()
    
