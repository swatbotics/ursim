import numpy
import os
from collections import namedtuple
import datetime

LoggerGroup = namedtuple('LoggerGroup',
                         'names, array, idx0, idx1')
        
class Logger:

    NUMERIC_TYPES = set('bufic')
    INITIAL_LOG_SIZE = 512

    def __init__(self, dt=None):

        self.groups = []
        self.total_variables = 0

        self.dt = dt

        self.have_time = False

        self.current_filename = None
        self.current_log = None
        self.log_rows = 0

        if dt is not None:
            self.total_variables += 1

    def add_variables(self, names, array):

        assert isinstance(array, numpy.ndarray)
        assert array.dtype.kind in self.NUMERIC_TYPES
        assert len(array.shape) == 1
        assert len(names) == array.size

        nvars = len(names)
        
        idx0 = self.total_variables
        idx1 = idx0 + nvars
        
        self.groups.append(LoggerGroup(names, array, idx0, idx1))

        self.total_variables += nvars

        if self.current_log:
            
            print('WARNING: adding variables after logging started!')
            print('  this is inefficient :(')
            print('  logged values for {}'.format(names[0]))
            print('  and other variables in this group will be zero.')

            self.current_log.resize((self.current_log.shape[0], self.total_variables))

    def begin_log(self):

        assert self.current_log is None and self.log_rows == 0

        self.current_filename = datetime.datetime.now().strftime('log_%Y%m%d_%H%M%S.npz')
        
        self.current_log = numpy.zeros((self.INITIAL_LOG_SIZE, self.total_variables),
                                       dtype=numpy.float32)


        return self.current_filename
            
    def append_log_row(self):

        if self.current_log is None:
            self.begin_log()
        elif self.current_log.shape[0] == self.log_rows:
            self.current_log.resize((2*self.current_log.shape[0], self.total_variables))

        for g in self.groups:
            self.current_log[self.log_rows, g.idx0:g.idx1] = g.array

        self.log_rows += 1

    def write_log(self):

        assert self.current_log is not None

        log_names = []
        
        written_portion = self.current_log[:self.log_rows]

        if self.dt is not None:
            written_portion[:, 0] = numpy.arange(self.log_rows, dtype=numpy.float32)*self.dt
            log_names.append('time')
        
        for g in self.groups:
            log_names.extend(g.names)

        log_names = numpy.array(log_names, dtype='U')

        numpy.savez_compressed(self.current_filename,
                               dt=self.dt,
                               names=log_names,
                               data=written_portion)

        self.current_filename = None
        self.current_log = None
        self.log_rows = 0
        
def read_log(filename, raw=False):
    
    npzfile = numpy.load(filename)
    
    dt = npzfile['dt']
    names = npzfile['names']
    data = npzfile['data']

    if raw:
        return dt, names, data
    else:
        lookup = dict([(str(name), data[:,col]) for col, name in enumerate(names)])
        return dt, lookup
        
def _test_logging():

    import matplotlib.pyplot as plt

    g1_data = numpy.zeros(3, dtype=numpy.float32)
    g2_data = numpy.zeros(2, dtype=int)

    l = Logger(dt=0.04)

    l.add_variables(['robot.position.x', 'robot.position.y', 'robot.angle'], g1_data)
    l.add_variables(['foo', 'bar'], g2_data)

    filename = l.begin_log()

    for theta in numpy.linspace(0, 2*numpy.pi, 32):
        x = numpy.sin(theta)
        y = -numpy.cos(theta)
        g1_data[:] = x, y, theta
        g2_data[:] = numpy.random.randint(32, size=2)
        l.append_log_row()

    l.write_log()

    dt, ldata = read_log(filename)

    plt.figure()

    time = ldata['time']

    nplots = len(ldata) - 1

    cnt = 0

    for key, values in ldata.items():
        if key == 'time':
            continue
        cnt += 1
        plt.subplot(nplots, 1, cnt)
        plt.plot(time, values, label=key)
        plt.legend()


    plt.figure()
    plt.plot(ldata['robot.position.x'], ldata['robot.position.y'])
    plt.axis('equal')

    plt.show()
    
    

if __name__ == '__main__':
    _test_logging()
