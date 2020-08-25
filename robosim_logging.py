import numpy
import sys
import os
from collections import namedtuple
import datetime
import matplotlib.pyplot as plt
import re

LoggerGroup = namedtuple('LoggerGroup',
                         'names, array, idx0, idx1')

MAX_PLOT_ROWS = 6
MAX_PLOT_COLS = 1

MAX_PLOTS_PER_FIGURE = MAX_PLOT_ROWS*MAX_PLOT_COLS

        
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

        print('starting log', self.current_filename)

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

        print('wrote log', self.current_filename)
        
        self.current_filename = None
        self.current_log = None
        self.log_rows = 0

    def finish(self):
        if self.current_log is not None:
            self.write_log()
        
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

    
PLOT_MERGES = [
    re.compile(r'.*\.(cmd_)?vel(\.[xy])'),
    re.compile(r'.*\.(cmd_)?wheel_vel(\.[lr])'), 
    re.compile(r'.*\.(cmd_)?vel\..*'),
    re.compile(r'robot\.bump(\.)'),
    re.compile(r'blobfinder\.(.*\.)num_detections'),
    re.compile(r'blobfinder\.(.*\.)max_area')
]

COLORS = dict(blue=[0, 0, 0.8],
              green=[0, 0.5, 0],
              orange=[1, 0.5, 0],
              purple=[0.5, 0, 1])

def plot_log(ldata):

    assert 'time' in ldata.keys()

    time = ldata['time']

    plots = []
    plot_lookup = dict()

    poses = []

    for name, trace in ldata.items():

        if name == 'time':
            continue

        if name.endswith('.pos.x'):
            py = name.replace('.pos.x', '.pos.y')
            if py in ldata:
                poses.append(name[:-2])

        matched_name = name

        for expr in PLOT_MERGES:
            m = expr.match(name)
            if m is not None:
                matched_name = m.group(0)
                if m.lastindex is not None:
                    for idx in range(m.lastindex, 0, -1):
                        span = m.span(idx)
                        if span is not None:
                            matched_name = (matched_name[:span[0]] +
                                            matched_name[span[1]:])

        if matched_name in plot_lookup:
            plot_idx = plot_lookup[matched_name]
        else:
            plot_idx = len(plots)
            plots.append([])
            plot_lookup[matched_name] = plot_idx

        plots[plot_idx].append((name, trace))


    num_plots = len(plots)
    cur_idx = 0

    total_figures = int(numpy.ceil(len(plots) / MAX_PLOTS_PER_FIGURE))
    if len(poses):
        total_figures += 1

    fig = plt.figure()
    fig_idx = 1
    fig.canvas.set_window_title('Figure 1/{}'.format(total_figures))
    
    for pidx, plist in enumerate(plots):
        
        if cur_idx >= MAX_PLOTS_PER_FIGURE:
            fig_idx += 1
            fig = plt.figure()
            fig.canvas.set_window_title('Figure {}/{}'.format(
                fig_idx, total_figures))
            cur_idx -= MAX_PLOTS_PER_FIGURE
            
        cur_idx += 1
        plt.subplot(MAX_PLOT_ROWS, MAX_PLOT_COLS, cur_idx)

        plist.sort()

        for name, trace in plist:
            kwargs = dict()
            for color, cvalue in COLORS.items():
                if color in name:
                    kwargs['color'] = cvalue
            plt.plot(time, trace, label=name, **kwargs)

        last_in_row = ((cur_idx % MAX_PLOT_ROWS) == 0 or
                        pidx + 1 == len(plots))

        if not last_in_row:
            plt.xticks([])

        plt.legend(loc='upper right', fontsize='xx-small')

    if len(poses):

        fig = plt.figure()
        fig_idx += 1
        
        fig.canvas.set_window_title('Figure {}/{}'.format(
            fig_idx, total_figures))
        
        for pname in poses:
            
            x = ldata[pname + '.x']
            y = ldata[pname + '.y']

            plt.plot(x, y, label=pname)
            
            tname = pname + '.angle'
            if tname in ldata:
                theta = ldata[tname]
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                plt.quiver(x[::8], y[::8], c[::8], s[::8],
                           label=tname)

        plt.legend(loc='upper right', fontsize='xx-small')
        plt.axis('equal')
            
    plt.show()
        
def _test_logging():

    import matplotlib.pyplot as plt

    g1_data = numpy.zeros(3, dtype=numpy.float32)
    g2_data = numpy.zeros(2, dtype=int)

    l = Logger(dt=0.04)

    l.add_variables(['robot.pos.x', 'robot.pos.y', 'robot.pos.angle'], g1_data)
    l.add_variables(['foo.vel.x', 'foo.vel.y'], g2_data)

    filename = l.begin_log()

    for theta in numpy.linspace(0, 2*numpy.pi, 32):
        x = numpy.sin(theta)
        y = -numpy.cos(theta)
        g1_data[:] = x, y, theta
        g2_data[:] = numpy.random.randint(32, size=2)
        l.append_log_row()

    l.write_log()

    dt, ldata = read_log(filename)

    os.unlink(filename)

    plot_log(ldata)

if __name__ == '__main__':

    if len(sys.argv) == 1:
        _test_logging()
    else:
        _, ldata = read_log(sys.argv[1])
        plot_log(ldata)
        
        
