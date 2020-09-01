######################################################################
#
# zoombot/plotter.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################
#
# Plot log file data.
#
######################################################################

import sys, os, re
from collections import Counter

import numpy
import matplotlib
from matplotlib import backend_bases
import matplotlib.pyplot as plt

from .datalog import read_log

######################################################################

MAX_PLOT_ROWS = 6
MAX_PLOT_COLS = 1

MAX_PLOTS_PER_FIGURE = MAX_PLOT_ROWS*MAX_PLOT_COLS

PLOT_MERGES = [
    re.compile(r'.*\.(cmd\.)?vel(\.[xy])'),
    re.compile(r'(.*\.)(cmd\.)?wheel_vel(\.raw|\.filtered)?\.l'),
    re.compile(r'(.*\.)(cmd\.)?wheel_vel(\.raw|\.filtered)?\.r'),
    re.compile(r'(robot|odom)\.pos.x'),
    re.compile(r'(robot|odom)\.pos.y'),
    re.compile(r'(robot|odom)\.pos.angle'),
    re.compile(r'(robot|odom)\.(cmd\.)?vel(\.raw|\.filtered)?\.forward'),
    re.compile(r'(robot|odom)\.(cmd\.)?vel(\.raw|\.filtered)?\.angle'),
    re.compile(r'motor\.vel\.(.*)'),
    re.compile(r'motor\.torque\.(.*)'),
    re.compile(r'.*\.(cmd\.)?vel\..*'),
    re.compile(r'robot\.(bump.*|motors_enabled)'),
    re.compile(r'blobfinder\.(.*\.)num_detections'),
    re.compile(r'blobfinder\.(.*\.)max_area'),
    re.compile(r'robot\.wheel_force\.(.*)'),
    re.compile(r'motor\.voltage\.(.*)'),
    #re.compile(r'motor\.(inferred_)?current\.(.*)'),
    re.compile(r'motor\..*current\.(.*)'),
    re.compile(r'profiling\.camera\.([^.]+)'),
    re.compile(r'profiling'),
]

DEFAULT_EXCLUDES = [
    'profiling',
]

COLORS = dict(blue=[0, 0, 0.8],
              green=[0, 0.5, 0],
              orange=[1, 0.5, 0],
              purple=[0.5, 0, 1])

######################################################################

def keyboard(event):
    if event.key == 't':
        fig = event.canvas.figure
        for ax in fig.axes:
            for child in ax.get_children():
                if isinstance(child, matplotlib.legend.Legend):
                    is_visible = child.get_visible()
                    child.set_visible(not is_visible)
        event.canvas.draw()

######################################################################

class PlotManager:

    def __init__(self, fig, axis_list):

        self.fig = fig
        self.axis_list = axis_list

        self.axis_line_name_lookup = dict()
        
    def mouse_press(self, event):
        print('mouse press at {}'.format((event.x, event.y)))

    def mouse_release(self, event):
        print('mouse release at {}'.format((event.x, event.y)))

    def mouse_motion(self, event):
        #print('mouse is at {}'.format((event.x, event.y)))
        pass

    def mouse_enter_axes(self, event):
        print('entered')
        event.inaxes.format_coord = lambda x, y: self.format_coord(event.inaxes, x, y)

    def format_coord(self, ax, x, y):

        outputs = ['t={:.2g}'.format(x)]

        for line in ax.lines:
            lx, ly = line.get_data()
            py = numpy.interp(x, lx, ly)
            name = self.axis_line_name_lookup[(ax, line)]
            outputs.append('{}={:.3g}'.format(name, py))
        
        return ' '.join(outputs)

    def setup(self):

        self.fig.canvas.mpl_connect('button_press_event', self.mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_motion)
        self.fig.canvas.mpl_connect('axes_enter_event', self.mouse_enter_axes)

        for idx, ax in enumerate(self.axis_list):
            
            names = [line.get_label() for line in ax.lines]

            all_atoms = set()

            name_atoms = [ name.split('.') for name in names ]
            
            for atoms in name_atoms:
                all_atoms = all_atoms | set(atoms)

            redundant_atoms = set()

            for atom in all_atoms:
                if all([atom in atoms for atoms in name_atoms]):
                    redundant_atoms.add(atom)
                    
            names = []

            for atoms in name_atoms:
                atoms_filtered = []
                for atom in atoms:
                    if atom not in redundant_atoms:
                        atoms_filtered.append(atom)
                names.append('.'.join(atoms_filtered))
                
            for name, line in zip(names, ax.lines):
                self.axis_line_name_lookup[(ax, line)] = name

######################################################################

def make_figure(filename, fig_idx, total_figures, subplots):  
    fig = plt.figure(fig_idx)
    fig.canvas.mpl_connect('key_press_event', keyboard)
    fig.canvas.set_window_title('{} - figure {}/{}'.format(filename, fig_idx, total_figures))
    fig.suptitle("Press 't' to toggle legend, 'q' to close")
    if subplots:
        axis_list = fig.subplots(MAX_PLOT_ROWS, MAX_PLOT_COLS, sharex=True)
        mgr = PlotManager(fig, axis_list)
        return fig, axis_list, mgr
    else:
        return fig

######################################################################
    
def plot_log(fname, ldata, trace_names=[]):

    assert 'time' in ldata.keys()

    time = ldata['time']

    plots = []
    plot_lookup = dict()

    poses = []

    for name, trace in ldata.items():

        if name == 'time':
            continue

        matches_trace = any([frag in name for frag in trace_names])
        matches_exclude = any([frag in name for frag in DEFAULT_EXCLUDES])

        if len(trace_names) and not matches_trace:
            continue

        if matches_exclude and not matches_trace:
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
                break

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

    managers = []

    fig_idx = 1
    fig, subplots, mgr = make_figure(fname, fig_idx, total_figures, True)
    managers.append(mgr)

    for pidx, plist in enumerate(plots):

        if cur_idx >= MAX_PLOTS_PER_FIGURE:
            mgr.setup()
            fig_idx += 1
            fig, subplots, mgr = make_figure(fname, fig_idx, total_figures, True)
            managers.append(mgr)
            cur_idx -= MAX_PLOTS_PER_FIGURE
            
        ax = subplots[cur_idx]
        cur_idx += 1

        plist.sort()

        for name, trace in plist:
            kwargs = dict()
            for color, cvalue in COLORS.items():
                if color in name:
                    kwargs['color'] = cvalue
            if 'angle' in name:
                trace = trace * 180 / numpy.pi
            line, = ax.plot(time, trace, label=name, **kwargs)

        last_in_row = ((cur_idx % MAX_PLOT_ROWS) == 0 or
                        pidx + 1 == len(plots))

        ax.legend(loc='upper right', fontsize='xx-small')

    mgr.setup()
    #sys.exit(0)

    if len(poses):

        fig_idx += 1
        fig = make_figure(fname, fig_idx, total_figures, False)

        for idx, pname in enumerate(poses):
            
            x = ldata[pname + '.x']
            y = ldata[pname + '.y']

            handle, = plt.plot(x, y, label=pname, zorder=2)
            color = numpy.array(matplotlib.colors.to_rgb(handle.get_color()))
            
            tname = pname + '.angle'
            if tname in ldata:
                theta = ldata[tname]
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                plt.quiver(x[::8], y[::8], c[::8], s[::8],
                           label=tname, color=(0.7*color + 0.3),
                           units='dots', width=3.0*handle.get_linewidth(),
                           zorder=1)

        plt.legend(loc='upper right', fontsize='xx-small')
        plt.axis('equal')
            
    plt.show()

######################################################################

def get_latest():
    expr = re.compile('log_[0-9]+_[0-9]+.npz')
    files = os.listdir()
    files = [f for f in files if expr.match(f)]
    files.sort()
    if not len(files):
        print('no log files found!')
        sys.exit(1)
    latest = files[-1]
    print('opening latest log file', latest)
    return latest

######################################################################

def main():

    #matplotlib.rcParams['toolbar'] = 'None'
    #toolbar = plt.get_current_fig_manager().toolbar
    #print(toolbar.__dict__)
    #for x in toolbar.actions():
    #    print(x.text())
    #    #if x.text() in unwanted_buttons:
    #    #    toolbar.removeAction(x)

    if len(sys.argv) == 1:
        filename = get_latest()
        extra_args = []
    else:
        if sys.argv[1] == '--latest' or sys.argv[1] == '-l':
            filename = get_latest()
        else:
            filename = sys.argv[1]
        extra_args = sys.argv[2:]

    _, ldata = read_log(filename)
    plot_log(os.path.basename(filename), ldata, extra_args)

######################################################################

if __name__ == '__main__':
    main()
        
