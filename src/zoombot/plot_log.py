######################################################################
#
# zoombot/plot_log.py
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

import numpy
import matplotlib
import matplotlib.pyplot as plt

from .datalog import read_log

######################################################################

MAX_PLOT_ROWS = 6
MAX_PLOT_COLS = 1

MAX_PLOTS_PER_FIGURE = MAX_PLOT_ROWS*MAX_PLOT_COLS

PLOT_MERGES = [
    re.compile(r'.*\.(cmd_)?vel(\.[xy])'),
    re.compile(r'(.*\.)(cmd_)?wheel_vel(\.raw|\.filtered)?\.l'),
    re.compile(r'(.*\.)(cmd_)?wheel_vel(\.raw|\.filtered)?\.r'),
    re.compile(r'(robot|odom)\.pos.x'),
    re.compile(r'(robot|odom)\.pos.y'),
    re.compile(r'(robot|odom)\.pos.angle'),
    re.compile(r'(robot|odom)\.(cmd_)?vel(\.raw|\.filtered)?\.forward'),
    re.compile(r'(robot|odom)\.(cmd_)?vel(\.raw|\.filtered)?\.angle'),
    re.compile(r'.*\.(cmd_)?vel\..*'),
    re.compile(r'robot\.(bump.*|motors_enabled)'),
    re.compile(r'blobfinder\.(.*\.)num_detections'),
    re.compile(r'blobfinder\.(.*\.)max_area'),
    re.compile(r'profiling(\..*)'),
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

def make_figure(fig_idx, total_figures, subplots):  
    fig = plt.figure(fig_idx)
    fig.canvas.mpl_connect('key_press_event', keyboard)
    fig.canvas.set_window_title('Figure {}/{}'.format(fig_idx, total_figures))
    fig.suptitle("Press 't' to toggle legend, 'q' to close")
    if subplots:
        ax = fig.subplots(MAX_PLOT_ROWS, MAX_PLOT_COLS, sharex=True)
        return fig, ax
    else:
        return fig

######################################################################
    
def plot_log(ldata, trace_names=[]):

    assert 'time' in ldata.keys()

    time = ldata['time']

    plots = []
    plot_lookup = dict()

    poses = []

    for name, trace in ldata.items():

        if name == 'time':
            continue

        if len(trace_names) and not any([frag in name for frag in trace_names]):
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

    fig_idx = 1
    fig, subplots = make_figure(fig_idx, total_figures, True)

    for pidx, plist in enumerate(plots):

        if cur_idx >= MAX_PLOTS_PER_FIGURE:
            fig_idx += 1
            fig, subplots = make_figure(fig_idx, total_figures, True)
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
            ax.plot(time, trace, label=name, **kwargs)

        last_in_row = ((cur_idx % MAX_PLOT_ROWS) == 0 or
                        pidx + 1 == len(plots))

        ax.legend(loc='upper right', fontsize='xx-small')


    if len(poses):

        fig_idx += 1
        fig = make_figure(fig_idx, total_figures, False)

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
    plot_log(ldata, extra_args)

######################################################################

if __name__ == '__main__':
    main()
        
