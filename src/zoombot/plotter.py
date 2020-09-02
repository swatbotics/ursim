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

DEFAULT_EXCLUDES = [
    'profiling_camera'
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
    elif event.key == 'r':
        print('reset view')
        fig = event.canvas.figure
        for ax in fig.axes:
            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)
            ax.autoscale_view(True)
        event.canvas.draw()

######################################################################

class PlotManager:

    def __init__(self, fig):

        self.fig = fig

        self.mouse_down = None
        
    def mouse_press(self, event):
        
        if event.inaxes is None:
            return

        if self.mouse_down is not None:
            return

        btn = None
        
        if event.button == backend_bases.MouseButton.LEFT:
            btn = 'left'
        elif event.button == backend_bases.MouseButton.RIGHT:
            btn = 'right'

        if btn is not None:
            self.mouse_down = (btn, event.inaxes, event.xdata, event.ydata)

    def set_xlim(self, ax, x0, x1):

        for sib in ax.figure.axes:

            ymin = 1e5
            ymax = -1e5

            for line in sib.lines:
                xdata, ydata = line.get_data()
                mask = (xdata >= x0) & (xdata <= x1)
                yvals = ydata[mask]
                if len(yvals):
                    ymin = min(ymin, yvals.min())
                    ymax = max(ymax, yvals.max())

            if ymax > ymin:
                m = 0.1*(ymax-ymin)
                sib.set_ylim(ymin-m, ymax+m)

        ax.set_xlim(x0, x1)
        ax.figure.canvas.draw()
        

    def mouse_release(self, event):
        
        if self.mouse_down is not None:
            
            btn, ax, x0, y0 = self.mouse_down

            x1, y1 = event.xdata, event.ydata

            if x1 is None or y1 is None:
                x1, y1 = ax.transData.inverted().transform((event.x, event.y))

            if btn == 'left':

                self.set_xlim(ax, x0, x1)

            elif btn == 'right':
                
                xmid = 0.5*(x0+x1)
                xmin, xmax = ax.get_xlim()
                xrng = numpy.abs(xmax-xmin)

                self.set_xlim(ax, xmid-0.75*xrng, xmid+0.75*xrng)

        self.mouse_down = None

    def mouse_enter_axes(self, event):
        event.inaxes.format_coord = lambda x, y: self.format_coord(event.inaxes, x, y)

    def format_coord(self, ax, x, y):

        outputs = ['t={:.2f}'.format(x)]

        for line in ax.lines:
            lx, ly = line.get_data()
            py = numpy.interp(x, lx, ly)
            name = re.sub(r'^[^.]+\.', '', line.get_label())
            outputs.append('{}={:.3g}'.format(name, py))
        
        return ' '.join(outputs)

    def setup(self):

        self.fig.canvas.mpl_connect('button_press_event', self.mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.mouse_release)
        self.fig.canvas.mpl_connect('axes_enter_event', self.mouse_enter_axes)


######################################################################

def make_figure(filename, fig_idx, total_figures, subplots):  
    fig = plt.figure(fig_idx)
    fig.canvas.mpl_connect('key_press_event', keyboard)
    fig.canvas.set_window_title('{} - figure {}/{}'.format(filename, fig_idx, total_figures))
    if subplots:
        fig.suptitle("Press 't' to toggle legend, 'q' to close, 'r' to reset view")
        axis_list = fig.subplots(MAX_PLOT_ROWS, MAX_PLOT_COLS, sharex=True)
        mgr = PlotManager(fig)
        return fig, axis_list, mgr
    else:
        fig.suptitle("Press 't' to toggle legend, 'q' to close")
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

        dot = name.find('.')
        if dot >= 0:
            group = name[:dot]
            remainder = name[dot+1:]
        else:
            group = name
            remainder = ''
        
        if group == 'pos_x':
            py = 'pos_y.' + remainder
            if py in ldata:
                poses.append(remainder)

        if group in plot_lookup:
            plot_idx = plot_lookup[group]
        else:
            plot_idx = len(plots)
            plots.append([])
            plot_lookup[group] = plot_idx

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

        for idx, (name, trace) in enumerate(plist):
            zorder = len(plist) - idx
            kwargs = dict()
            for color, cvalue in COLORS.items():
                if color in name:
                    kwargs['color'] = cvalue
            if 'angle' in name:
                trace = trace * 180 / numpy.pi
            line, = ax.plot(time, trace, label=name, **kwargs, zorder=zorder)

        last_in_row = ((cur_idx % MAX_PLOT_ROWS) == 0 or
                        pidx + 1 == len(plots))

        ax.legend(loc='upper right', fontsize='xx-small')

    mgr.setup()
    #sys.exit(0)

    if len(poses):

        fig_idx += 1
        fig = make_figure(fname, fig_idx, total_figures, False)

        for idx, pname in enumerate(poses):

            print('found pose', pname)
            
            x = ldata['pos_x.' + pname]
            y = ldata['pos_y.' + pname]

            handle, = plt.plot(x, y, label=pname, zorder=2)
            color = numpy.array(matplotlib.colors.to_rgb(handle.get_color()))
            
            tname = 'angle.' + pname
            if tname in ldata:
                theta = ldata[tname]
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                plt.quiver(x[::8], y[::8], c[::8], s[::8],
                           color=(0.7*color + 0.3),
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

    _, ldata = read_log(filename, as_dict=True)
    plot_log(os.path.basename(filename), ldata, extra_args)

######################################################################

if __name__ == '__main__':
    main()
        
