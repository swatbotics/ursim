######################################################################
#
# zoombot/datalog.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################
#
# Class to log data and a function to read logged data from saved
# files. Underlying file format is the numpy compressed storage format
# .npz.
#
######################################################################

import sys, datetime
from collections import namedtuple

import numpy
import glfw

######################################################################

LoggerGroup = namedtuple('LoggerGroup',
                         'names, array, idx0, idx1')

######################################################################

class LogTimer:

    def __init__(self, array, idx, denom, display=None):
        assert idx < len(array) 
        if isinstance(denom, datetime.timedelta):
            denom = denom.total_seconds()
        self.array = array
        self.idx = idx
        self.denom = denom
        self.display = display

    def __enter__(self):
        self.start = glfw.get_time()

    def __exit__(self, type, value, traceback):
        elapsed = glfw.get_time() - self.start
        self.array[self.idx] = elapsed / self.denom
        if self.display is not None:
            print('{}: {}'.format(self.display, self.array[self.idx]))

######################################################################

class Logger:

    NUMERIC_TYPES = set('bufic')
    INITIAL_LOG_SIZE = 8000

    def __init__(self, dt=None):

        self.groups = []
        self.total_variables = 0

        self.variables = dict()

        if isinstance(dt, datetime.timedelta):
            dt = dt.total_seconds()

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
        assert not any([(name in self.variables) for name in names])
        
        nvars = len(names)
        
        idx0 = self.total_variables
        idx1 = idx0 + nvars

        group_idx = len(self.groups)

        for var_idx, name in enumerate(names):
            self.variables[name] = (group_idx, var_idx)
        
        self.groups.append(LoggerGroup(names, array, idx0, idx1))

        self.total_variables += nvars

        if self.current_log:
            
            print('WARNING: adding variables after logging started!')
            print('  this is inefficient :(')
            print('  logged values for {}'.format(names[0]))
            print('  and other variables in this group will be zero.')

            self.current_log.resize(self.current_log.shape[0],
                                    self.total_variables)

    def begin_log(self):

        assert self.current_log is None and self.log_rows == 0

        self.current_filename = datetime.datetime.now().strftime(
            'log_%Y%m%d_%H%M%S.npz')
        
        self.current_log = numpy.zeros((self.INITIAL_LOG_SIZE,
                                        self.total_variables),
                                       dtype=numpy.float32)

        print('starting log', self.current_filename)

        return self.current_filename
            
    def append_log_row(self):

        if self.current_log is None:
            self.begin_log()
        elif self.current_log.shape[0] == self.log_rows:
            self.current_log.resize(2*self.current_log.shape[0],
                                    self.total_variables)

        for g in self.groups:
            self.current_log[self.log_rows, g.idx0:g.idx1] = g.array

        self.log_rows += 1

    def write_log(self):

        assert self.current_log is not None

        log_names = []
        
        written_portion = self.current_log[:self.log_rows]

        if self.dt is not None:
            written_portion[:, 0] = numpy.arange(self.log_rows,
                                                 dtype=numpy.float32)*self.dt
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

    def lookup_variable(self, name):
        return self.variables[name]

    def timer(self, name, denom=1.0, display=None):
        group_idx, var_idx = self.lookup_variable(name)
        group = self.groups[group_idx]
        return LogTimer(group.array, var_idx, denom, display)

######################################################################            
        
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

