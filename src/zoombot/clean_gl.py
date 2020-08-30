######################################################################
#
# zoombot/clean_gl
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################
#
# PyOpenGL's OpenGL module wants you to do an import *
# but this module just does a search and replace on its
# exports to re-export them so for instance
#
#   OpenGL.GL.glEnable -> clean_gl.gl.Enable
#   OpenGL.GL_TRUE     -> clean_gl.gl.TRUE
#
# makes life easier so you can track where things are coming from
# without doing import *, which is dirty
#
# then in your own file you can do
#
#   from clean_gl import gl
#
######################################################################

import OpenGL.GL

class Namespace:
    pass

gl = Namespace()

for name, value in OpenGL.GL.__dict__.items():
    if name.startswith('gl'):
        newname = name[2:]
    elif name.startswith('GL_'):
        newname = name[3:]
    elif name.startswith('GL'):
        newname = name[2:]
    else:
        continue
    setattr(gl, newname, value)
