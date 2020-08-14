# PyOpenGL's OpenGL module wants you to do an import *
# but this module just does a search and replace on its
# exports to re-export them so for instance
#
#   OpenGL.GL.glEnable -> CleanGL.gl.Enable
#   OpenGL.GL_TRUE     -> CleanGL.gl.TRUE
#
# makes life easier so you can track where things are coming from
# without doing import *, which is dirty
#
# then in your own file you can do
#
#   from CleanGL import gl

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
