######################################################################
#
# zoombot/transform2d.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################
#
# Class and utility functions for 2D rigid transformations.
#
######################################################################

import numpy

######################################################################

class Transform2D:

    """Class to encapsulate a 2D rigid transformation."""

    def __init__(self, *args):
        """Intialize a new Transform2D. Can be called in one of several ways:

          * Transform2D(x, y, angle) 

          * Transform2D(position, angle) where position is a list,
            tuple or numpy array containing two numbers

          * Transform2D(other) is effectively the same as calling
            Transform2D(other.position, other.angle)

          * Transform2D() constructs the identity transform

        Angles are always specified in radians.

        The resulting object's translation and rotation can be accessed 
        through the position and angle members, as shown below:

           >>> xform = Transform2D((3, 2), 1)
           >>> print(xform.position)
           [3. 2.]
           >>> print(xform.angle)
           1
        """
        
        self.position = numpy.empty(2, dtype=numpy.float32)
        self._angle = numpy.float32()

        if len(args) == 3:
            self.position[:] = args[:2]
            self._angle = args[2]
        elif len(args) == 2:
            self.position[:] = args[0]
            self._angle = args[1]
        elif len(args) == 1:
            self.position[:] = args[0].position
            self._angle = args[0].angle
        elif len(args) == 0:
            self.position[:] = 0
            self._angle = numpy.float32(0)
        else:
            raise RuntimeError('invalid arguments to Transform2D.__init__')
        
        self._matrix = numpy.zeros((3, 3), dtype=numpy.float32)

    def copy(self):
        """Returns a duplicate of this object."""
        return self.__class__(self.position, self.angle)

    def rotation_matrix(self):
        """Returns the 2x2 rotation matrix associated with this
        transformation, of the form 

          R = [ c, -s ]
              [ s,  c ]

        where c = numpy.cos(self.angle) and s = numpy.sin(self.angle).
        """
        return self.matrix[:2,:2]

    @property
    def matrix(self):
        """Property to retrieve the homogeneous 3x3 transformation matrix
        associated with this transformation, of the form
        
              [ c, -s, x ] 
          M = [ s,  c, y ]
              [ 0,  0, 1 ]
        
        where c = numpy.cos(self.angle), s = numpy.sin(self.angle), 
        and (x, y) are the elements of self.position.
        """
        if self._matrix[2,2] != 1:
            x, y = self.position
            c = numpy.cos(self._angle)
            s = numpy.sin(self._angle)
            self._matrix[:] = [[c, -s, x], [s, c, y], [0, 0, 1]]
        return self._matrix

    @property
    def angle(self):
        """Property to get/set the angle part of this transformation."""
        return self._angle

    @angle.setter
    def angle(self, value):
        # NB: need to invalidate matrix when angle is set
        self._angle = value
        self._matrix[2,2] = 0

    def transform_fwd(self, other):
        """Right-multiply the other object by this transformation T2.  

        If the other object is a list, tuple, or numpy array
        containing two numbers [u, v], returns a transformed point
        of the form

          [ u' ] = [ c -s ] [ u ] + [ x ]
          [ v' ]   [ s  c ] [ v ]   [ y ]

        where c = numpy.cos(self.angle), s = numpy.sin(self.angle), 
        and (x, y) are the elements of self.position.

        If the other object is also a transformation T1, returns the
        composition of transformations T3 = T2 * T1 such that
        
          T3(P) = T2( T1(P) ).
        """

        R2 = self.rotation_matrix()
        t2 = self.position
        
        if isinstance(other, self.__class__):
            
            t1 = other.position
            
            return self.__class__(numpy.dot(R2, t1) + t2,
                                  self.angle + other.angle)

        else:
            
            return numpy.dot(R2, other) + t2


    def transform_inv(self, other):
        """Right-multiply the other object by the inverse of this
        transformation T2.
        
        If the other object is a list, tuple, or numpy array
        containing two numbers [u, v], returns a transformed point
        of the form

          [ u' ] = [  c s ] ( [ u ] - [ x ] )
          [ v' ]   [ -s c ] ( [ v ]   [ y ] ) 

        where c = numpy.cos(self.angle), s = numpy.sin(self.angle), 
        and (x, y) are the elements of self.position.

        If the other object is also a transformation T1, returns the
        composition of transformations T3 = T2^(-1) * T1 such that
        
          T3(P) = T2^(1)( T1(P) ).
        """

        R2inv = self.rotation_matrix().T
        t2 = self.position

        if isinstance(other, self.__class__):

            t1 = other.position

            return self.__class__(numpy.dot(R2inv, t1 - t2),
                                  other.angle - self.angle)

        else:

            return numpy.dot(R2inv, other - t2)

    def inverse(self):

        """Return a new transformation corresponding to the inverse of this
        transformation. The inverse's angle is the negative of the
        current angle, and the inverse's position is given by

          [ x' ] = [ c   s ] [ -x ]
          [ y' ]   [ -s  c ] [ -y ]

        where c = numpy.cos(self.angle), s = numpy.sin(self.angle), 
        and (x, y) are the elements of self.position.
        
        This method does not modify the original transformation.
        """

        R2inv = self.rotation_matrix().T
        pinv = numpy.dot(R2inv, -self.position)

        return self.__class__(pinv, -self.angle)

    def __mul__(self, other):
        """Convenience wrapper around self.transform_fwd(other)."""
        return self.transform_fwd(other)

    def __str__(self):
        """Convert to string."""
        return repr(self)

    def __repr__(self):
        """Convert to string that can be evaluated to construct an 
        equivalent transformation."""
        return 'Transform2D(({}, {}), {})'.format(
            repr(self.position[0]), repr(self.position[1]),
            repr(self._angle))

######################################################################
        
def _test_transform_2d():

    for attempt in range(100):

        x0, y0 = numpy.random.random(2)*2 - 1
        angle0 = (numpy.random.random()*2-1) * numpy.pi

        T0 = Transform2D(x0, y0, angle0)

        T0_alternatives = [
            Transform2D((x0, y0), angle0),
            Transform2D(T0.position, T0.angle),
            Transform2D(T0),
        ]

        print('T0 =', T0)

        for T0_alt in T0_alternatives:
            assert numpy.all(T0_alt.position == T0.position)
            assert numpy.isclose(T0_alt.angle, T0.angle)

        T0inv = T0.inverse()

        print('T0inv =', T0inv)

        T0T0inv = T0 * T0inv

        print('T0T0inv =', T0T0inv)
        assert numpy.abs(T0T0inv.position).max() < 1e-6
        assert T0T0inv.angle == 0.0

        T0invT0 = T0 * T0inv

        print('T0invT0 =', T0invT0)
        assert numpy.abs(T0invT0.position).max() < 1e-6
        assert T0invT0.angle == 0.0

        x1, y1 = numpy.random.random(2)*2 - 1
        angle1 = (numpy.random.random()*2-1) * numpy.pi

        T1 = Transform2D((x1, y1), angle1)
        
        x, y = numpy.random.random(2)*2 - 1
        p = numpy.array([x, y])
        print('p =', p)


        T0invT0p = T0.transform_inv(T0.transform_fwd(p))

        print('T0invT0p =', T0invT0p)

        assert numpy.all(numpy.isclose(T0invT0p, p))

        T0T0invp = T0.transform_fwd(T0.transform_inv(p))

        print('T0T0invp =', T0T0invp)

        assert numpy.all(numpy.isclose(T0T0invp, p, 1e-4))
        

        T1T0 = T1 * T0
        T1T0T0inv = T1T0 * T0.inverse()
        T1T0T0invT1inv = T1T0T0inv * T1.inverse()

        print('T1 =', T1)
        print('T1T0 =', T1T0)
        print('T1T0T0inv =', T1T0T0inv)
        print('T1T0T0invT1inv =', T1T0T0invT1inv)

        assert numpy.all(numpy.isclose(T1T0T0inv.position, T1.position, 1e-4))
        assert numpy.isclose(T1T0T0inv.angle, T1.angle)

        assert numpy.abs(T1T0T0invT1inv.position).max() < 1e-5
        assert numpy.abs(T1T0T0invT1inv.angle) < 1e-5
        

        print()

    T = Transform2D((2, 1), numpy.pi/2)

    pA = [0, 0]
    TpA = T * pA
    TpA_expected = [2, 1]

    pB = [1, 0]
    TpB = T * pB
    TpB_expected = [2, 2]

    pC = [0, 1]
    TpC = T * pC
    TpC_expected = [1, 1]

    print('T =', T)
    print('pA =', pA)
    print('TpA = ', TpA)
    assert numpy.all(numpy.isclose(TpA, TpA_expected))
    assert numpy.all(numpy.isclose(T.transform_inv(TpA), pA))
 
    print('pB =', pB)
    print('TpB = ', TpB)
    assert numpy.all(numpy.isclose(TpB, TpB_expected))
    assert numpy.all(numpy.isclose(T.transform_inv(TpB), pB))

    print('pC =', pC)
    print('TpC = ', TpC)
    assert numpy.all(numpy.isclose(TpC, TpC_expected))
    assert numpy.all(numpy.isclose(T.transform_inv(TpC), pC))

    Tnull = Transform2D()

    assert numpy.all(Tnull.position == 0)
    assert Tnull.angle == 0

    Tcopy = Transform2D(T)
    assert numpy.all(Tcopy.position == (2, 1))

    Tcopy.position = (1, 2)
    assert numpy.all(Tcopy.position == (1, 2))
    assert numpy.all(T.position == (2, 1))

    print('...transforms seem to work OK!')

######################################################################
    
if __name__ == '__main__':

    _test_transform_2d()
