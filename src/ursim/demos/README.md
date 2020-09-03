ursim demos
===========

This directory contains demonstrations illustrating several important aspects of writing programs for the robot simulator.

All of the programs here can be easily made to work outside of the ursim module by replacing the relative imports

~~~Python
from .. import ctrl
from ..app import RoboSimApp
~~~

with the absolute import line

~~~Python
from ursim import ctrl, RoboSimApp
~~~
