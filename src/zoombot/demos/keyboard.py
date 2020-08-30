######################################################################
#
# zoombot/demos/keyboard.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################


from ..app import RoboSimApp
from ..find_path import find_path

def main():

    app = RoboSimApp()
    app.sim.load_svg(find_path('environments/first_environment.svg'))
    app.run()

if __name__ == '__main__':
    main()
