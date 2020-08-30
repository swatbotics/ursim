from ..app import RoboSimApp
from ..find_path import find_path

def main():

    app = RoboSimApp()
    app.sim.load_svg(find_path('environments/first_environment.svg'))
    app.run()

if __name__ == '__main__':
    main()
