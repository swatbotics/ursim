ursim: Untitled Robot Simulator
===============================

Robot simulator developed for ENGR 028: Mobile Robotics,
taught at Swarthmore College in Fall 2020.

Installation
------------

#### This project was developed and tested with Python 3.7

Python 2.7 reached end of life in January 2020. You should verify that
you have a working Python 3 interpreter by going to your command
prompt and typing

    python --version

It should report back at least version 3.5.

If you don't see 3.5 or above, try running `python3 --version` or
`python3.7 --version` instead. If either of the latter two commands
yields a working Python interpreter of the correct version, replace
`python` in the second command below with either `python3` or
`python3.7` as appropriate.

Now clone this repository, then go into a terminal and enter the
following commands to creat and activate a virtual environment,
depending on your OS. Don't forget to replace `PATH_TO_URSIM` with
whatever path you cloned this repository into!

#### Windows:

    cd PATH_TO_URSIM
    python -m venv dev_env
    dev_env\Scripts\activate.bat

#### Mac OS or Linux:

    cd PATH_TO_URSIM
    python3 -m venv dev_env
    source dev_env/bin/activate
    
Note for Ubuntu users: please run the command

    sudo apt-get install libportaudio2

before running the software.
    
#### Finishing setup:

In either case, you will see the prefix `(dev_env)` in front of your
terminal command prompt to let you know the command worked.

Once you can see `(dev_env)` at the start of your terminal command
prompt, you can run the commands

    pip install --upgrade setuptools wheel pip
    python setup.py develop
    
First-time use
==============

If the setup.py command worked, you should then be able to run

    python -m ursim.demos.keyboard

which will bring up a demo with interactive control of a simulated
robot. You can use the mouse to aim the camera. The following keys
will control the robot:

   | Key          | Action        |
   | :--          | :--           |
   | `U`          | forward-left  |
   | `I`          | forward       |
   | `O`          | forward-right |
   | `J`          | spin left     |
   | `K`          | backward      |
   | `L`          | spin right    |
   | `Esc` or `Q` | quit          |

If everything worked so far, congratulations!

Leaving the virtual environment
-------------------------------

The `dev_env`
[virtual environment](https://docs.python.org/3/library/venv.html)
that we set up in the previous section is a useful way to store Python
modules for a project without needing to install them globally to the
entire system. If you ever want to return to the normal set of Python
packages available outside of the virtual environment, you can run the
command

    deactivate

whenever you see the `(dev_env)` prompt to return to your normal
environment. You can also use the `exit` command to exit the
terminal command prompt as you would normally.

Re-entering the virtual environment from a new terminal session
---------------------------------------------------------------

Whenever you open a new terminal session, you will need to re-activate
the virtual environment before using `ursim` by repeating these two
commands, depending on your OS.

#### Windows:

    cd PATH_TO_URSIM
    dev_env\Scripts\activate.bat

#### Mac OS or Linux:

    cd PATH_TO_URSIM
    source dev_env/bin/activate

You will want to confirm you see `(dev_env)` appear at the left of
your prompt before continuing.

Useful scripts and other demos
------------------------------

There are two scripts from the `ursim` you will frequently use when
working with the simulator.

#### `ursim.plotter`

The first useful program is the `ursim.plotter` module.  Try running
to plot variables from the log file created when you ran the keyboard
demo:

    python -m ursim.plotter

If the program exits with the message `no log files found!`, you can
re-run the keyboard control demo in the current directory and try
again.

It will bring up a series of windows that graph data from the most
recent data log in the current directory. On the timeseries plots that
appear (all but the final pose window), instead of using the toolbar
buttons to zoom in and out, I suggest just swiping over the time
interval of interest with your mouse. So to zoom all traces in a
figure to the time interval [1, 2], click the left mouse button in a
plot near t=1, drag to t=2, and release the mouse button. You can use
the `R` key to reset the view, or right-click to zoom out.

You can plot another log file besides the most recent one by supplying
a file name on the command line:

    python -m ursim.plotter LOGFILE.npz

You can also filter variables within a log file by supplying text that
will be matched against variable names. For example, to plot all
variables in a given log file matching `wheel` or `motor`, run

    python -m ursim.plotter LOGFILE.npz wheel motor

You can also specify the most recent log filename by using the `-l` flag:

    python -m ursim.plotter -l wheel motor

#### `ursim.cleanup`

If you want to get rid of a number of log files in the current directory,
you can run

    python -m ursim.cleanup

It will prompt you to confirm deletion of the log files. You can also run

    python -m ursim.cleanup -y

to skip the prompt. This also deletes camera image files created with
the `C` key (see below).

#### Other demos

There are a number of other robot demos in the `ursim.demos`
package. Try running each of the following:

    python -m ursim.demos.square
    python -m ursim.demos.bump
    python -m ursim.demos.blob_detection
    python -m ursim.demos.ctrl_datalog

Although the robot direction control keys are not available in these
demos, the following keys are always available in the simulator:

   | Key                 | Action                                           |
   | :-:                 | :--                                              |
   | `1`                 | toggle visualization of camera object detections |
   | `2`                 | toggle visualization of laser range data         |
   | `3`                 | toggle visualization of robot camera view        |
   | `R`                 | reset the simulation to the initial state        |
   | `C`                 | save camera images to current directory          |
   | `V`                 | restore the camera to the initial view           |
   | `Enter` or `Return` | pause/unpause simulation                         |
   | `Space`             | advance simulation by single step                |
   | `Esc` or `Q`        | quit                                             |

Writing your own robot controllers
----------------------------------

Now you have all the tools you need to start trying out your own robot
controllers.  I suggest you start by reading through the the code in
[the demos directory](src/ursim/demos).

I also strongly suggest that you read the relevant documentation in the
[`ursim.ctrl`](src/ursim/ctrl.py) submodule.

If you are interested in writing to the data log from your controller,
I suggest you check out the [`ursim.datalog`](src/ursim/datalog.py)
documentation as well.

This documentation is available through the built-in
[Python documentation system](https://docs.python.org/3/library/pydoc.html),
so you can run the commands

    python -m pydoc ursim.ctrl
    python -m pydoc ursim.datalog

You can also use the usual `pydoc` flags to output HTML or run a
documentation webserver.
