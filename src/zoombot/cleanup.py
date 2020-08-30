######################################################################
#
# zoombot/cleanup.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import os, re, sys

def main():

    confirmed = False

    if len(sys.argv) > 1:
        if len(sys.argv) == 2 and sys.argv[1] == '-y':
            confirmed = True
        else:
            print('usage: cleanup [-y]')
            print()
            sys.exit(1)


    log_file_expr = re.compile(r'log_[0-9]+_[0-9]+.npz')
    camera_file_expr = re.compile(r'camera_\w+_[0-9]+.png')

    to_remove = []

    print('searching for camera images and log files:')

    for fname in sorted(os.listdir()):
        if log_file_expr.match(fname) or camera_file_expr.match(fname):
            print('  found', fname)
            to_remove.append(fname)

    if len(to_remove):

        if not confirmed:
            while True:
                answer = input('remove {} files? [y/n] '.format(len(to_remove))).lower()
                if answer == 'n':
                    sys.exit(0)
                elif answer == 'y':
                    break

        for fname in to_remove:
            os.unlink(fname)
            print('  deleted', fname)

    else:

        print('no files found!')

if __name__ == '__main__':
    main()
