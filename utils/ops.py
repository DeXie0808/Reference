from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time


def mkdir(folders):
    if not isinstance(folders, (list, tuple)):
        folders = [folders]
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)


def add_path(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)

def Time():
    return time.time()