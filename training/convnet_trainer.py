"""
A tool used to train the convolutional neural network for my Udacity Machine Learning Nanodegree Capstone Project.
"""

from __future__ import print_function

import sys
import time

from training import TRAINING_RATIO
from training.loading import *

USAGE_MESSAGE = """Usage: python convnet_trainer.py <input-data-folders>
This script expects a set of training folders as parameters.
These folders are expected to contain the set of training images and a file called '%s'.
If your training folder does not contain such a file, then see the documentation for the h5train2csv tool to generate it."""%(CSVFILE)

def get_training_folder_names():
    if len(sys.argv) < 2:
        raise RuntimeError('Missing list of training folders.')
    return sys.argv[1:]


if __name__ == '__main__':
    try:
        training_folders = get_training_folder_names()
        print("Loading training data from sources: %s"%training_folders)
        print("This may take a few minutes depending on how much data you loaded and how many CPU cores you have.")
        train_time_start = time.time()
        training_data = load_training_data(training_folders)
        # train_validation_split_point = train_validation_split(training_data, TRAINING_RATIO)
        train_time_end = time.time()
        print("Loaded %d training samples in %fs."%(len(training_data), train_time_end - train_time_start))
    except RuntimeError as x:
        print('RuntimeError: ' + x.message)
        print(USAGE_MESSAGE)
        sys.exit(1)
    except Exception as x:
        print(x)
        print('Unknown error. Terminating.')
        sys.exit(2)
