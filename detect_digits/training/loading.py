"""
Contains a set of utility functions for loading training data during the conv net training process.
"""

import itertools
import multiprocessing
import os
import random
import scipy as sp
import scipy.misc

from functools import partial

from detect_digits.training import CSVFILE
from detect_digits import IMAGE_SIZE, NUM_LENGTH_CLASSES

def validate_training_folders(training_folders):
    for training_folder in training_folders:
        if not os.path.isdir(training_folder):
            raise RuntimeError("The provided training folder '%s' does not exist."%training_folder)
        if not os.path.isfile(os.path.join(training_folder, CSVFILE)):
            raise RuntimeError("The provided training folder '%s' does not contain the training file descriptor '%s'."%(training_folder, CSVFILE))

def process_training_sample(folder, training_sample_line):
    """
    Performs processing of a single training sample line from a training CSV descriptor.
    :param training_sample_line: A single line from the training CSV descriptor.
    :return: A training sample tuple (training_image, length_class, digit_classes)
    """
    split_line = training_sample_line.split(',')
    assert len(split_line) == 2
    training_image_path = os.path.join(folder, split_line[0])
    training_image_label = split_line[1].strip()
    assert os.path.exists(training_image_path)
    training_image = sp.misc.imread(training_image_path)
    training_image = sp.misc.imresize(training_image, (IMAGE_SIZE, IMAGE_SIZE))
    length_class = min(len(training_image_label), NUM_LENGTH_CLASSES - 1)
    return training_image, length_class

def load_training_data(training_folders):
    """
    Load the training data. Utilizes all available CPU cores to speed up the process.
    :param training_folders: The set of training folders that contains the training data.
    :return: A set of training samples.
    """
    validate_training_folders(training_folders)
    thread_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    training_data_master = list()
    for training_folder in training_folders:
        partial_process_sample = partial(process_training_sample, training_folder)
        with open(os.path.join(training_folder, CSVFILE), 'r') as training_file:
            training_data = thread_pool.map(partial_process_sample, itertools.islice(training_file, 1, None))
            training_data_master.append(training_data)
    thread_pool.close()
    flattend_training_data = [training_sample for training_sublist in training_data_master for training_sample in training_sublist]
    del training_data
    random.shuffle(flattend_training_data)
    return flattend_training_data
