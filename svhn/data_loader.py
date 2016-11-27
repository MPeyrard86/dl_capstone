import multiprocessing
import os
import random
from functools import partial

import numpy as np
import scipy as sp
import scipy.misc

import itertools

from svhn import CSVFILE, MAX_DIGITS, IMAGE_SIZE, NUM_LENGTH_CLASSES, NUM_DIGIT_CLASSES

def process_sample(folder, training_sample_line):
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
    # Populate the digit classifications
    digit_classes = [int(x) for x in training_image_label]
    # Pad the ending of the digit with the empty label
    for _ in range(MAX_DIGITS-len(training_image_label)):
        digit_classes.append(NUM_DIGIT_CLASSES-1)
    return training_image, length_class, digit_classes

def load_data(training_folders):
    """
    Load the training data. Utilizes all available CPU cores to speed up the process.
    :param training_folders: The set of training folders that contains the training data.
    :return: A set of training samples.
    """
    thread_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    training_data_master = list()
    for training_folder in training_folders:
        partial_process_sample = partial(process_sample, training_folder)
        with open(os.path.join(training_folder, CSVFILE), 'r') as training_file:
            training_data = thread_pool.map(partial_process_sample, itertools.islice(training_file, 1, None))
            training_data_master.append(training_data)
    thread_pool.close()
    flattend_training_data = filter(lambda y: len(y[2]) <= MAX_DIGITS, [training_sample for training_sublist in training_data_master for training_sample in training_sublist])
    del training_data
    # random.shuffle(flattend_training_data)
    return flattend_training_data