import itertools
import multiprocessing
import os
from functools import partial

import scipy as sp
import scipy.misc

from svhn import CSVFILE, MAX_DIGITS, IMAGE_SIZE, NUM_LENGTH_CLASSES, NUM_DIGIT_CLASSES, IMAGE_FORMATS

def process_sample(folder, training_sample_line):
    """
    Performs processing of a single training sample line from a training CSV descriptor.
    :param training_sample_line: A single line from the training CSV descriptor.
    :return: A training sample tuple (training_image, length_class, digit_classes)
    """
    split_line = training_sample_line.strip().split(',')
    assert len(split_line) == 6
    training_image_path = os.path.join(folder, split_line[0])
    training_image_label = split_line[1]
    assert os.path.exists(training_image_path)
    training_image = sp.misc.imread(training_image_path)
    left = int(split_line[2])
    top = int(split_line[3])
    right = int(split_line[4])
    bottom = int(split_line[5])

    height = bottom - top
    delta_height = 0.5 * (1.15 * height - height)
    top = int(max(top - delta_height, 0))
    bottom = int(max(bottom + delta_height, training_image.shape[0]))
    training_image = training_image[top:bottom, left:right]

    training_image = sp.misc.imresize(training_image, (IMAGE_SIZE, IMAGE_SIZE))
    length_class = min(len(training_image_label), NUM_LENGTH_CLASSES - 1)
    # Populate the digit classifications
    digit_classes = [int(x) for x in training_image_label]
    # Pad the ending of the digit with the empty label
    for _ in range(MAX_DIGITS - len(training_image_label)):
        digit_classes.append(NUM_DIGIT_CLASSES - 1)
    return training_image, length_class, digit_classes

def process_image(image_file):
    return sp.misc.imresize(sp.misc.imread(image_file), (IMAGE_SIZE, IMAGE_SIZE))

def load_image_data(images_folder):
    image_files = filter(lambda x: any(x.endswith(y) for y in IMAGE_FORMATS), os.listdir(images_folder))
    image_files = [os.path.join(images_folder, f) for f in image_files]
    thread_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    try:
        return image_files, thread_pool.map(process_image, image_files)
    finally:
        thread_pool.close()

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
    return flattend_training_data

def get_model_file(model_folder):
    """
    Finds the most advanced model checkpoint from the given folder.
    The most advanced model is determined by the highest iteration count in the filename.
    :param model_folder: The folder containing the model checkpoints.
    :return: The filename (not including the folder) of the checkpoint file.
    """
    checkpoint_files = filter(lambda x: '.chk-' in x and not x.endswith('.meta'), os.listdir(model_folder))
    return max(checkpoint_files, key=lambda f: int(f.split('-')[-1]))

