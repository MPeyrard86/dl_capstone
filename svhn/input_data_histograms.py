import argparse
import os
import sys

import itertools
import matplotlib.pyplot as plt
import numpy as np

from svhn import CSVFILE

def get_digit_lengths(training_filename):
    """
    Determine how many training examples have digits of a certain lengh.
    :param training_filename: The trining file name.
    :return: An array of ints. First element is a count of samples with length 1, etc.
    """
    digit_lengths = list()
    with open(training_filename, 'r') as training_file:
        for sample in itertools.islice(training_file, 1, None):
            split_line = sample.split(',')
            assert len(split_line) == 2
            training_image_label = split_line[1].strip()
            label_len = len(training_image_label)
            if label_len <= 5:
                digit_lengths.append(label_len)
    return digit_lengths

def get_digits(training_filename):
    """
    Extracts each individual digit from each data entry.
    :param training_filename: The CSV file containing the digit labels.
    """
    digits = list()
    with open(training_filename, 'r') as training_file:
        for sample in itertools.islice(training_file, 1, None):
            split_line = sample.split(',')
            training_image_label = split_line[1].strip()
            # Ignore digits over length 5.
            if len(training_image_label) <= 5:
                for d in training_image_label:
                    digits.append(int(d))
    return digits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training-folder", required=True, type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.training_folder):
        print "Provided training folder does not exist."
        sys.exit(1)
    tfilename = os.path.join(args.training_folder, CSVFILE)
    if not os.path.isfile(tfilename):
        print "Provided training file does not exist."
        sys.exit(1)

    lbuckets = get_digit_lengths(tfilename)
    counts = np.bincount(lbuckets)[1:]
    total = np.sum(counts)
    len1_ratio = float(counts[0]) / total
    len2_ratio = float(counts[1]) / total
    len3_ratio = float(counts[2]) / total
    len4_ratio = float(counts[3]) / total
    len5_ratio = float(counts[4]) / total
    print len1_ratio, len2_ratio, len3_ratio, len4_ratio, len5_ratio

    fig, ax = plt.subplots()
    ax.bar([1,2,3,4,5], counts, align='center')
    ax.set(xticks=[1,2,3,4,5])
    ax.set_xlabel("Digit Length")
    ax.set_ylabel("Frequency")
    plt.show()

    _digits = get_digits(tfilename)
    fig, ax = plt.subplots()
    ax.bar(range(10), np.bincount(_digits), align='center')
    ax.set(xticks=range(10))
    ax.set_xlabel("Digits")
    ax.set_ylabel("Frequency")
    plt.show()