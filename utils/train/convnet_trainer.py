"""
Command line utility for training the conv net for this project.
"""

from __future__ import print_function
import itertools
import os
import sys


def show_usage():
    print('Usage: python convnet_trainer.py <input CSV> <output-folder>')


if len(sys.argv) != 3:
    show_usage()
    sys.exit(1)

input_file = sys.argv[1]
output_dir = sys.argv[2]

if not input_file.endswith(".csv") or not os.path.exists(input_file):
    show_usage()
    sys.exit(1)
if os.path.isfile(output_dir):
    print("Output directory ", output_dir, " is actually a file.")
    sys.exit(1)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_set = list()
with open(input_file, 'r') as training_file:
    for line in itertools.islice(training_file, 1, None):
        sp = line.split(',')
        assert len(sp) == 2
        training_set.append((sp[0], sp[1].strip()))
