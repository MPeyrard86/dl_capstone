"""
Command line utility for converting the provided .mat training data into a more convenient CSV format.
"""

from __future__ import print_function

import h5py
import os
import sys

# TODO: Remove hard coding?
MATFILE = 'digitStruct.mat'
CSVFILE = 'digitStruct.csv'

def display_usage():
    print("Usage: python h5train2csv.py <data-folder>")

def parse_str_obj(str_obj):
    return ''.join(chr(x) for x in str_obj)

def parse_int_dataset(file, int_dataset):
    int_ref = int_dataset[0]
    if isinstance(int_ref, h5py.Reference):
        int_obj = file[int_ref]
        return int(int_obj[0])
    return int(int_ref)

def get_training_sample_filename(file, name_ref):
    filename_obj = file[name_ref]
    return parse_str_obj(filename_obj)

def get_training_sample_bounding_boxes(file, bbox_ref):
    bounding_box_obj = file[bbox_ref]
    label_dataset = bounding_box_obj['label']
    left_dataset = bounding_box_obj['left']
    top_dataset = bounding_box_obj['top']
    width_dataset = bounding_box_obj['width']
    height_dataset = bounding_box_obj['height']

    assert label_dataset.shape == left_dataset.shape == top_dataset.shape == width_dataset.shape == height_dataset.shape
    num_boxes = label_dataset.shape[0]
    bboxes = list()
    for i in xrange(num_boxes):
        bbox = dict()
        bbox['label'] = parse_int_dataset(file, label_dataset[i])
        bbox['left'] = parse_int_dataset(file, left_dataset[i])
        bbox['top'] = parse_int_dataset(file, top_dataset[i])
        bbox['width'] = parse_int_dataset(file, width_dataset[i])
        bbox['height'] = parse_int_dataset(file, height_dataset[i])
        bboxes.append(bbox)
    return bboxes


if __name__ == '__main__':
    if len(sys.argv) < 2:
        display_usage()
        sys.exit(1)

    data_folder = sys.argv[1]
    if not os.path.isdir(data_folder):
        print("The provided data folder %s does not exist." % data_folder)
        sys.exit(1)

    input_matfile = os.path.join(data_folder, MATFILE)
    if not os.path.isfile(input_matfile):
        print("The provided data folder %s does not contain a file called %s." % (data_folder, MATFILE))
        sys.exit(1)

    digit_struct_file = h5py.File(input_matfile, 'r')
    digit_struct = digit_struct_file['digitStruct']
    dataset_names = digit_struct['name']
    dataset_bounding_boxes = digit_struct['bbox']

    assert dataset_names.shape == dataset_bounding_boxes.shape
    num_training_examples = dataset_names.shape[0]

    training_samples = list()
    for i in xrange(num_training_examples):
        training_sample_filename = get_training_sample_filename(digit_struct_file, dataset_names[i][0])
        training_sample_bboxes = get_training_sample_bounding_boxes(digit_struct_file, dataset_bounding_boxes[i][0])
        training_samples.append((training_sample_filename, training_sample_bboxes))

    output_csvfile = os.path.join(data_folder, CSVFILE)
    with open(output_csvfile, 'w') as output_file:
        output_file.write('filename,label\n')
        for filename, bboxes in training_samples:
            output_file.write('{},{}\n'.format(filename, ''.join([str(b['label']) for b in bboxes])))

    digit_struct_file.close()
