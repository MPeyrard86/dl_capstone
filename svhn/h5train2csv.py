"""
Command line utility for converting the provided .mat training data into a more convenient CSV format.
"""

import h5py
import os
import sys

from svhn import CSVFILE, MATFILE

def display_usage():
    """
    Displays the usage message.
    """
    print("Usage: python h5train2csv.py <data-folder>")

def parse_str_obj(str_obj):
    """
    Extracts the characters from the string object and concatenates them into an actual string.
    :param str_obj: The string object.
    """
    return ''.join(chr(x) for x in str_obj)

def parse_int_dataset(file, int_dataset):
    """
    Parses an integer from the provided data set.
    :param file: The hd5 file.
    :param int_dataset: The data set containing the integer.
    :return: The parsed integer.
    """
    int_ref = int_dataset[0]
    if isinstance(int_ref, h5py.Reference):
        int_obj = file[int_ref]
        return int(int_obj[0])
    return int(int_ref)

def get_training_sample_filename(file, name_ref):
    """
    Extracts the training sample filename from the name reference.
    """
    filename_obj = file[name_ref]
    return parse_str_obj(filename_obj)

def get_training_sample_bounding_boxes(file, bbox_ref):
    """
    Extracts the bounding boxes from the object.
    """
    bounding_box_obj = file[bbox_ref]
    label_dataset = bounding_box_obj['label']
    left_dataset = bounding_box_obj['left']
    top_dataset = bounding_box_obj['top']
    width_dataset = bounding_box_obj['width']
    height_dataset = bounding_box_obj['height']

    assert label_dataset.shape == left_dataset.shape == top_dataset.shape == width_dataset.shape == height_dataset.shape
    num_boxes = label_dataset.shape[0]
    labels = list()
    lefts = list()
    tops = list()
    rights = list()
    bottoms = list()
    for i in xrange(num_boxes):
        label = parse_int_dataset(file, label_dataset[i])
        left = parse_int_dataset(file, left_dataset[i])
        top = parse_int_dataset(file, top_dataset[i])
        width = parse_int_dataset(file, width_dataset[i])
        height = parse_int_dataset(file, height_dataset[i])

        labels.append(str(label))
        # bounds.append((left, top, left+width, top-height))
        lefts.append(left)
        tops.append(top)
        rights.append(left+width)
        bottoms.append(top+height)
    bbox = {
        'label': ''.join(labels),
        'left': max(min(lefts), 0),
        'top': max(min(tops), 0),
        'right': max(max(rights), 0),
        'bottom': max(max(bottoms), 0)
    }

    return bbox


if __name__ == '__main__':
    if len(sys.argv) < 2:
        display_usage()
        sys.exit(1)

    data_folder = sys.argv[1]
    if not os.path.isdir(data_folder):
        print "The provided data folder %s does not exist." % data_folder
        sys.exit(1)

    input_matfile = os.path.join(data_folder, MATFILE)
    if not os.path.isfile(input_matfile):
        print "The provided data folder %s does not contain a file called %s." % (data_folder, MATFILE)
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
        output_file.write('filename,label,left,top,right,bottom\n')
        for filename, bb in training_samples:
            output_file.write('%s,%s,%s,%s,%s,%s\n'%(filename, bb['label'], bb['left'], bb['top'], bb['right'], bb['bottom']))

    digit_struct_file.close()