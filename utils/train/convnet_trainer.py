"""
Command line utility for training the conv net for this project.
"""

from __future__ import print_function
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.misc
import tensorflow as tf

img_side_length = 64
num_colour_channels = 3
max_digits = 5
num_length_classes = max_digits + 2 # +2: One for 0, one for 5+.
num_digit_classes = 10

def show_usage():
    print('Usage: python convnet_trainer.py <input CSV> <output-folder>')

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
def max_pool(X):
    return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def input_conv_layer(input, kernel_size, input_channels, output_channels):
    W_conv = weight_variable([kernel_size, kernel_size, input_channels, output_channels])
    b_conv = bias_variable([output_channels])
    return tf.nn.relu(conv2d(input, W_conv) + b_conv)
def conv_layer(input, kernel_size, input_channels, output_channels, drop_prob):
    W_conv = weight_variable([kernel_size, kernel_size, input_channels, output_channels])
    b_conv = bias_variable([output_channels])
    h_relu = tf.nn.relu(conv2d(input, W_conv) + b_conv)
    h_pool = max_pool(h_relu)
    return tf.nn.dropout(h_pool, drop_prob)


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
input_file_dir = os.path.dirname(input_file)

training_set = list()
with open(input_file, 'r') as training_file:
    for line in itertools.islice(training_file, 1, None):
        split_line = line.split(',')
        assert len(split_line) == 2
        training_image_path = os.path.join(input_file_dir, split_line[0])
        training_image_label = split_line[1].strip()
        training_set.append((training_image_path, training_image_label))
        assert os.path.exists(training_image_path)

normalized_training_set = list()
for training_sample in training_set:
    # Load and normalize the image.
    training_image = sp.misc.imread(training_sample[0])
    training_image = sp.misc.imresize(training_image, (img_side_length, img_side_length))
    training_image = training_image - np.mean(training_image)
    # Conver the label into a set of one-hot vectors.
    training_label = training_sample[1]
    length_label = np.zeros(num_length_classes)
    length_class = min(len(training_label), num_length_classes-1)
    length_label[length_class] = 1.0
    digit_labels = [np.zeros(num_digit_classes) for _ in range(max_digits)]
    for x in range(min(length_class, max_digits)):
        digit_labels[x][int(training_label[x])] = 1.0
    # Record normalized training samples.
    normalized_training_set.append((training_image, length_label, digit_labels))

conv1_num_filters = 48
conv2_num_filters = 64
conv3_num_filters = 128
conv4_num_filters = 160
conv5_num_filters = 192
conv6_num_filters = 192
conv7_num_filters = 192
conv8_num_filters = 192
fc1_num_features = 3072
fc2_num_features = 3072
fc3_num_features = 3072
convolution_kernel_size = 3

# Initialize input and training label placeholder variables.
X = tf.placeholder(tf.float32, shape=[None, img_side_length, img_side_length, num_colour_channels])
y_length = tf.placeholder(tf.float32, shape=[None, num_length_classes], name='y_length')
y_length_cls = tf.argmax(y_length, dimension=1)
y_digit1 = tf.placeholder(tf.float32, shape=[None, num_digit_classes], name='y_digit1')
y_digit1_cls = tf.argmax(y_digit1, dimension=1)
y_digit2 = tf.placeholder(tf.float32, shape=[None, num_digit_classes], name='y_digit2')
y_digit2_cls = tf.argmax(y_digit2, dimension=1)
y_digit3 = tf.placeholder(tf.float32, shape=[None, num_digit_classes], name='y_digit3')
y_digit3_cls = tf.argmax(y_digit3, dimension=1)
y_digit4 = tf.placeholder(tf.float32, shape=[None, num_digit_classes], name='y_digit4')
y_digit4_cls = tf.argmax(y_digit4, dimension=1)
y_digit5 = tf.placeholder(tf.float32, shape=[None, num_digit_classes], name='y_digit5')
y_digit5_cls = tf.argmax(y_digit5, dimension=1)

# Use the same dropout probability for all applicable layers.
dropout_probability = tf.placeholder(tf.float32)

# Construct convolutional layers.
conv1 = input_conv_layer(X, convolution_kernel_size, num_colour_channels, conv1_num_filters)
conv2 = conv_layer(conv1, convolution_kernel_size, conv1_num_filters, conv2_num_filters, dropout_probability)
conv3 = conv_layer(conv2, convolution_kernel_size, conv2_num_filters, conv3_num_filters, dropout_probability)
conv4 = conv_layer(conv3, convolution_kernel_size, conv3_num_filters, conv4_num_filters, dropout_probability)
conv5 = conv_layer(conv4, convolution_kernel_size, conv4_num_filters, conv5_num_filters, dropout_probability)
conv6 = conv_layer(conv5, convolution_kernel_size, conv5_num_filters, conv6_num_filters, dropout_probability)
conv7 = conv_layer(conv6, convolution_kernel_size, conv6_num_filters, conv7_num_filters, dropout_probability)
conv8 = conv_layer(conv7, convolution_kernel_size, conv7_num_filters, conv8_num_filters, dropout_probability)

