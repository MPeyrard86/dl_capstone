"""
Command line utility for training the conv net for this project.
"""

from __future__ import print_function
import itertools
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.misc
import tensorflow as tf

image_size = 64 # Size of one dimension
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
def conv_layer(input, kernel_size, input_channels, output_channels, drop_prob, pool=False):
    W_conv = weight_variable([kernel_size, kernel_size, input_channels, output_channels])
    b_conv = bias_variable([output_channels])
    layer = tf.nn.relu(conv2d(input, W_conv) + b_conv)
    return tf.nn.dropout(max_pool(layer) if pool else layer, drop_prob)
def dense_layer(input, input_features, output_features, drop_prob):
    W_fc = weight_variable([input_features, output_features])
    b_fc = bias_variable([output_features])
    return tf.nn.dropout(tf.nn.relu(tf.matmul(tf.reshape(input, [-1, input_features]), W_fc) + b_fc), drop_prob)
def output_layer(input, input_features, output_features):
    W_fc = weight_variable([input_features, output_features])
    b_fc = bias_variable([output_features])
    return tf.nn.softmax(tf.matmul(input, W_fc) + b_fc)

def sample_training(training_source, num_samples):
    sample = random.sample(training_source, num_samples)
    sampled_images = [x[0] for x in sample]
    sampled_lengths = [x[1] for x in sample]
    return sampled_images, sampled_lengths

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
    training_image = sp.misc.imresize(training_image, (image_size, image_size))
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

sess = tf.InteractiveSession()

# Initialize input and training label placeholder variables.
X = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_colour_channels])
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
conv3 = conv_layer(conv2, convolution_kernel_size, conv2_num_filters, conv3_num_filters, dropout_probability, pool=True)
conv4 = conv_layer(conv3, convolution_kernel_size, conv3_num_filters, conv4_num_filters, dropout_probability)
conv5 = conv_layer(conv4, convolution_kernel_size, conv4_num_filters, conv5_num_filters, dropout_probability, pool=True)
conv6 = conv_layer(conv5, convolution_kernel_size, conv5_num_filters, conv6_num_filters, dropout_probability)
conv7 = conv_layer(conv6, convolution_kernel_size, conv6_num_filters, conv7_num_filters, dropout_probability, pool=True)
conv8 = conv_layer(conv7, convolution_kernel_size, conv7_num_filters, conv8_num_filters, dropout_probability)

pooled_image_size = image_size / 2**3
fc1_input_size = pooled_image_size*pooled_image_size*conv8_num_filters

# Logistic classifier for y_length
length_fc1 = dense_layer(conv8, fc1_input_size, fc1_num_features, dropout_probability)
length_fc2 = dense_layer(length_fc1, fc1_num_features, fc2_num_features, dropout_probability)
length_final = dense_layer(length_fc2, fc2_num_features, num_length_classes, dropout_probability)
# Logistic classifier for digit 1
digit1_fc1 = dense_layer(conv8, fc1_input_size, fc1_num_features, dropout_probability)
digit1_fc2 = dense_layer(digit1_fc1, fc1_num_features, fc2_num_features, dropout_probability)
digit1_final = dense_layer(digit1_fc2, fc2_num_features, num_digit_classes, dropout_probability)
# Logistic classifier for digit 2
digit2_fc1 = dense_layer(conv8, fc1_input_size, fc1_num_features, dropout_probability)
digit2_fc2 = dense_layer(digit2_fc1, fc1_num_features, fc2_num_features, dropout_probability)
digit2_final = dense_layer(digit2_fc2, fc2_num_features, num_digit_classes, dropout_probability)
# Logistic classifier for digit 3
digit3_fc1 = dense_layer(conv8, fc1_input_size, fc1_num_features, dropout_probability)
digit3_fc2 = dense_layer(digit3_fc1, fc1_num_features, fc2_num_features, dropout_probability)
digit3_final = dense_layer(digit3_fc2, fc2_num_features, num_digit_classes, dropout_probability)
# Logistic classifier for digit 4
digit4_fc1 = dense_layer(conv8, fc1_input_size, fc1_num_features, dropout_probability)
digit4_fc2 = dense_layer(digit4_fc1, fc1_num_features, fc2_num_features, dropout_probability)
digit4_final = dense_layer(digit4_fc2, fc2_num_features, num_digit_classes, dropout_probability)
# Logistic classifier for digit 5
digit5_fc1 = dense_layer(conv8, fc1_input_size, fc1_num_features, dropout_probability)
digit5_fc2 = dense_layer(digit5_fc1, fc1_num_features, fc2_num_features, dropout_probability)
digit5_final = dense_layer(digit5_fc2, fc2_num_features, num_digit_classes, dropout_probability)

# Training for y_length
cross_entropy_length = tf.nn.softmax_cross_entropy_with_logits(logits=length_final, labels=y_length)
length_optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_length)
correct_length = tf.equal(tf.argmax(length_final, 1), tf.argmax(y_length, 1))
length_accuracy = tf.reduce_mean(tf.cast(correct_length, tf.float32))
# Training for digit 1
cross_entropy_digit1 = tf.nn.softmax_cross_entropy_with_logits(logits=digit1_final, labels=y_digit1)
digit1_optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_digit1)
correct_digit1 = tf.equal(tf.argmax(digit1_final, 1), tf.argmax(y_digit1, 1))
# Training for digit 2
cross_entropy_digit2 = tf.nn.softmax_cross_entropy_with_logits(logits=digit2_final, labels=y_digit2)
digit2_optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_digit2)
correct_digit2 = tf.equal(tf.argmax(digit2_final, 1), tf.argmax(y_digit2, 1))
# Training for digit 3
cross_entropy_digit3 = tf.nn.softmax_cross_entropy_with_logits(logits=digit3_final, labels=y_digit3)
digit3_optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_digit3)
correct_digit3 = tf.equal(tf.argmax(digit3_final, 1), tf.argmax(y_digit3, 1))
# Training for digit 4
cross_entropy_digit4 = tf.nn.softmax_cross_entropy_with_logits(logits=digit4_final, labels=y_digit4)
digit4_optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_digit4)
correct_digit4 = tf.equal(tf.argmax(digit4_final, 1), tf.argmax(y_digit4, 1))
# Training for digit 5
cross_entropy_digit5 = tf.nn.softmax_cross_entropy_with_logits(logits=digit5_final, labels=y_digit5)
digit5_optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_digit5)
correct_digit5 = tf.equal(tf.argmax(digit5_final, 1), tf.argmax(y_digit5, 1))

sess.run(tf.initialize_all_variables())
for i in range(2000):
    batch = sample_training(normalized_training_set, 50)
    if i%100 == 0:
        acc_len = length_accuracy.eval(feed_dict={X: batch[0], y_length: batch[1], dropout_probability: 1.0})
        print('step %d, training accuracy: %g'%(i, acc_len))
    length_optimizer.run(feed_dict={X: batch[0], y_length: batch[1], dropout_probability:0.5})

