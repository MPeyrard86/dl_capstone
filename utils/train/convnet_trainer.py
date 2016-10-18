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

def weight_variable(name, shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def conv_weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
def bias_variable(name, shape):
    return tf.Variable(tf.constant(1.0, shape=shape), name=name)
def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
def max_pool(X):
    return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def input_conv_layer(name, input, kernel_size, input_channels, output_channels):
    W_conv = conv_weight_variable("W_" + name, [kernel_size, kernel_size, input_channels, output_channels])
    b_conv = bias_variable("b_" + name, [output_channels])
    return tf.nn.relu(conv2d(input, W_conv) + b_conv)
def conv_layer(name, input, kernel_size, input_channels, output_channels, drop_prob, pool=False):
    W_conv = conv_weight_variable("W_" + name, [kernel_size, kernel_size, input_channels, output_channels])
    b_conv = bias_variable("b_" + name, [output_channels])
    layer = tf.nn.relu(conv2d(input, W_conv) + b_conv)
    return tf.nn.dropout(max_pool(layer) if pool else layer, drop_prob)
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    nf = layer_shape[1:4].num_elements()
    flattened_layer = tf.reshape(layer, [-1, nf])
    return flattened_layer, nf
def dense_layer(name, input, input_features, output_features, drop_prob):
    W_fc = weight_variable("W_" + name, [input_features, output_features])
    b_fc = bias_variable("b_" + name, [output_features])
    return tf.nn.dropout(tf.nn.relu(tf.matmul(input, W_fc) + b_fc), drop_prob)
def output_layer(name, input, input_features, output_features):
    W_fc = weight_variable("W_" + name, [input_features, output_features])
    b_fc = bias_variable("b_" + name, [output_features])
    layer = tf.matmul(input, W_fc) + b_fc
    return layer

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

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

#Temporary for debugging
normalized_training_set = normalized_training_set[0:1000]



conv_kernel_size = 3
conv1_depth = 48
conv2_depth = 64
conv3_depth = 128
conv4_depth = 160
conv5_depth = 192
conv6_depth = 192
conv7_depth = 192
conv8_depth = 192
fc1_features = 4096

training_batch_size = 8
training_iterations = 20000

graph = tf.Graph()

with graph.as_default():
    X = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_colour_channels])
    y_length = tf.placeholder(tf.float32, shape=[None, num_length_classes], name='y_length')
    y_length_cls = tf.argmax(y_length, 1)
    keep_probability = tf.placeholder(tf.float32)

    conv1 = input_conv_layer("conv1", X, conv_kernel_size, num_colour_channels, conv1_depth)
    conv2 = conv_layer("conv2", conv1, conv_kernel_size, conv1_depth, conv2_depth, keep_probability)
    conv3 = conv_layer("conv3", conv2, conv_kernel_size, conv2_depth, conv3_depth, keep_probability)
    conv4 = conv_layer("conv4", conv3, conv_kernel_size, conv3_depth, conv4_depth, keep_probability, pool=True)
    conv5 = conv_layer("conv5", conv4, conv_kernel_size, conv4_depth, conv5_depth, keep_probability)
    conv6 = conv_layer("conv6", conv5, conv_kernel_size, conv5_depth, conv6_depth, keep_probability)
    conv7 = conv_layer("conv7", conv6, conv_kernel_size, conv6_depth, conv7_depth, keep_probability, pool=True)
    conv8 = conv_layer("conv8", conv7, conv_kernel_size, conv7_depth, conv8_depth, keep_probability)

    flat, nf = flatten_layer(conv8)
    fc1 = dense_layer("fc1", flat, nf, fc1_features, keep_probability)
    digit_length = output_layer("output", fc1, fc1_features, num_length_classes)

    y_pred = tf.nn.softmax(digit_length)
    y_pred_cls = tf.argmax(y_pred, 1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=digit_length, labels=y_length)
    loss_function = tf.reduce_mean(cross_entropy)


    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=digit_length, labels=y_length)
    learning_rate = tf.train.exponential_decay(0.05, tf.Variable(0), 10000, 0.95)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

    correct_prediction = tf.equal(y_pred_cls, y_length_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #
    # digit_length_training_predictions = tf.nn.softmax(digit_length)

    save = tf.train.Saver()



with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for i in range(training_iterations):
        batch = sample_training(normalized_training_set, training_batch_size)
        # _, l, predictions = session.run([optimizer, loss, y_length], feed_dict={X: batch[0], y_length: batch[1], keep_probability: 0.5})
        fd = {X: batch[0], y_length: batch[1], keep_probability: 0.5}
        session.run(optimizer, fd)
        if i % 100 == 0:
            acc = session.run(accuracy, fd)
            print("iteration {0:>6}, accuracy: {1:>6.1%}".format(i + 1, acc))
            # print('Minibatch loss at step %d: %f' % (i, l))
            # print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch[1]))
