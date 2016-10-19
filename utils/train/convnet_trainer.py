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

# Length of each size of the image (square shape assumed).
image_size = 32
# Number of colour channels on the input.
input_channels = 3
# Number of length classes. 0-5, and a 5+ class.
num_length_classes = 7
# Number of digit classifications. 0-9, and a blank class.
num_digit_classes = 11

training_batch_size = 1024
conv_kernel_size = 5
conv_depth1 = 16
conv_depth2 = 32
conv_depth3 = 64
fc_size1 = 128

training_shape = (training_batch_size, image_size, image_size, input_channels)

# max_digits = 5
# num_length_classes = max_digits + 2 # +2: One for 0, one for 5+.
# num_digit_classes = 10

def show_usage():
    print('Usage: python convnet_trainer.py <input CSV> <output-folder>')

# def weight_variable(name, shape):
#     return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
# def conv_weight_variable(name, shape):
#     return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
# def bias_variable(name, shape):
#     return tf.Variable(tf.constant(1.0, shape=shape), name=name)
# def conv2d(X, W):
#     return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
# def max_pool(X):
#     return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#
# def input_conv_layer(name, input, kernel_size, input_channels, output_channels):
#     W_conv = conv_weight_variable("W_" + name, [kernel_size, kernel_size, input_channels, output_channels])
#     b_conv = bias_variable("b_" + name, [output_channels])
#     return tf.nn.relu(conv2d(input, W_conv) + b_conv)
# def conv_layer(name, input, kernel_size, input_channels, output_channels, drop_prob, pool=False):
#     W_conv = conv_weight_variable("W_" + name, [kernel_size, kernel_size, input_channels, output_channels])
#     b_conv = bias_variable("b_" + name, [output_channels])
#     layer = tf.nn.relu(conv2d(input, W_conv) + b_conv)
#     return tf.nn.dropout(max_pool(layer) if pool else layer, drop_prob)
# def flatten_layer(layer):
#     layer_shape = layer.get_shape()
#     nf = layer_shape[1:4].num_elements()
#     flattened_layer = tf.reshape(layer, [-1, nf])
#     return flattened_layer, nf
# def dense_layer(name, input, input_features, output_features, drop_prob):
#     W_fc = weight_variable("W_" + name, [input_features, output_features])
#     b_fc = bias_variable("b_" + name, [output_features])
#     return tf.nn.dropout(tf.nn.relu(tf.matmul(input, W_fc) + b_fc), drop_prob)
# def output_layer(name, input, input_features, output_features):
#     W_fc = weight_variable("W_" + name, [input_features, output_features])
#     b_fc = bias_variable("b_" + name, [output_features])
#     layer = tf.matmul(input, W_fc) + b_fc
#     return layer
#
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])
#
def sample_training(training_source, num_samples):
    sample = random.sample(training_source, num_samples)
    sampled_images = np.asarray([x[0] for x in sample])
    sampled_lengths = np.asarray([x[1] for x in sample])
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
    # training_image = training_image - np.mean(training_image)
    # Conver the label into a set of one-hot vectors.
    training_label = training_sample[1]
    length_label = np.zeros(num_length_classes)
    length_class = min(len(training_label), num_length_classes-1)
    length_label[length_class] = 1.0
    #digit_labels = [np.zeros(num_digit_classes) for _ in range(max_digits)]
    #for x in range(min(length_class, max_digits)):
    #    digit_labels[x][int(training_label[x])] = 1.0
    # Record normalized training samples.
    normalized_training_set.append((training_image.astype(np.float), length_class))

svhn_graph = tf.Graph()
with svhn_graph.as_default():
    # X_train is the training image.
    X_train = tf.placeholder(tf.float32, shape=(training_batch_size, image_size, image_size, input_channels), name="X_train")
    # y_train are the training labels, using sparse encoding.
    y_length = tf.placeholder(tf.int32, shape=training_batch_size, name="y_length")
    # Keep probability during dropout phase.
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Initialize convolutional shapes.
    conv1_shape = [conv_kernel_size, conv_kernel_size, input_channels, conv_depth1]
    conv2_shape = [conv_kernel_size, conv_kernel_size, conv_depth1, conv_depth2]
    conv3_shape = [conv_kernel_size, conv_kernel_size, conv_depth2, fc_size1]

    # Initialize convolutional weights and biases.
    W_conv1 = tf.get_variable("W_conv1", shape=conv1_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_conv1 = tf.Variable(tf.constant(1.0, shape=[conv_depth1]), name="b_conv1")
    W_conv2 = tf.get_variable("W_conv2", shape=conv2_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_conv2 = tf.Variable(tf.constant(1.0, shape=[conv_depth2]), name="b_conv2")
    W_conv3 = tf.get_variable("W_conv3", shape=conv3_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_conv3 = tf.Variable(tf.constant(1.0, shape=[fc_size1]), name="b_conv3")

    # Initialize output shapes.
    length_shape = [fc_size1, num_length_classes]

    # Initialize output weights and biases.
    W_length = tf.get_variable("W_length", shape=length_shape, initializer=tf.contrib.layers.xavier_initializer())
    b_length = tf.Variable(tf.constant(1.0, shape=[num_length_classes]), name="b_length")

    # Construct the CNN.
    conv1 = tf.nn.conv2d(X_train, W_conv1, [1,1,1,1], 'VALID', name="conv1")
    relu1 = tf.nn.relu(conv1 + b_conv1)
    norm1 = tf.nn.local_response_normalization(relu1) # Is this needed?
    pool1 = tf.nn.max_pool(norm1, [1,2,2,1], [1,2,2,1], 'SAME')
    conv2 = tf.nn.conv2d(pool1, W_conv2, [1,1,1,1], padding='VALID', name='conv2')
    relu2 = tf.nn.relu(conv2 + b_conv2)
    norm2 = tf.nn.local_response_normalization(relu2)
    pool2 = tf.nn.max_pool(norm2, [1,2,2,1], [1,2,2,1], 'SAME')
    conv3 = tf.nn.conv2d(pool2, W_conv3, [1,1,1,1], padding='VALID', name='conv3')
    relu3 = tf.nn.relu(conv3 + b_conv3)
    drop3 = tf.nn.dropout(relu3, keep_prob)
    drop3_shape = drop3.get_shape().as_list()
    flattened_layer = tf.reshape(drop3, [drop3_shape[0], drop3_shape[1] * drop3_shape[2] * drop3_shape[3]])
    output_length = tf.matmul(flattened_layer, W_length) + b_length

    # Initialize the optimizer
    training_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output_length, y_length))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(training_loss, global_step=global_step)

    # Initialize the final prediction.
    training_prediction = tf.nn.softmax(output_length)

num_training_epochs = 1000000
with tf.Session(graph=svhn_graph) as session:
    tf.initialize_all_variables().run()
    for i in range(num_training_epochs):
        batch = sample_training(normalized_training_set, training_batch_size)
        feed_dict = {X_train: batch[0], y_length: batch[1], keep_prob: 0.5}
        _, l, pred = session.run([optimizer, training_loss, training_prediction], feed_dict)
        if i%500 == 0:
            print('Minibatch loss at step %d: %f'%(i, l))
            print('Minibatch accuracy: %.1f%%'%accuracy(pred, batch[1]))
