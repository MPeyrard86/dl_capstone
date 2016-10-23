"""

"""

import numpy as np
import tensorflow as tf

from detect_digits import *

def create_conv_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
def create_fc_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
def create_bias(name, shape):
    return tf.Variable(tf.constant(1.0, shape=shape), name=name)

def create_conv_layer(input, weights, biases):
    conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], 'SAME')
    relu = tf.nn.relu(conv + biases)
    norm = tf.nn.local_response_normalization(relu)
    return norm
def create_fc_layer(input, weights, biases):
    return tf.matmul(input, weights) + biases
def create_dropout_layer(input, dropout_keep_prob):
    return tf.nn.dropout(input, dropout_keep_prob)
def create_maxpool_layer(input):
    return tf.nn.max_pool(input, [1,2,2,1], [1,2,2,1], 'SAME')
def flatten_conv_layer(input):
    input_shape = input.get_shape().as_list()
    return tf.reshape(input, shape=[input_shape[0], input_shape[1]*input_shape[2]*input_shape[3]])

def create_model(X, dropout_keep_prob, reuse):
    """

    :param X:
    :param dropout_keep_prob:
    :return:
    """
    with tf.variable_scope("svhn", reuse=reuse) as scope:
        # Initialize weights and biases.
        W_conv1 = create_conv_variable("W_conv1", CONV1_SHAPE)
        b_conv1 = create_bias("b_conv1", [CONV1_DEPTH])
        W_conv2 = create_conv_variable("W_conv2", CONV2_SHAPE)
        b_conv2 = create_bias("b_conv2", [CONV2_DEPTH])
        W_conv3 = create_conv_variable("W_conv3", CONV3_SHAPE)
        b_conv3 = create_bias("b_conv3", [CONV3_DEPTH])
        W_conv4 = create_conv_variable("W_conv4", CONV4_SHAPE)
        b_conv4 = create_bias("b_conv4", [CONV4_DEPTH])
        W_conv5 = create_conv_variable("W_conv5", CONV5_SHAPE)
        b_conv5 = create_bias("b_conv5", [CONV5_DEPTH])
        W_conv6 = create_conv_variable("W_conv6", CONV6_SHAPE)
        b_conv6 = create_bias("b_conv6", [CONV6_DEPTH])
        W_conv7 = create_conv_variable("W_conv7", CONV7_SHAPE)
        b_conv7 = create_bias("b_conv7", [CONV7_DEPTH])
        W_conv8 = create_conv_variable("W_conv8", CONV8_SHAPE)
        b_conv8 = create_bias("b_conv8", [CONV8_DEPTH])
        W_fc = create_fc_variable("W_fc", FC_SHAPE)
        b_fc = create_bias("b_fc", [FC_LENGTH])

        # Output layer weights and biases.
        W_length = create_fc_variable("W_length", [FC_LENGTH, NUM_LENGTH_CLASSES])
        b_length = create_bias("b_length", [NUM_LENGTH_CLASSES])
        # TODO: Add outputs for each digit.

        # Create the convolutional neural network
        conv1 = create_dropout_layer(create_conv_layer(X, W_conv1, b_conv1), dropout_keep_prob)
        conv2 = create_dropout_layer(create_conv_layer(conv1, W_conv2, b_conv2), dropout_keep_prob)
        conv3 = create_maxpool_layer(create_conv_layer(conv2, W_conv3, b_conv3))
        conv4 = create_dropout_layer(create_conv_layer(conv3, W_conv4, b_conv4), dropout_keep_prob)
        conv5 = create_dropout_layer(create_conv_layer(conv4, W_conv5, b_conv5), dropout_keep_prob)
        conv6 = create_maxpool_layer(create_conv_layer(conv5, W_conv6, b_conv6))
        conv7 = create_dropout_layer(create_conv_layer(conv6, W_conv7, b_conv7), dropout_keep_prob)
        conv8 = create_dropout_layer(create_conv_layer(conv7, W_conv8, b_conv8), dropout_keep_prob)
        conv8_flattend = flatten_conv_layer(conv8)
        feature_layer = create_fc_layer(conv8_flattend, W_fc, b_fc)
        # Create output layers
        length_output = create_fc_layer(feature_layer, W_length, b_length)
        return length_output
