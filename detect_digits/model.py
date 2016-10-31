"""

"""

import tensorflow as tf

from detect_digits import *

def create_convolutional_weight_tensor(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
def create_convolutional_bias_tensor(name, shape):
    return tf.Variable(tf.constant(1.0, shape=shape), name=name)

def flatten_convolutional_layer(input):
    input_shape = input.get_shape().as_list()
    return tf.reshape(input, [-1, input_shape[1] * input_shape[2] * input_shape[3]])
def create_convolutional_layer(input, weights, biases, keep_prob):
    layer = tf.nn.conv2d(input, weights, [1,1,1,1], 'SAME')
    # layer = tf.nn.batch_normalization(layer) TODO: Play around with this later...
    layer = tf.nn.local_response_normalization(layer)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = tf.nn.relu(layer + biases)
    layer = tf.nn.max_pool(layer, [1,2,2,1], [1,2,2,1], 'SAME')
    return layer
def create_fully_connected_layer(input, weights, biases, keep_prob):
    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.dropout(layer, keep_prob)
    return layer
def create_output_layer(input, weights, biases):
    return tf.matmul(input, weights) + biases

def create_convolutional_network(X, keep_prob):
    with tf.variable_scope("svhn") as scope:
        # Input and hidden layer weights.
        W_conv1 = create_convolutional_weight_tensor('W_conv1', CONV1_SHAPE)
        b_conv1 = create_convolutional_bias_tensor('b_conv1', [CONV1_DEPTH])
        W_conv2 = create_convolutional_weight_tensor('W_conv2', CONV2_SHAPE)
        b_conv2 = create_convolutional_bias_tensor('b_conv2', [CONV2_DEPTH])
        W_conv3 = create_convolutional_weight_tensor('W_conv3', CONV3_SHAPE)
        b_conv3 = create_convolutional_bias_tensor('b_conv3', [CONV3_DEPTH])
        W_fc1 = create_convolutional_weight_tensor('W_fc1', FC1_SHAPE)
        b_fc1 = create_convolutional_bias_tensor('b_fc1', [FC1_LENGTH])
        W_fc2 = create_convolutional_weight_tensor('W_fc2', FC2_SHAPE)
        b_fc2 = create_convolutional_bias_tensor('b_fc2', [FC2_LENGTH])

        # Output layer weights and biases.
        W_digit1 = create_convolutional_weight_tensor("W_digit1", DIGIT_OUTPUT_SHAPE)
        b_digit1 = create_convolutional_bias_tensor("b_digit1", [NUM_DIGIT_CLASSES])
        W_digit2 = create_convolutional_weight_tensor("W_digit2", DIGIT_OUTPUT_SHAPE)
        b_digit2 = create_convolutional_bias_tensor("b_digit2", [NUM_DIGIT_CLASSES])
        W_digit3 = create_convolutional_weight_tensor("W_digit3", DIGIT_OUTPUT_SHAPE)
        b_digit3 = create_convolutional_bias_tensor("b_digit3", [NUM_DIGIT_CLASSES])
        W_digit4 = create_convolutional_weight_tensor("W_digit4", DIGIT_OUTPUT_SHAPE)
        b_digit4 = create_convolutional_bias_tensor("b_digit4", [NUM_DIGIT_CLASSES])
        W_digit5 = create_convolutional_weight_tensor("W_digit5", DIGIT_OUTPUT_SHAPE)
        b_digit5 = create_convolutional_bias_tensor("b_digit5", [NUM_DIGIT_CLASSES])

        # Create the convolutional network
        conv_layer1 = create_convolutional_layer(X, W_conv1, b_conv1, keep_prob)
        conv_layer2 = create_convolutional_layer(conv_layer1, W_conv2, b_conv2, keep_prob)
        conv_layer3 = create_convolutional_layer(conv_layer2, W_conv3, b_conv3, keep_prob)
        flattened_transitional_layer = flatten_convolutional_layer(conv_layer3)
        fully_connected_layer1 = create_fully_connected_layer(flattened_transitional_layer, W_fc1, b_fc1, keep_prob)
        fully_connected_layer2 = create_fully_connected_layer(fully_connected_layer1, W_fc2, b_fc2, keep_prob)

        # Define the output layers
        digit_output1 = create_output_layer(fully_connected_layer2, W_digit1, b_digit1)
        digit_output2 = create_output_layer(fully_connected_layer2, W_digit2, b_digit2)
        digit_output3 = create_output_layer(fully_connected_layer2, W_digit3, b_digit3)
        digit_output4 = create_output_layer(fully_connected_layer2, W_digit4, b_digit4)
        digit_output5 = create_output_layer(fully_connected_layer2, W_digit5, b_digit5)

        return digit_output1, digit_output2, digit_output3, digit_output4, digit_output5
