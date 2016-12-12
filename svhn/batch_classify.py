import argparse
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

import scipy as sp
import scipy.misc

from svhn import *
from svhn.data_loader import load_data, load_image_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates a testing set.")
    parser.add_argument("-m", "--model-folder", required=True)
    parser.add_argument("-d", "--data-folder", required=True)
    parser.add_argument("-o", "--output-folder", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.model_folder):
        print 'Provided model folder does not exist.'
        sys.exit(1)
    training_mean_filename = os.path.join(args.model_folder, TRAINING_IMAGES_MEAN)
    if not os.path.isfile(training_mean_filename):
        print 'Model folder does not contain mean file.'
        sys.exit(1)

    if not os.path.isdir(args.data_folder):
        print 'Data folder does not exist.'
        sys.exit(1)

    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    training_mean = float(0)
    with open(training_mean_filename) as training_mean_file:
        training_mean = float(training_mean_file.readline())

    print "Loading data from sources: %s" % args.data_folder
    print "This may take a few minutes depending on how much data you loaded and how many CPU cores you have."
    train_time_start = time.time()
    image_files, image_data = load_image_data(args.data_folder)
    image_data = np.array(image_data) - training_mean

    train_time_end = time.time()
    print "Loaded %d samples in %.2fs." % (len(image_data), train_time_end - train_time_start)

    svhn_training_graph = tf.Graph()
    with svhn_training_graph.as_default():
        X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLOR_CHANNELS), name="X")
        y_digits = tf.placeholder(tf.int32, shape=(None, MAX_DIGITS))

        # Create convnet weights and biases
        W_conv1 = tf.get_variable("W_conv1",
                                  shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, IMAGE_COLOR_CHANNELS, CONV1_DEPTH],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv1 = tf.Variable(tf.zeros([CONV1_DEPTH]), name="b_conv1")
        W_conv2 = tf.get_variable("W_conv2", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV1_DEPTH, CONV2_DEPTH],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv2 = tf.Variable(tf.zeros([CONV2_DEPTH]), name="b_conv2")
        W_conv3 = tf.get_variable("W_conv3", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV2_DEPTH, CONV3_DEPTH],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv3 = tf.Variable(tf.zeros([CONV3_DEPTH]), name="b_conv3")
        W_conv4 = tf.get_variable("W_conv4", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV3_DEPTH, CONV4_DEPTH],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv4 = tf.Variable(tf.zeros([CONV4_DEPTH]), name="b_conv4")
        W_conv5 = tf.get_variable("W_conv5", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV4_DEPTH, CONV5_DEPTH],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv5 = tf.Variable(tf.zeros([CONV5_DEPTH]), name="b_conv5")
        W_conv6 = tf.get_variable("W_conv6", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV5_DEPTH, CONV6_DEPTH],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv6 = tf.Variable(tf.zeros([CONV6_DEPTH]), name="b_conv6")
        W_conv7 = tf.get_variable("W_conv7", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV6_DEPTH, CONV7_DEPTH],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv7 = tf.Variable(tf.zeros([CONV7_DEPTH]), name="b_conv7")
        W_conv8 = tf.get_variable("W_conv8", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV7_DEPTH, CONV8_DEPTH],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv8 = tf.Variable(tf.zeros([CONV8_DEPTH]), name="b_conv8")
        W_fc1 = tf.get_variable("W_fc1", shape=[((IMAGE_SIZE / 4.0) ** 2) * CONV8_DEPTH, FC1_LENGTH],
                                initializer=tf.contrib.layers.xavier_initializer())
        b_fc1 = tf.Variable(tf.zeros([FC1_LENGTH]), name="b_fc1")
        W_fc2 = tf.get_variable("W_fc2", shape=[FC1_LENGTH, FC2_LENGTH],
                                initializer=tf.contrib.layers.xavier_initializer())
        b_fc2 = tf.Variable(tf.zeros([FC2_LENGTH]), name="b_fc2")

        # Create output weights and biases
        W_digit1 = tf.get_variable("W_digit1", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES],
                                   initializer=tf.contrib.layers.xavier_initializer())
        b_digit1 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit1")
        W_digit2 = tf.get_variable("W_digit2", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES],
                                   initializer=tf.contrib.layers.xavier_initializer())
        b_digit2 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit2")
        W_digit3 = tf.get_variable("W_digit3", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES],
                                   initializer=tf.contrib.layers.xavier_initializer())
        b_digit3 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit3")
        W_digit4 = tf.get_variable("W_digit4", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES],
                                   initializer=tf.contrib.layers.xavier_initializer())
        b_digit4 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit4")
        W_digit5 = tf.get_variable("W_digit5", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES],
                                   initializer=tf.contrib.layers.xavier_initializer())
        b_digit5 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit5")

        def create_model(input, keep_prob):
            # Create input layers
            conv_layer1 = tf.nn.relu(tf.nn.conv2d(input, W_conv1, [1, 1, 1, 1], 'SAME') + b_conv1)
            conv_layer1 = tf.nn.local_response_normalization(conv_layer1)
            conv_layer1 = tf.nn.dropout(conv_layer1, keep_prob)
            conv_layer2 = tf.nn.relu(tf.nn.conv2d(conv_layer1, W_conv2, [1, 1, 1, 1], 'SAME') + b_conv2)
            conv_layer2 = tf.nn.local_response_normalization(conv_layer2)
            conv_layer2 = tf.nn.dropout(conv_layer2, keep_prob)
            conv_layer3 = tf.nn.relu(tf.nn.conv2d(conv_layer2, W_conv3, [1, 1, 1, 1], 'SAME') + b_conv3)
            conv_layer3 = tf.nn.local_response_normalization(conv_layer3)
            conv_layer3 = tf.nn.dropout(conv_layer3, keep_prob)
            conv_layer4 = tf.nn.relu(tf.nn.conv2d(conv_layer3, W_conv4, [1, 1, 1, 1], 'SAME') + b_conv4)
            conv_layer4 = tf.nn.local_response_normalization(conv_layer4)
            conv_layer4 = tf.nn.max_pool(conv_layer4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            conv_layer4 = tf.nn.dropout(conv_layer4, keep_prob)
            conv_layer5 = tf.nn.relu(tf.nn.conv2d(conv_layer4, W_conv5, [1, 1, 1, 1], 'SAME') + b_conv5)
            conv_layer5 = tf.nn.local_response_normalization(conv_layer5)
            conv_layer5 = tf.nn.dropout(conv_layer5, keep_prob)
            conv_layer6 = tf.nn.relu(tf.nn.conv2d(conv_layer5, W_conv6, [1, 1, 1, 1], 'SAME') + b_conv6)
            conv_layer6 = tf.nn.local_response_normalization(conv_layer6)
            conv_layer6 = tf.nn.dropout(conv_layer6, keep_prob)
            conv_layer7 = tf.nn.relu(tf.nn.conv2d(conv_layer6, W_conv7, [1, 1, 1, 1], 'SAME') + b_conv7)
            conv_layer7 = tf.nn.local_response_normalization(conv_layer7)
            conv_layer7 = tf.nn.dropout(conv_layer7, keep_prob)
            conv_layer8 = tf.nn.relu(tf.nn.conv2d(conv_layer7, W_conv8, [1, 1, 1, 1], 'SAME') + b_conv8)
            conv_layer8 = tf.nn.local_response_normalization(conv_layer8)
            conv_layer8 = tf.nn.max_pool(conv_layer8, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            conv_layer8 = tf.nn.dropout(conv_layer8, keep_prob)
            conv_shape = conv_layer8.get_shape().as_list()
            flat_layer = tf.reshape(conv_layer8, [-1, conv_shape[1] * conv_shape[2] * conv_shape[3]])
            fc_layer1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flat_layer, W_fc1) + b_fc1), keep_prob)
            fc_layer2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc_layer1, W_fc2) + b_fc2), keep_prob)
            # Create output layers
            digit1 = tf.matmul(fc_layer2, W_digit1) + b_digit1
            digit2 = tf.matmul(fc_layer2, W_digit2) + b_digit2
            digit3 = tf.matmul(fc_layer2, W_digit3) + b_digit3
            digit4 = tf.matmul(fc_layer2, W_digit4) + b_digit4
            digit5 = tf.matmul(fc_layer2, W_digit5) + b_digit5
            return digit1, digit2, digit3, digit4, digit5

    training_prediction_outputs = create_model(X, 1.0)
    training_prediction = tf.pack(
        [tf.nn.softmax(training_prediction_outputs[i]) for i in range(len(training_prediction_outputs))])

    checkpoint_filename = os.path.join(args.model_folder, 'model_checkpoint.chk-149900')
    with open(os.path.join(args.output_folder, BATCH_OUTPUT_FILE), 'w') as predictions_file:
        predictions_file.write("filename,prediction\n")
        with tf.Session(graph=svhn_training_graph) as session:
            checkpoint_saver = tf.train.Saver()
            checkpoint_saver.restore(session, checkpoint_filename)
            test_batch_size = 1000
            num_batches = int(math.ceil(len(image_data) / float(test_batch_size)))
            correct_predictions = 0
            total_predictions = 0
            for i in range(num_batches):
                lower_bound = i * test_batch_size
                upper_bound = (i + 1) * test_batch_size
                upper_bound = upper_bound if upper_bound < len(image_data) else len(image_data)
                fd = {X: image_data[lower_bound:upper_bound]}
                test_predictions = session.run([training_prediction], fd)
                predicted_labels = [x.flatten() for x in np.argmax(test_predictions, 3).transpose()]
                for i in range(lower_bound,upper_bound):
                    prediction = ''.join(str(x) for x in filter(lambda y: y < 10, predicted_labels[i-lower_bound]))
                    predictions_file.write("%s,%s\n"%(image_files[i], prediction))
                total_predictions += upper_bound - lower_bound

