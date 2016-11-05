"""
A tool used to train the convolutional neural network for my Udacity Machine Learning Nanodegree Capstone Project.
"""

from __future__ import print_function

import argparse
import datetime
import sys
import time

import numpy as np

from detect_digits.model import *
from detect_digits.training import *
from detect_digits.training.loading import *

USAGE_MESSAGE = """Usage: python convnet_trainer.py <input-data-folders>
This script expects a set of training folders as parameters.
These folders are expected to contain the set of training images and a file called '%s'.
If your training folder does not contain such a file, then see the documentation for the h5train2csv tool to generate it."""%(CSVFILE)

def get_training_folder_names():
    if len(sys.argv) < 2:
        raise RuntimeError('Missing list of training folders.')
    return sys.argv[1:]

def sample_training(training_source, num_samples):
    sample = random.sample(training_source, num_samples)
    sampled_images = np.asarray([x[0] for x in sample])
    sampled_lengths = np.asarray([x[1] for x in sample])
    sampled_digits = np.asarray([x[2] for x in sample]).reshape((num_samples, MAX_DIGITS))
    return sampled_images, sampled_lengths, sampled_digits

def reformat_validation(vdata):
    vimages = np.asarray([x[0] for x in vdata])
    vdigits = np.asarray([x[2] for x in vdata])
    return vimages, vdigits

def calculate_accuracy(y_pred, y_labels):
    predicted_labels = np.argmax(y_pred, 2).transpose()
    num_correct_predictions = np.sum([np.array_equal(x,y) for x,y in zip(predicted_labels, y_labels)])
    return float(num_correct_predictions)/y_labels.shape[0]

if __name__ == '__main__':
    # Parse command line parameters
    parser = argparse.ArgumentParser(description="Performs training for a digit detector for the Stree-View House Numbers identification task.")
    parser.add_argument("-t", "--training-folders", required=True, nargs="+")
    parser.add_argument("-o", "--training-output", required=True)
    parser.add_argument("-b", "--batch-size", required=False, type=int, default=512)
    args = parser.parse_args()

    if not os.path.isdir(args.training_output):
        os.makedirs(args.training_output)

    # Load training data pointed to by command line parameters
    print("Loading training data from sources: %s"%args.training_folders)
    print("This may take a few minutes depending on how much data you loaded and how many CPU cores you have.")
    train_time_start = time.time()
    training_data = load_training_data(args.training_folders)
    train_validation_split_point = int(TRAINING_RATIO * len(training_data))
    train_data = training_data[0:train_validation_split_point]
    validation_data = training_data[train_validation_split_point:]
    v_images, v_digits = reformat_validation(validation_data)
    train_time_end = time.time()
    print("Loaded %d training samples in %fs."%(len(training_data), train_time_end - train_time_start))

    svhn_training_graph = tf.Graph()
    with svhn_training_graph.as_default():
        X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLOR_CHANNELS), name="X")
        X_validation = tf.constant(v_images, dtype=tf.float32)
        y_digits = tf.placeholder(tf.int32, shape=(None, MAX_DIGITS))

        # Create convnet weights and biases
        W_conv1 = tf.get_variable("W_conv1", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, IMAGE_COLOR_CHANNELS, CONV1_DEPTH], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv1 = tf.Variable(tf.zeros([CONV1_DEPTH]), name="b_conv1")
        W_conv2 = tf.get_variable("W_conv2", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV1_DEPTH, CONV2_DEPTH], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv2 = tf.Variable(tf.zeros([CONV2_DEPTH]), name="b_conv2")
        W_conv3 = tf.get_variable("W_conv3", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV2_DEPTH, CONV3_DEPTH], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv3 = tf.Variable(tf.zeros([CONV3_DEPTH]), name="b_conv3")
        W_conv4 = tf.get_variable("W_conv4", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV3_DEPTH, CONV4_DEPTH], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv4 = tf.Variable(tf.zeros([CONV4_DEPTH]), name="b_conv4")
        W_fc1 = tf.get_variable("W_fc1", shape=[((IMAGE_SIZE/4.0)**2)*CONV4_DEPTH, FC1_LENGTH], initializer=tf.contrib.layers.xavier_initializer())
        b_fc1 = tf.Variable(tf.zeros([FC1_LENGTH]), name="b_fc1")
        W_fc2 = tf.get_variable("W_fc2", shape=[FC1_LENGTH, FC2_LENGTH], initializer=tf.contrib.layers.xavier_initializer())
        b_fc2 = tf.Variable(tf.zeros([FC2_LENGTH]))

        # Create output weights and biases
        W_digit1 = tf.get_variable("W_digit1", shape=[FC1_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit1 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit1")
        W_digit2 = tf.get_variable("W_digit2", shape=[FC1_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit2 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit2")
        W_digit3 = tf.get_variable("W_digit3", shape=[FC1_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit3 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit3")
        W_digit4 = tf.get_variable("W_digit4", shape=[FC1_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit4 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit4")
        W_digit5 = tf.get_variable("W_digit5", shape=[FC1_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit5 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit5")

        def create_model(input, keep_prob):
            # Create input layers
            conv_layer1 = tf.nn.relu(tf.nn.conv2d(input, W_conv1, [1,1,1,1], 'SAME') + b_conv1)
            conv_layer1 = tf.nn.local_response_normalization(conv_layer1)
            conv_layer1 = tf.nn.dropout(conv_layer1, keep_prob)
            conv_layer2 = tf.nn.relu(tf.nn.conv2d(conv_layer1, W_conv2, [1,1,1,1], 'SAME') + b_conv2)
            conv_layer2 = tf.nn.local_response_normalization(conv_layer2)
            conv_layer2 = tf.nn.max_pool(conv_layer2, [1,2,2,1], [1,2,2,1], 'SAME')
            conv_layer3 = tf.nn.relu(tf.nn.conv2d(conv_layer2, W_conv3, [1,1,1,1], 'SAME') + b_conv3)
            conv_layer3 = tf.nn.local_response_normalization(conv_layer3)
            conv_layer3 = tf.nn.dropout(conv_layer3, keep_prob)
            conv_layer4 = tf.nn.relu(tf.nn.conv2d(conv_layer3, W_conv4, [1,1,1,1], 'SAME') + b_conv4)
            conv_layer4 = tf.nn.local_response_normalization(conv_layer4)
            conv_layer4 = tf.nn.max_pool(conv_layer4, [1,2,2,1], [1,2,2,1], 'SAME')
            conv_shape = conv_layer4.get_shape().as_list()
            flat_layer = tf.reshape(conv_layer4, [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]])
            fc_layer1 = tf.nn.relu(tf.matmul(flat_layer, W_fc1) + b_fc1)
            # Create output layers
            digit1 = tf.matmul(fc_layer1, W_digit1) + b_digit1
            digit2 = tf.matmul(fc_layer1, W_digit2) + b_digit2
            digit3 = tf.matmul(fc_layer1, W_digit3) + b_digit3
            digit4 = tf.matmul(fc_layer1, W_digit4) + b_digit4
            digit5 = tf.matmul(fc_layer1, W_digit5) + b_digit5
            return digit1, digit2, digit3, digit4, digit5

        training_outputs = create_model(X, 0.5)
        training_loss = tf.add_n([tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(training_outputs[i], y_digits[:,i])) for i in range(len(training_outputs))])
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.025, global_step, 20000, 0.95)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(training_loss, global_step=global_step)
        # optimizer = tf.train.AdamOptimizer(0.01).minimize(training_loss)

        # For doing training predictions, create a model without any dropout.
        training_prediction_outputs = create_model(X, 1.0)
        training_prediction = tf.pack([tf.nn.softmax(training_prediction_outputs[i]) for i in range(len(training_prediction_outputs))])
        validation_model = create_model(X_validation, 1.0)
        validation_prediction = tf.pack([tf.nn.softmax(validation_model[i]) for i in range(len(validation_model))])

    training_stats_filename = os.path.join(args.training_output,  datetime.datetime.now().strftime('%Y-%m-%d-%H-%M.csv'))
    with open(training_stats_filename, 'w') as training_stats_file:
        training_stats_file.write("epoch,train_loss,train_acc,validation_acc")
        with tf.Session(graph=svhn_training_graph) as session:
            tf.initialize_all_variables().run()
            for epoch in xrange(1, 10000000+1):
                training_batch = sample_training(train_data, args.batch_size)
                # validation_batch = sample_training(validation_data, args.batch_size)
                train_feed_dict = {X: training_batch[0], y_digits: training_batch[2]}
                # validation_feed_dict = {X: validation_batch[0]}
                _, l, train_pred = session.run([optimizer, training_loss, training_prediction], train_feed_dict)
                # validation_pred = session.run([training_prediction], validation_feed_dict)
                if epoch%100 == 0:
                    tacc = 100.0*calculate_accuracy(train_pred, training_batch[2])
                    vacc = 100.0*calculate_accuracy(validation_prediction.eval(), v_digits)
                    training_stats_file.write("%d,%f,%f,%f\n"%(epoch, l, tacc, vacc))
                    training_stats_file.flush()

                    print("training loss at step %d: %f"%(epoch, l))
                    print("training accuracy: %f%%"%(tacc))
                    print("validation accuracy: %f%%"%(vacc))
                    print()
