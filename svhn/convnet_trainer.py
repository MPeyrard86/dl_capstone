"""
A tool used to train the convolutional neural network for my Udacity Machine Learning Nanodegree Capstone Project.
"""

import argparse
import datetime
import itertools
import multiprocessing
import os
import random
import time
from functools import partial

import numpy as np
import scipy as sp
import scipy.misc
import tensorflow as tf

from svhn import *
from svhn.data_loader import get_model_file

def sample_data(data, num_samples):
    """
    Acquires a random sample of data points from the given data set.
    :param data: The set of data to sample from.
    :param num_samples: The number of data points to sample.
    :return: A list of sampled data points.
    """
    sample = random.sample(data, num_samples)
    sampled_images = np.asarray([x[0] for x in sample])
    sampled_digits = np.asarray([x[1] for x in sample]).reshape((num_samples, MAX_DIGITS))
    return sampled_images, sampled_digits

def calculate_accuracy(predictions, correct_labels):
    """
    Calculates the accuracy of the given predictions by comparing them to the provided labels.
    The accuracy is calculated simply as the ratio of matching predictions to the total number of samples provided.
    :param predictions: The model's predictions.
    :param correct_labels: The correct labels from the data.
    :return: The ratio of matching predictions to the total number of samples provided.
    """
    predicted_labels = np.argmax(predictions, 2).transpose()
    num_correct_predictions = np.sum([np.array_equal(x,y) for x,y in zip(predicted_labels, correct_labels)])
    return float(num_correct_predictions) / correct_labels.shape[0]

def create_output_folder(output_dir_base):
    """
    Creates the model output folder. The output folder's name is based on the current date and time.
    :param output_dir_base: The base folder.
    :return: The full output folder location.
    """
    training_run_folder = os.path.join(output_dir_base, datetime.datetime.now().strftime('%Y%m%d-%H%M'))
    os.makedirs(training_run_folder)
    return training_run_folder

def validate_training_folders(training_folders):
    """
    Verifies that the provided training folders exist and contain a CSV file.
    If the conditions fail then a runtime exception is thrown.
    :param training_folders: The set of provided trianing folders.
    :return: Nothing.
    """
    for training_folder in training_folders:
        if not os.path.isdir(training_folder):
            raise RuntimeError("The provided training folder '%s' does not exist."%(training_folder))
        if not os.path.isfile(os.path.join(training_folder, CSVFILE)):
            raise RuntimeError("The provided training folder '%s' does not contain the training file descriptor '%s'."%(training_folder, CSVFILE))

def process_training_sample(folder, training_sample_line):
    """
    Performs processing of a single training sample line from a training CSV descriptor.
    :param training_sample_line: A single line from the training CSV descriptor.
    :return: A training sample tuple (training_image, length_class, digit_classes)
    """
    split_line = training_sample_line.strip().split(',')
    training_image_path = os.path.join(folder, split_line[0])
    training_image_label = split_line[1]
    training_image = sp.misc.imread(training_image_path)
    left = int(split_line[2])
    top = int(split_line[3])
    right = int(split_line[4])
    bottom = int(split_line[5])

    height = bottom - top
    delta_height = 0.5*(1.15 * height - height)
    top = int(max(top - delta_height, 0))
    bottom = int(max(bottom + delta_height, training_image.shape[0]))
    training_image = training_image[top:bottom, left:right]

    training_image = sp.misc.imresize(training_image, (IMAGE_SIZE, IMAGE_SIZE))
    length_class = min(len(training_image_label), NUM_LENGTH_CLASSES - 1)
    # Populate the digit classifications
    digit_classes = [int(x) for x in training_image_label]
    # Pad the ending of the digit with the empty label
    for _ in range(MAX_DIGITS-len(training_image_label)):
        digit_classes.append(NUM_DIGIT_CLASSES-1)
    return training_image, length_class, digit_classes

def load_training_data(training_folders):
    """
    Load the training data. Utilizes all available CPU cores to speed up the process.
    :param training_folders: The set of training folders that contains the training data.
    :return: A set of training samples.
    """
    validate_training_folders(training_folders)
    thread_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    training_data_master = list()
    for training_folder in training_folders:
        partial_process_sample = partial(process_training_sample, training_folder)
        with open(os.path.join(training_folder, CSVFILE), 'r') as training_file:
            training_data = thread_pool.map(partial_process_sample, itertools.islice(training_file, 1, None))
            training_data_master.append(training_data)
    thread_pool.close()
    flattend_training_data = filter(lambda y: len(y[2]) <= MAX_DIGITS, [training_sample for training_sublist in training_data_master for training_sample in training_sublist])
    del training_data
    random.shuffle(flattend_training_data)
    return flattend_training_data

if __name__ == '__main__':
    # Parse command line parameters
    parser = argparse.ArgumentParser(description="Performs training for a digit detector for the Stree-View House Numbers identification task.")
    parser.add_argument("-t", "--training-folders", required=True, nargs="+")
    parser.add_argument("-o", "--training-output", required=False)
    parser.add_argument("-b", "--batch-size", required=False, type=int, default=512)
    parser.add_argument("-v", "--validation-size", required=False, type=int, default=5000)
    parser.add_argument("-r", "--resume-from", required=False)
    args = parser.parse_args()

    if args.training_output is not None and not os.path.isdir(args.training_output):
        os.makedirs(args.training_output)

    # Load training data pointed to by command line parameters
    print "Loading training data from sources: %s"%args.training_folders
    print "This may take a few minutes depending on how much data you loaded and how many CPU cores you have."
    train_time_start = time.time()
    training_data = load_training_data(args.training_folders)

    images = np.asarray([x[0] for x in training_data], dtype=np.float32)
    labels = np.asarray([x[2] for x in training_data], dtype=np.float32)

    training_images = images[args.validation_size:]
    training_images_mean = np.mean(training_images)
    training_images -= training_images_mean
    training_labels = labels[args.validation_size:]
    training_set = zip(training_images, training_labels)
    validation_images = images[:args.validation_size]
    validation_images -= training_images_mean
    validation_labels = labels[:args.validation_size]
    validation_set = zip(validation_images, validation_labels)
    train_time_end = time.time()
    print "Loaded %d training samples in %.2fs."%(len(training_data), train_time_end - train_time_start)

    svhn_training_graph = tf.Graph()
    with svhn_training_graph.as_default():
        X_train = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLOR_CHANNELS), name="X_train")
        X_validation = tf.constant(validation_images, dtype=tf.float32, name="X_validation")
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
        W_conv5 = tf.get_variable("W_conv5", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV4_DEPTH, CONV5_DEPTH], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv5 = tf.Variable(tf.zeros([CONV5_DEPTH]), name="b_conv5")
        W_conv6 = tf.get_variable("W_conv6", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV5_DEPTH, CONV6_DEPTH], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv6 = tf.Variable(tf.zeros([CONV6_DEPTH]), name="b_conv6")
        W_conv7 = tf.get_variable("W_conv7", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV6_DEPTH, CONV7_DEPTH], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv7 = tf.Variable(tf.zeros([CONV7_DEPTH]), name="b_conv7")
        W_conv8 = tf.get_variable("W_conv8", shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV7_DEPTH, CONV8_DEPTH], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv8 = tf.Variable(tf.zeros([CONV8_DEPTH]), name="b_conv8")
        W_fc1 = tf.get_variable("W_fc1", shape=[((IMAGE_SIZE/4.0)**2)*CONV8_DEPTH, FC1_LENGTH], initializer=tf.contrib.layers.xavier_initializer())
        b_fc1 = tf.Variable(tf.zeros([FC1_LENGTH]), name="b_fc1")
        W_fc2 = tf.get_variable("W_fc2", shape=[FC1_LENGTH, FC2_LENGTH], initializer=tf.contrib.layers.xavier_initializer())
        b_fc2 = tf.Variable(tf.zeros([FC2_LENGTH]), name="b_fc2")

        # Create output weights and biases
        W_digit1 = tf.get_variable("W_digit1", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit1 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit1")
        W_digit2 = tf.get_variable("W_digit2", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit2 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit2")
        W_digit3 = tf.get_variable("W_digit3", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit3 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit3")
        W_digit4 = tf.get_variable("W_digit4", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit4 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit4")
        W_digit5 = tf.get_variable("W_digit5", shape=[FC2_LENGTH, NUM_DIGIT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        b_digit5 = tf.Variable(tf.zeros([NUM_DIGIT_CLASSES]), name="b_digit5")

        def create_model(input, keep_prob):
            # Create input layers
            conv_layer1 = tf.nn.relu(tf.nn.conv2d(input, W_conv1, [1,1,1,1], 'SAME') + b_conv1)
            conv_layer1 = tf.nn.local_response_normalization(conv_layer1)
            conv_layer1 = tf.nn.dropout(conv_layer1, keep_prob)
            conv_layer2 = tf.nn.relu(tf.nn.conv2d(conv_layer1, W_conv2, [1,1,1,1], 'SAME') + b_conv2)
            conv_layer2 = tf.nn.local_response_normalization(conv_layer2)
            conv_layer2 = tf.nn.dropout(conv_layer2, keep_prob)
            conv_layer3 = tf.nn.relu(tf.nn.conv2d(conv_layer2, W_conv3, [1,1,1,1], 'SAME') + b_conv3)
            conv_layer3 = tf.nn.local_response_normalization(conv_layer3)
            conv_layer3 = tf.nn.dropout(conv_layer3, keep_prob)
            conv_layer4 = tf.nn.relu(tf.nn.conv2d(conv_layer3, W_conv4, [1,1,1,1], 'SAME') + b_conv4)
            conv_layer4 = tf.nn.local_response_normalization(conv_layer4)
            conv_layer4 = tf.nn.max_pool(conv_layer4, [1,2,2,1], [1,2,2,1], 'SAME')
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
            flat_layer = tf.reshape(conv_layer8, [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]])
            fc_layer1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flat_layer, W_fc1) + b_fc1), keep_prob)
            fc_layer2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc_layer1, W_fc2) + b_fc2), keep_prob)
            # Create output layers
            digit1 = tf.matmul(fc_layer2, W_digit1) + b_digit1
            digit2 = tf.matmul(fc_layer2, W_digit2) + b_digit2
            digit3 = tf.matmul(fc_layer2, W_digit3) + b_digit3
            digit4 = tf.matmul(fc_layer2, W_digit4) + b_digit4
            digit5 = tf.matmul(fc_layer2, W_digit5) + b_digit5
            return digit1, digit2, digit3, digit4, digit5

        training_outputs = create_model(X_train, TRAINING_KEEP_PROB)
        training_loss = tf.add_n([tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(training_outputs[i], y_digits[:,i])) for i in range(len(training_outputs))])
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_EPOCHS, DECAY_RATE)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(training_loss, global_step=global_step)

        # For doing training predictions, create a model without any dropout.
        training_prediction_outputs = create_model(X_train, 1.0)
        training_prediction = tf.pack([tf.nn.softmax(training_prediction_outputs[i]) for i in range(len(training_prediction_outputs))])
        validation_model = create_model(X_validation, 1.0)
        validation_prediction = tf.pack([tf.nn.softmax(validation_model[i]) for i in range(len(validation_model))])

    # Record the training images mean in the output folder so that if we reload the model, we know how to adjust
    # the input.
    training_stats_folder = create_output_folder(args.training_output)
    mean_filename = os.path.join(training_stats_folder, TRAINING_IMAGES_MEAN)
    with open(mean_filename, 'w') as mean_file:
        mean_file.write(str(training_images_mean))

    training_stats_filename = os.path.join(training_stats_folder, TRAINING_STATS_FILE)
    checkpoint_filename = os.path.join(training_stats_folder, get_model_file(training_stats_folder))
    with open(training_stats_filename, 'w') as training_stats_file:
        training_stats_file.write("epoch,train_loss,train_acc,validation_acc\n")
        with tf.Session(graph=svhn_training_graph) as session:
            checkpoint_saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            best_validation_accuracy = float(0)
            for epoch in xrange(1, MAX_EPOCHS+1):
                training_batch = sample_data(training_set, args.batch_size)
                train_feed_dict = {X_train: training_batch[0], y_digits: training_batch[1]}
                _, t_loss, t_pred = session.run([optimizer, training_loss, training_prediction], train_feed_dict)
                if epoch%EPOCH_GROUP_SIZE == 0:
                    tacc = 100.0*calculate_accuracy(t_pred, training_batch[1])
                    vacc = 100.0*calculate_accuracy(validation_prediction.eval(), validation_labels)
                    training_stats_file.write("%d,%f,%f,%f\n" % (epoch, t_loss, tacc, vacc))
                    training_stats_file.flush()

                    print "training loss at step %d: %.2f"%(epoch, t_loss)
                    print "training accuracy: %.2f%%"%(tacc)
                    print "validation accuracy: %.2f%%"%(vacc)
                    if vacc > best_validation_accuracy:
                        best_validation_accuracy = vacc
                        print "Best validation accuracy seen so far. Checkpointing..."
                        checkpoint_saver.save(session, checkpoint_filename, global_step=global_step)
                    else:
                        print "Best so far is %.2f%%"%(best_validation_accuracy)
                    print