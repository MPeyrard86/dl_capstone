"""
A tool used to train the convolutional neural network for my Udacity Machine Learning Nanodegree Capstone Project.
"""

from __future__ import print_function

import sys
import time

import numpy as np
import tensorflow as tf

from detect_digits import *
from detect_digits.model import create_model
from training import TRAINING_RATIO
from training.loading import *

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
    return sampled_images, sampled_lengths

def calculate_accuracy(y_pred, y_labels):
    num_correct_predictions = np.sum(np.argmax(y_pred, axis=1) == y_labels)
    return float(num_correct_predictions)/y_labels.shape[0]

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

if __name__ == '__main__':
    try:
        training_batch_size = 512

        # Parse command line parameters
        training_folders = get_training_folder_names()

        # Load training data pointed to by command line parameters
        print("Loading training data from sources: %s"%training_folders)
        print("This may take a few minutes depending on how much data you loaded and how many CPU cores you have.")
        train_time_start = time.time()
        training_data = load_training_data(training_folders)
        train_validation_split_point = int(TRAINING_RATIO * len(training_data))
        train_data = training_data[0:train_validation_split_point]
        validation_data = training_data[train_validation_split_point:]
        train_time_end = time.time()
        print("Loaded %d training samples in %fs."%(len(training_data), train_time_end - train_time_start))

        # Create the conv net
        svhn_training_graph = tf.Graph()
        with svhn_training_graph.as_default():
            X_training_image = tf.placeholder(tf.float32, shape=(training_batch_size, IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLOR_CHANNELS), name="X_training_image")
            X_validation_image = tf.placeholder(tf.float32, shape=(training_batch_size, IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLOR_CHANNELS), name="X_validation_image")
            y_length = tf.placeholder(tf.int32, shape=training_batch_size, name="y_length")
            dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            training_output_length = create_model(X_training_image, dropout_keep_prob, False)
            validation_output_length = create_model(X_validation_image, dropout_keep_prob, True)
            # Set up the training process
            length_training_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(training_output_length, y_length))
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
            length_training_optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(length_training_loss, global_step=global_step)

            length_training_prediction = tf.nn.softmax(training_output_length)
            length_validation_prediction = tf.nn.softmax(validation_output_length)

        num_training_epochs = 1000000
        with tf.Session(graph=svhn_training_graph) as session:
            tf.initialize_all_variables().run()
            for i in xrange(num_training_epochs):
                training_batch = sample_training(train_data, training_batch_size)
                validation_batch = sample_training(validation_data, training_batch_size)
                training_feed_dict = {X_training_image: training_batch[0],
                                      y_length: training_batch[1],
                                      dropout_keep_prob: 0.5}
                validation_feed_dict = {X_validation_image: validation_batch[0],
                                        y_length: validation_batch[1],
                                        dropout_keep_prob: 1.0}
                _, _train_loss, _train_predition = session.run([length_training_optimizer, length_training_loss, length_training_prediction],
                                                               training_feed_dict)
                _validation_prediction = session.run([length_validation_prediction], validation_feed_dict)
                if i%2 == 0:
                    acc = calculate_accuracy(_train_predition, training_batch[1])
                    print("Training step %d."%(i))
                    print("Batch training loss: %f."%(_train_loss))
                    print("Training accuracy: %f, Cross-validation accuracy: %f."%(
                        calculate_accuracy(_train_predition, training_batch[1]),
                        calculate_accuracy(_validation_prediction, validation_batch[1])))
                    print()

    except:
        raise
    # except RuntimeError as x:
    #     print('RuntimeError: ' + x.message)
    #     print(USAGE_MESSAGE)
    #     sys.exit(1)
    # except Exception as x:
    #     print(x)
    #     print('Unknown error. Terminating.')
    #     sys.exit(2)
