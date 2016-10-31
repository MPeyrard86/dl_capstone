"""
A tool used to train the convolutional neural network for my Udacity Machine Learning Nanodegree Capstone Project.
"""

from __future__ import print_function

import argparse
import sys
import time

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

def calculate_accuracy(y_pred, y_labels):
    predicted_labels = np.argmax(y_pred, 2).transpose()
    __p = [np.array_equal(x,y) for x,y in zip(predicted_labels, y_labels)]
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
    train_time_end = time.time()
    print("Loaded %d training samples in %fs."%(len(training_data), train_time_end - train_time_start))

    # Create the conv net
    svhn_training_graph = tf.Graph()
    with svhn_training_graph.as_default():
        X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLOR_CHANNELS), name="X")
        y_digits = tf.placeholder(tf.int32, shape=(None, MAX_DIGITS))
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # output_length = create_model(X, dropout_keep_prob)
        digit1, digit2, digit3, digit4, digit5 = create_convolutional_network(X, dropout_keep_prob)
        # Set up the training process
        digit1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(digit1, y_digits[:, 0]))
        digit2_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(digit2, y_digits[:, 1]))
        digit3_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(digit3, y_digits[:, 2]))
        digit4_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(digit4, y_digits[:, 3]))
        digit5_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(digit5, y_digits[:, 4]))
        full_digit_loss = digit1_loss + digit2_loss + digit3_loss + digit4_loss + digit5_loss

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(full_digit_loss, global_step=global_step)

        # Define predictions
        digit1_prediction = tf.nn.softmax(digit1)
        digit2_prediction = tf.nn.softmax(digit2)
        digit3_prediction = tf.nn.softmax(digit3)
        digit4_prediction = tf.nn.softmax(digit4)
        digit5_prediction = tf.nn.softmax(digit5)
        full_digit_prediction = tf.pack([digit1_prediction, digit2_prediction, digit3_prediction, digit4_prediction, digit5_prediction])

    # Number of epochs we can go without improvement before we simply give up.
    with tf.Session(graph=svhn_training_graph) as session:
        saver = tf.train.Saver()
        tf.initialize_all_variables().run()

        epochs_without_improvement = 0
        best_validation_accuracy = 0

        # Stats tracking variables
        epoch_groups_without_improvement = 0
        best_average_validation_accuracy = 0
        training_accuracy_sum = 0
        training_loss_sum = 0
        validation_accuracy_sum = 0
        validation_loss_sum = 0
        # Set up checkpointing and stats tracking files
        checkpoint_filename = os.path.join(args.training_output, CHECKPOINT_FILE)
        training_stats_filename = os.path.join(args.training_output, TRAINING_STATS_FILE)
        with open(training_stats_filename, 'w') as training_stats_file:
            training_stats_file.write("epoch,train_loss,train_acc,validation_loss,validation_acc\n")
            for epoch in xrange(MAX_EPOCHS):
                # Randomly sample the data for training and validation runs
                training_batch = sample_training(train_data, args.batch_size)
                validation_batch = sample_training(validation_data, args.batch_size)
                training_feed_dict = {X: training_batch[0], y_digits: training_batch[2],dropout_keep_prob: 0.5}
                validation_feed_dict = {X: validation_batch[0], y_digits: validation_batch[2],dropout_keep_prob: 1.0}
                _, training_loss, training_prediction = session.run(
                    [optimizer, full_digit_loss, full_digit_prediction], training_feed_dict)
                validation_loss, validation_prediction = session.run(
                    [full_digit_loss, full_digit_prediction], validation_feed_dict)

                training_accuracy = calculate_accuracy(training_prediction, training_batch[2])
                training_accuracy_sum += training_accuracy
                training_loss_sum += training_loss
                validation_accuracy = calculate_accuracy(validation_prediction, validation_batch[2])
                validation_accuracy_sum += validation_accuracy
                validation_loss_sum += validation_loss

                if epoch != 0 and epoch%EPOCH_GROUP_SIZE == 0:
                    training_accuracy_average = training_accuracy_sum/EPOCH_GROUP_SIZE
                    training_loss_average = training_loss_sum/EPOCH_GROUP_SIZE
                    validation_accuracy_average = validation_accuracy_sum/EPOCH_GROUP_SIZE
                    validation_loss_average = validation_loss_sum/EPOCH_GROUP_SIZE
                    # Log stats to training stats file
                    training_stats_file.write("%d,%f,%f,%f,%f\n"%(
                        epoch, training_loss_average, training_accuracy_average,
                        validation_loss_average, validation_accuracy_average))
                    # Check stats and checkpoint if we have an overall improvement
                    if validation_accuracy_average > best_average_validation_accuracy:
                        best_average_validation_accuracy = validation_accuracy_average
                        print("Average validation accuracy after %d epochs is best seen so far, checkpointing..."%epoch)
                        saver.save(session, checkpoint_filename, global_step=global_step)
                        epoch_groups_without_improvement = 0
                    else:
                        epoch_groups_without_improvement += 1
                    # Report stats to stdout
                    print("Train/Validation accuracies: %f/%f" % (training_accuracy_average, validation_accuracy_average))
                    print("%f %f" % (training_loss_average, validation_loss_average))
                    print()
                    # Terminate early if we haven't improved in long enouch
                    # if epoch_groups_without_improvement >= END_EARLY_THRESHOLD:
                    #     print("No improvement in %d epochs. Terminating."%(epoch_groups_without_improvement*EPOCH_GROUP_SIZE))
                    #     break
                    # Reset epoch group state variables
                    training_accuracy_sum = 0
                    training_loss_sum = 0
                    validation_accuracy_sum = 0
                    validation_loss_sum = 0
