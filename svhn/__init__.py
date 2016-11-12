"""
A set of constants that describes the convolutional neural network.
"""

__author__ = 'Matthew Peyrard'
__version__ = 0.1

# Images are normalized to 32x32x3
IMAGE_SIZE = 32
IMAGE_COLOR_CHANNELS = 3

MAX_DIGITS = 5
# The digit may be of discrete length 0-5, and we also add another class for 6+.
NUM_LENGTH_CLASSES = 7
# Each digit may be 0-9, and we also add another class to represent 'not present'.
NUM_DIGIT_CLASSES = 11
EMPTY_DIGIT_LABEL = NUM_DIGIT_CLASSES-1

# Convolutional layers hyper-parameters
MAXPOOL_KERNEL_SIZE = 2
CONV_KERNEL_SIZE = 5

# Layer sizes
CONV1_DEPTH = 48
CONV2_DEPTH = 64
CONV3_DEPTH = 128
CONV4_DEPTH = 160
CONV5_DEPTH = 192
CONV6_DEPTH = 192
CONV7_DEPTH = 192
CONV8_DEPTH = 192
FC1_LENGTH = 4096
FC2_LENGTH = 4096

# Learning Rates
INITIAL_LEARNING_RATE = 3e-2
FINAL_LEARNING_RATE = 5e-3
DECAY_EPOCHS = 5e4
DECAY_RATE = (FINAL_LEARNING_RATE/INITIAL_LEARNING_RATE)**(1.0/DECAY_EPOCHS)

MATFILE = "digitStruct.mat"
CSVFILE = "digitStruct.csv"
CHECKPOINT_FILE = "svhn_model.ckpt"
TRAINING_STATS_FILE = "svhn_training_stats.csv"

# The number of epoch group runs without improvement before we decide to terminate.
END_EARLY_THRESHOLD = 500
# Number of epochs to accumulate before we re-evaluate.
EPOCH_GROUP_SIZE = 100
# Maximum number of epochs.
MAX_EPOCHS = 1000000