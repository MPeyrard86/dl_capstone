"""
Module that contains all the tools needed to load training data and train the conv net.
"""

__author__ = "Matthew Peyrard"
__version__ = 0.1

MATFILE = "digitStruct.mat"
CSVFILE = "digitStruct.csv"
CHECKPOINT_FILE = "svhn_model.ckpt"
TRAINING_STATS_FILE = "svhn_training_stats.csv"

# The percentage of imported data to use for training. The remaining is for validation.
TRAINING_RATIO = 0.9
# The number of epoch group runs without improvement before we decide to terminate.
END_EARLY_THRESHOLD = 500
# Number of epochs to accumulate before we re-evaluate.
EPOCH_GROUP_SIZE = 100
# Maximum number of epochs.
MAX_EPOCHS = 1000001