import os
import sys

import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print 'The path to a training stats file is expected.'
    sys.exit(1)

training_stats_filename = sys.argv[1]
if not os.path.exists(training_stats_filename):
    print "The provided training stats file '%s' does not exist."%(training_stats_filename)
    sys.exit(2)

with open(training_stats_filename, 'r') as tf:
    raw_data = [x.split(',') for x in tf][1:]
    epochs = [x[0] for x in raw_data]
    loss = [x[1] for x in raw_data]
    tacc = [x[2] for x in raw_data]
    vacc = [x[3] for x in raw_data]

    plt.plot(epochs, loss, label='training loss')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(epochs, tacc, label='training accuracy')
    plt.plot(epochs, vacc, label='validation accuracy')
    plt.legend(loc='upper right')
    plt.show()