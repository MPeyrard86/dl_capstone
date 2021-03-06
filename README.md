# Street View House Numbers
Udacity Machine Learning Nanodegree capstone project.

## Introduction
This project was implemented in Python2 using Tensorflow 0.10. Later versions may introduce incompatabilities in the API and/or model formats. NumPy and SciPy are also required to run this project. It is also highly recommended that a reasonably powerful GPU be used for running the training program.

## Data Location
The data used and produced during this project was too large to upload to GitHub. Instead it has been uploaded to my Google Drive: https://drive.google.com/open?id=0B918sU9DDf8DTElNWGhwN1pTTXM

Two pre-trained models are provided, one called lessdata\_model and another called moredata\_model. The lessdata model was trained only using the SVHN training set. The moredata model was trained using both the training and extra sets from SVHN. 

All of the training data has also been uploaded under the training\_data folder. These data sets have been slightly modified. The hd5 file has been replaced by a CSV file. The program that performed this conversion is called h5train2csv, and is included in the GitHub repo. It will also be discussed later in this README.

The final data folder on the Drive is the custom\_testing\_data, which contains images that I personally collected that do not belong to any labelled data set. This was used to help test the neural networks built during this project.

## convnet\_trainer
The convnet\_trainer is the core of this project. It is the application that trains the convolutional neural network. It is meant to be run on the command line and requires the following parameters:

-t Training folders. Specify one or more training folders. The training folders should contain all training images and a CSV file containing the filename -> label mappings. Generating the CSV file from the hd5 files is done using the h5train2csv python script. 

-o Output folder. Specify a folder for which model checkpoints and other generated data should be placed. The output folder will contain checkpoint files, a file containing the training image mean, and a CSV file containing the training/validation statistics.

-b Batch size. The batch size to use during training.

-v Validation size. The size of the validation set to use during training.

## h5train2csv
This script is used to convert the h5 file to a CSV format. It is a very simple script that is meant to be run on the command line. Simply provide the _folder_ containing the h5 file as a parameter to the script and it will produce a CSV file (in the same folder) called digitStruct.csv. For example:

`python h5train2csv /home/matt/Downloads/extra`

The convnet\_trainer does not accept the h5 file, therefore running this script on the SVHN training data prior to training is _mandatory_. As previously stated, the uploaded data is already converted to CSV.

## test\_evaluation
This script was written to load a model and a labelled data set and evaluate its performance/accuracy. It requires two parameters:

-m Model folder location. This corresponds to the output (-o option) from the convnet\_trainer program. This is the model that will be loaded and evaluated.

-d Data folder. The labelled data to evaluate the model against.

Example:

`python test_evaluation.py -m /mnt/storage/capstone_stats/20161128-2101 -d /home/matt/Downloads/test`

## batch\_classify
This script was written to evaluate images that are not part of a labelled training/validation/testing set. It is the script that would be used in production. It takes in a batch of images and produces a CSV file containing the predictions for those images. It requires the following parameters:

-m Model folder location. This corresponds to the output (-o option) from the convnet\_trainer program. This is the model that will be loaded and used to make predictions.

-d Data folder. This folder should contain a set of images to evaluate.

-o Output folder. A file called predictions.csv will be written to this folder containing the predictions from the model for the given images in the data folder. 

Example:

`python batch_classify.py -m /mnt/storage/capstone_stats/20161128-2101 -d /home/matt/Downloads/custom -o /home/matt/Downloads/custom`
