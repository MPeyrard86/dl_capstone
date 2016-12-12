# dl\_capstone Street View House Numbers
Udacity Machine Learning Nanodegree capstone project.

## Introduction
This project was implemented in Python2 using Tensorflow 0.10. Later versions may introduce incompatabilities in the API and/or model formats.

## Data Location
The data used and produced during this project was too large to upload to GitHub. Instead it has been uploaded to my Google Drive: https://drive.google.com/open?id=0B918sU9DDf8DTElNWGhwN1pTTXM

Two pre-trained models are provided, one called lessdata\_model and another called moredata\_model. The lessdata model was trained only using the SVHN training set. The moredata model was trained using both the training and extra sets from SVHN. 

All of the training data has also been uploaded under the training\_data folder. These data sets have been slightly modified. The hd5 file has been replaced by a CSV file. The program that performed this conversion is called h5train2csv, and is included in the GitHub repo. It will also be discussed later in this README.

The final data folder on the Drive is the custom\_testing\_data, which contains images that I personally collected that do not belong to any labelled data set. This was used to help test the neural networks built during this project.


