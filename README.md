# Drowsiness-Detection-using-Deep-Learning
CSCE 689 Project

## Project consists of three ipython files for detecting yawning.

1.Feature Extraction.ipython:  Coded to prepare the training data. It builds feature files from the videos used for training. Training videos are labelled from 1.avi to 41.avi which are stored to my google drive folder. It outputs:
'Data.npy': Complete landmark information from all videos
'Timestamps.csv': time indices of all frames extracted from the video
"Features_final.csv": Stores dataframe of the five features computed using formulation mentioned in the report for all frames
"Labels.csv": Label for each frame for the video number from which it was extracted
"combinedlabels.csv": Label and timeframe concatanated together.

2.Model.ipython: Takes input 'Features_final.csv' and 'Labels_final.csv'. The first file is what the Feature Extraction.ipython dumped. The second file is essentially 'combinedlabels.csv' but has one more column consisting of the label whether the action in the frame is drowsiness (1 for drowsiness and 0 otherwise). This ipython file shapes the data to be fed to neural network and then tunes it's parameters. It outputs trained model (model.h5) and plots indicating training and validation performance.

3.test.ipython: This can be used for testing the model exported from Model.ipython. It publishes the time-label graph as "Test_g.png" and the data as "timeLabel_testg.json" where g is a number you can put at the begining to identify the files. It takes input the video in the avi format and you can use the custom name of your video along after you mention g value.
