# Third Submission
##### 1. Feature Extraction.ipython: 
Coded to prepare the training data. It builds feature files from the videos used for training. Training videos are labelled from 0.avi to 58.avi which are stored to my google drive folder. It outputs:
'Data.npy': Complete landmark information from all videos
'Timestamps.csv': time indices of all frames extracted from the video
"Features_final.csv": Stores dataframe of the five features computed using formulation mentioned in the report for all frames
"Labels.csv": Label for each frame for the video number from which it was extracted
"combinedlabels.csv": Label and timeframe concatanated together.
Train images with their classes (For transfer learning)
Validation images with their classes(For transfer learning)
Prediction train and test images (Validation images without class separation to get prediction labels for transfer learning)

##### 2. Model.ipython: 
Takes input 'Features_final.csv' and 'Labels_final.csv'. The first file is what the Feature Extraction.ipython dumped along with the training and testing input output data files.
The second file is essentially 'combinedlabels.csv' but has one more column consisting of the label whether the action in the frame is 
drowsiness (1 for drowsiness and 0 otherwise). This ipython file shapes the data to be fed to neural network and then tunes it's parameters. It outputs trained model (model.h5) and plots indicating training and validation performance.

##### 3. test.ipython: 
This can be used for testing the model exported from Model.ipython. 
It publishes the time-label graph as "Test_g.png" and the data as "timeLabel_testg.json" where g is a number you can put at the begining to identify the files. 
It takes input the video in the avi format and you can use the custom name of your video along after you mention g value.

##### 4. VGG-16.ipython:
It takes as input the Labels_final.csv and the folders containing the train and test(validation) images (with appropriate names of folder).
It outputs the trained transfer learning model "vgg16_2.h5".

##### 5. VGG-19.ipython:
It takes as input the Labels_final.csv and the folders containing the train and test(validation) images (with appropriate names of folder).
It outputs the trained transfer learning model "vgg19_2.h5".

##### 6. ResNet.ipython:
It takes as input the Labels_final.csv and the folders containing the train and test(validation) images (with appropriate names of folder).
It outputs the trained transfer learning model "rsn.h5".

##### 5. Ensemble.ipython:
It takes as input the the processed input and output data as dumped by model.ipython. It runs the ensemble of 8 models from scikit learn, the LSTM model trained in model.h5, and the 3 transfer learning model. It dumps the 8 models along with the ninth blending function (blend.h5). This is used in test.ipython.

