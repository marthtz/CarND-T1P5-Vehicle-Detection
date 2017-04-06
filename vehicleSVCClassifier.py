################################################################################
#  ___                     _   
# |_ _|_ __  _ __  ___ _ _| |_ 
#  | || '  \| '_ \/ _ \ '_|  _|
# |___|_|_|_| .__/\___/_|  \__|
#           |_|                
################################################################################
import cv2
import pickle
import copy
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC

from tools import *



################################################################################
#  __  __      _      
# |  \/  |__ _(_)_ _  
# | |\/| / _` | | ' \ 
# |_|  |_\__,_|_|_||_|
################################################################################


# 1. Read in cars and notcars
################################################################################
cars = []
notcars = []

for root, dirs, files in os.walk("data/", topdown=False):
    for name in files:
        #print(os.path.join(root, name))
        fname = os.path.join(root, name)
        if 'png' in fname or 'jpg' in fname:
            if 'non-vehicles' in fname:
                notcars.append(fname)
            else:
                cars.append(fname)

print('Number of car items: {}'.format(len(cars)))
print('Number of not car items: {}'.format(len(notcars)))


# 2. Set parameters for model
################################################################################
color_space = 'YUV'#'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32   # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


# 3. Extract features from data sets
################################################################################
car_features = extractFeatures(cars,
                               color_space=color_space,
                               spatial_size=spatial_size,
                               hist_bins=hist_bins,
                               orient=orient,
                               pix_per_cell=pix_per_cell,
                               cell_per_block=cell_per_block,
                               hog_channel=hog_channel,
                               spatial_feat=spatial_feat,
                               hist_feat=hist_feat,
                               hog_feat=hog_feat)

notcar_features = extractFeatures(notcars,
                                  color_space=color_space,
                                  spatial_size=spatial_size,
                                  hist_bins=hist_bins,
                                  orient=orient,
                                  pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block,
                                  hog_channel=hog_channel,
                                  spatial_feat=spatial_feat,
                                  hist_feat=hist_feat,
                                  hog_feat=hog_feat)


# 4. Shuffle feature sets
################################################################################
rand_state = 0
car_features = shuffle(car_features, random_state=rand_state)
notcar_features = shuffle(notcar_features, random_state=rand_state)


# 5. Scale all features
################################################################################
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)


# 6. Define the labels vector
################################################################################
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# 7. Split up data to ensure both categeries have the same split
################################################################################
scaled_car_features = scaled_X[:len(car_features)][:]
scaled_notcar_features = scaled_X[len(car_features):][:]

# Split up data into randomized training and test sets
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(scaled_car_features,
                                                            np.ones(len(car_features)),
                                                            test_size=0.2,
                                                            random_state=rand_state)
# Split up data into randomized training and test sets
X_train_nc, X_test_nc, y_train_nc, y_test_nc = train_test_split(scaled_notcar_features,
                                                            np.zeros(len(notcar_features)),
                                                            test_size=0.2,
                                                            random_state=rand_state)

# 8. Re-combine features and shuffle again
################################################################################
X_train = np.vstack((X_train_c, X_train_nc))
X_test = np.vstack((X_test_c, X_test_nc))
y_train = np.hstack((y_train_c, y_train_nc))
y_test = np.hstack((y_test_c, y_test_nc))

X_train, y_train = shuffle(X_train, y_train, random_state=rand_state)
X_test, y_test = shuffle(X_test, y_test, random_state=rand_state)


# 9. Define and train linear SVC classifier
################################################################################
print('Using:',orient,'orientations',pix_per_cell,
      'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)


# 10. Check the score of the SVC
################################################################################
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

n_predict = 10
startVal = np.random.randint(len(X_test)-n_predict)
print('My SVC predicts: ', svc.predict(X_test[startVal:startVal+n_predict]))
print('For these',n_predict, 'labels: ', y_test[startVal:startVal+n_predict])


# 11. Save the classifier
################################################################################
dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
dist_pickle["color_space"] = color_space

pickle.dump(dist_pickle, open("svc_YUV_32_indv.pkl", "wb"))
