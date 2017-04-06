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

from sklearn.svm import LinearSVC



################################################################################
#                         _    ___     _         
#  __ ___ _ ___ _____ _ _| |_ / __|___| |___ _ _ 
# / _/ _ \ ' \ V / -_) '_|  _| (__/ _ \ / _ \ '_|
# \__\___/_||_\_/\___|_|  \__|\___\___/_\___/_|  
#                                                
#
# Convert an RGB image to another color space
#
################################################################################
def convertColor(img, conv):
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)



################################################################################
#         _         _  _ _    _   
#  __ ___| |___ _ _| || (_)__| |_ 
# / _/ _ \ / _ \ '_| __ | (_-<  _|
# \__\___/_\___/_| |_||_|_/__/\__|
#
# Define a function to compute color histogram features
#
################################################################################
def colorHist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0],
                                    channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features



################################################################################
#  _    _      ___            _   _      _ 
# | |__(_)_ _ / __|_ __  __ _| |_(_)__ _| |
# | '_ \ | ' \\__ \ '_ \/ _` |  _| / _` | |
# |_.__/_|_||_|___/ .__/\__,_|\__|_\__,_|_|
#                 |_|                      
#
# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
################################################################################
def binSpatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()

    return np.hstack((color1, color2, color3))



################################################################################
#           _   _  _  ___   ___ ___         _                   
#  __ _ ___| |_| || |/ _ \ / __| __|__ __ _| |_ _  _ _ _ ___ ___
# / _` / -_)  _| __ | (_) | (_ | _/ -_) _` |  _| || | '_/ -_|_-<
# \__, \___|\__|_||_|\___/ \___|_|\___\__,_|\__|\_,_|_| \___/__/
# |___/                                                         
#
# Define a function to return HOG features and visualization
################################################################################
def getHOGFeatures(img, orient, pix_per_cell, cell_per_block,
                   vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=True,
                                  feature_vector=False)
        return features, hog_image
    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=feature_vec)
        return features



################################################################################
#          _               _   ___         _                   
#  _____ _| |_ _ _ __ _ __| |_| __|__ __ _| |_ _  _ _ _ ___ ___
# / -_) \ /  _| '_/ _` / _|  _| _/ -_) _` |  _| || | '_/ -_|_-<
# \___/_\_\\__|_| \__,_\__|\__|_|\___\__,_|\__|\_,_|_| \___/__/
#
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
################################################################################
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extractFeaturesSingleImg(img, color_space='RGB', spatial_size=(32, 32),
                    hist_bins=32, orient=9, 
                    pix_per_cell=8, cell_per_block=2, hog_channel=0,
                    spatial_feat=True, hist_feat=True, hog_feat=True):

    file_features = []

    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        feature_image = convertColor(img, conv=color_space)
    else:
        feature_image = np.copy(img)

    if spatial_feat == True:
        spatial_features = binSpatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)

    if hist_feat == True:
        hist_features = colorHist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)

    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(getHOGFeatures(feature_image[:,:,channel],
                                                   orient,
                                                   pix_per_cell,
                                                   cell_per_block,
                                                   vis=False,
                                                   feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = getHOGFeatures(feature_image[:,:,hog_channel],
                                          orient,
                                          pix_per_cell,
                                          cell_per_block,
                                          vis=False,
                                          feature_vec=True)

        # Append the new feature vector to the features list
        file_features.append(hog_features)

    return np.concatenate(file_features)



################################################################################
#          _               _   ___         _                   
#  _____ _| |_ _ _ __ _ __| |_| __|__ __ _| |_ _  _ _ _ ___ ___
# / -_) \ /  _| '_/ _` / _|  _| _/ -_) _` |  _| || | '_/ -_|_-<
# \___/_\_\\__|_| \__,_\__|\__|_|\___\__,_|\__|\_,_|_| \___/__/
#
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
################################################################################
def extractFeatures(imgs, color_space='RGB', spatial_size=(32, 32),
                    hist_bins=32, orient=9, 
                    pix_per_cell=8, cell_per_block=2, hog_channel=0,
                    spatial_feat=True, hist_feat=True, hog_feat=True):

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        try:
            image = mpimg.imread(file)  # read as RGB
            image = (image*255).astype(np.uint8)
        except:
            print(file)
            sys.exit()

        file_features = extractFeaturesSingleImg(image,
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

        features.append(file_features)

    return features



################################################################################
#                             _     __      ___         _               
#  __ _ ___ _ _  ___ _ _ __ _| |_ __\ \    / (_)_ _  __| |_____ __ _____
# / _` / -_) ' \/ -_) '_/ _` |  _/ -_) \/\/ /| | ' \/ _` / _ \ V  V (_-<
# \__, \___|_||_\___|_| \__,_|\__\___|\_/\_/ |_|_||_\__,_\___/\_/\_//__/
# |___/                                                                 
#
# Generate a list of search windows
################################################################################
def generateWindows(imgShape, yStart, yStop, windowSize, overlap):
    # Initialize a list to append window positions to
    windowList = []

    # Compute the span of the region to be searched
    xSpan = imgShape[1]
    ySpan = yStop - yStart

    for winSize in windowSize:
        # Compute the number of pixels per step in x/y
        pixelPerStepX = np.int(winSize * (1 - overlap))
        pixelPerStepY = np.int(winSize * (1 - overlap))

        # Compute the number of windows in x/y
        windowsX = (xSpan//pixelPerStepX) - 1
        windowsY = (ySpan//pixelPerStepY) - 1

        # Loop through finding x and y window positions
        for indexX in range(windowsX):
            for indexY in range(windowsY):
                # Calculate window position
                startx = (indexX * pixelPerStepX)
                endx = startx + winSize
                starty = (indexY * pixelPerStepY) + yStart
                endy = starty + winSize

                # Append window position to list
                windowList.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return windowList



################################################################################
#          _    _ _  _          _   
#  __ _ __| |__| | || |___ __ _| |_ 
# / _` / _` / _` | __ / -_) _` |  _|
# \__,_\__,_\__,_|_||_\___\__,_|\__|
#                                   
# Add a value of one for each pixel in a detection windows that is true
#
################################################################################
def addHeat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes



################################################################################
#                 _     _____ _               _        _    _ 
#  __ _ _ __ _ __| |_  |_   _| |_  _ _ ___ __| |_  ___| |__| |
# / _` | '_ \ '_ \ | || || | | ' \| '_/ -_|_-< ' \/ _ \ / _` |
# \__,_| .__/ .__/_|\_, ||_| |_||_|_| \___/__/_||_\___/_\__,_|
#      |_|  |_|     |__/                                      
#
# Set all values in a heatmap to 0 that are below a given threshold
#
################################################################################
def applyThreshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



################################################################################
#     _                 _         _         _        _ ___ ___                 
#  __| |_ _ __ ___ __ _| |   __ _| |__  ___| |___ __| | _ ) _ ) _____ _____ ___
# / _` | '_/ _` \ V  V / |__/ _` | '_ \/ -_) / -_) _` | _ \ _ \/ _ \ \ / -_|_-<
# \__,_|_| \__,_|\_/\_/|____\__,_|_.__/\___|_\___\__,_|___/___/\___/_\_\___/__/
#                                                                              
#
# Use map of prediction labels to generate boxes for around each labeled area
#
################################################################################
def drawLabeledBBoxes(img, labels):

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        bbox_xLen = bbox[1][0] - bbox[0][0]
        bbox_yLen = bbox[1][1] - bbox[0][1]

        if bbox_xLen < 64 or bbox_yLen < 64:
            pass
        else:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    # Return the image
    return img



################################################################################
#   __ _         _  ___             ___ _           _     
#  / _(_)_ _  __| |/ __|__ _ _ _ __/ __(_)_ _  __ _| |___ 
# |  _| | ' \/ _` | (__/ _` | '_(_-<__ \ | ' \/ _` | / -_)
# |_| |_|_||_\__,_|\___\__,_|_| /__/___/_|_||_\__, |_\___|
#                                             |___/       
#
# Find cars on a given image by extracting features for each search window
# individually.
#
################################################################################
def findCarsSingle(img, windows, svc, X_scaler, orient, pix_per_cell,
                   cell_per_block, spatial_size, hist_bins, color_space):

    box_list = []
    draw_img = np.copy(img)

    # Debug code
    #scan_img = np.copy(img)
    
    # Search all windows
    for window in windows:
        img_tosearch = cv2.resize(img[window[0][1]:window[1][1],
                                      window[0][0]:window[1][0]],
                                  (64, 64))

        # Extract features from image
        features = extractFeaturesSingleImg(img_tosearch,
                                            color_space=color_space,
                                            spatial_size=spatial_size,
                                            hist_bins=hist_bins,
                                            orient=orient,
                                            pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            hog_channel='ALL',
                                            spatial_feat=True,
                                            hist_feat=True,
                                            hog_feat=True)

        # Scale features and make a prediction
        test_features = X_scaler.transform(np.array(features).reshape(1, -1))
        test_prediction = svc.predict(test_features)

        boxPt1 = (window[0][0], window[0][1])
        boxPt2 = (window[1][0], window[1][1])

        if test_prediction == 1:
            # boxPt1 = (window[0][0], window[0][1])
            # boxPt2 = (window[1][0], window[1][1])

            box_list.append((boxPt1, boxPt2))
            cv2.rectangle(draw_img, boxPt1, boxPt2, (0,0,255), 6)
            color = (0,255,0)
        else:
            color = (255,0,0)

        # cv2.rectangle(scan_img,boxPt1, boxPt2,color,6)
        # cv2.imwrite( "output_images/test_{:04d}.jpg".format(frameCounter), cv2.cvtColor(scan_img, cv2.COLOR_BGR2RGB))
        # frameCounter += 1

    return draw_img, box_list



################################################################################
#   __ _         _  ___             
#  / _(_)_ _  __| |/ __|__ _ _ _ ___
# |  _| | ' \/ _` | (__/ _` | '_(_-<
# |_| |_|_||_\__,_|\___\__,_|_| /__/
#                                   #
# Find cars on a given image by extracting HOG  features only once for the
# entire region of interest and then subsample the features for each search
# windows for detection.
#
################################################################################
def findCars(img, ystart, ystop, scale, cells_overlap, svc, X_scaler, orient,
             pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, setDebug):

    if setDebug is True:
        draw_img = np.copy(img)

    # List to store all detection windows
    box_list = []

    # Limit image to region of interest only
    img_tosearch = img[ystart:ystop,:,:]

    # Scale region of interest to realize differenct search window sizes
    ctrans_tosearch = convertColor(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Separate each channel of the image
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1

    # Instead of overlap, define how many cells to step
    cells_per_step = cells_overlap
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step +1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step +1

    # Compute individual channel HOG features for the entire image
    hog1 = getHOGFeatures(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = getHOGFeatures(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = getHOGFeatures(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Search all windows
    for xb in range(nxsteps):
        for yb in range(nysteps):
            # debug
            # scan_img = np.copy(img)

            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window,
                                                xleft:xleft+window],
                                (64,64))
          
            # Get color features
            spatial_features = binSpatial(subimg, size=spatial_size)
            hist_features = colorHist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features,
                                                          hist_features,
                                                          hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            

            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)

            if test_prediction == 1:
                boxPt1 = (xbox_left, ytop_draw+ystart)
                boxPt2 = (xbox_left+win_draw,ytop_draw+win_draw+ystart)

                box_list.append((boxPt1, boxPt2))

                if setDebug is True:
                    cv2.rectangle(draw_img, boxPt1, boxPt2, (0,0,255), 6)

    if setDebug is True:
        return draw_img, box_list
    else:
        return box_list


################################################################################
#     _     _              ___            
#  __| |___| |__ _  _ __ _|_ _|_ __  __ _ 
# / _` / -_) '_ \ || / _` || || '  \/ _` |
# \__,_\___|_.__/\_,_\__, |___|_|_|_\__, |
#                    |___/          |___/ 
#
# Create a debug image to visualze processing steps
#
################################################################################
def debugImg(detect_s1_img, detect_s2_img, heatmap, draw_img):
    # Debug image is same size as original image
    # Structure is 2 images in the upper half and 2 images in the lower half

    dbgImgX = int(detect_s1_img.shape[1])
    dbgImgY = int(detect_s1_img.shape[0])

    # Upper 2 imgs
    finalImgX = dbgImgX // 2
    finalImgY = dbgImgY // 2

    # Init debug image
    dbgImg = np.zeros((dbgImgY, dbgImgX, 3), dtype=np.uint8)

    # Prepare heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    # Detection image with windows size 1
    dbgImg[0:finalImgY, 0:finalImgX] = cv2.resize(detect_s1_img, (finalImgX, finalImgY), interpolation=cv2.INTER_AREA)
    # Detection image with windows size 1
    dbgImg[0:finalImgY, finalImgX:dbgImgX] = cv2.resize(detect_s2_img, (finalImgX, finalImgY), interpolation=cv2.INTER_AREA)
    # Heatmap
    dbgImg[finalImgY:dbgImgY, 0:finalImgX] = cv2.resize(heatmap, (finalImgX, finalImgY), interpolation=cv2.INTER_AREA)
    # Final detection image
    dbgImg[finalImgY:dbgImgY, finalImgX:dbgImgX] = cv2.resize(draw_img, (finalImgX, finalImgY), interpolation=cv2.INTER_AREA)

    # Return final debug image
    return dbgImg
