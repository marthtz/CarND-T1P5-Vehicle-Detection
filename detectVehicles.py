################################################################################
#  ___                     _      
# |_ _|_ __  _ __  ___ _ _| |_ ___
#  | || '  \| '_ \/ _ \ '_|  _(_-<
# |___|_|_|_| .__/\___/_|  \__/__/
#           |_|                   
################################################################################
import numpy as np
import cv2
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import glob
from skimage.feature import hog
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import collections

from tools import *



################################################################################
#                             ___ _           _ _          
#  _ __ _ _ ___  __ ___ _____| _ (_)_ __  ___| (_)_ _  ___ 
# | '_ \ '_/ _ \/ _/ -_|_-<_-<  _/ | '_ \/ -_) | | ' \/ -_)
# | .__/_| \___/\__\___/__/__/_| |_| .__/\___|_|_|_||_\___|
# |_|                              |_|                     
################################################################################
def processPipeline(img):
    global svc
    global X_scaler
    global orient
    global pix_per_cell
    global cell_per_block
    global spatial_size
    global hist_bins
    global color_space
    global heatmapFIFO
    global heatmapFIFOThres
    global searchWin
    global setDebug

    # 1. Reset new heatmap
    ############################################################################
    heatInit = np.zeros_like(img[:,:,0]).astype(np.float)

    # Window list for individual search window approach
    # windowList = generateWindows(img[:,:,0].shape, ystart, ystop, [64, 128], 0.5)

    # 2. Init lists for detected items
    ############################################################################
    box_listlist = []
    if setDebug is True:
        detect_imglist = []

    # 3. Find cars in image
    ############################################################################
    for sW in searchWin:
        scale = sW[0]
        ystart = sW[1]
        ystop = sW[2]
        cells_overlap = sW[3]

        if setDebug is True:
            detect_img, box_list = findCars(img, ystart, ystop, scale,
                                            cells_overlap, svc, X_scaler, orient,
                                            pix_per_cell, cell_per_block,
                                            spatial_size, hist_bins, color_space, setDebug)
            detect_imglist.append(detect_img)
        else:
            box_list = findCars(img, ystart, ystop, scale,
                                cells_overlap, svc, X_scaler, orient,
                                pix_per_cell, cell_per_block,
                                spatial_size, hist_bins, color_space, setDebug)
        box_listlist += box_list

    # 3. Find cars in image with individual search window approach
    #detect_img, box_list = findCarsSingle(img, windowList, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)

    # 4. Add heat to each positive detection box in box list
    ############################################################################
    heat = addHeat(heatInit, box_listlist)

    if procImg is True:
        # 5. Apply threshold to help remove false positives
        ########################################################################
        heatmapLabel = applyThreshold(heat, 1)
    else:
        # 6. Add latest heatmap to FIFO
        ############################################################################
        heatmapFIFO.append(heat)

        # 7. Combine heatmaps
        ############################################################################
        heatmapLabel = sum(heatmapFIFO)

        # 8. Threshold combined heatmaps
        ############################################################################
        heatmapLabel[heatmapLabel <= heatmapFIFOThres] = 0

    # 9. Find final boxes from heatmap using label function
    ############################################################################
    labels = label(heatmapLabel)
    finalImg = drawLabeledBBoxes(np.copy(img), labels)

    # 10. Generate debug image with intermediate processing steps
    ############################################################################
    if setDebug is True:
        heatmap = np.clip(heatmapLabel, 0, 255)
        heatmap_dbg = heatmap.astype(np.uint8)
        maxVal = np.amax(heatmap_dbg)
        heatmap_dbg *= (255//maxVal)
        finalImg = debugImg(detect_imglist[0], detect_imglist[1], heatmap_dbg, finalImg)

    return finalImg




################################################################################
#   ___ _     _          _    
#  / __| |___| |__  __ _| |___
# | (_ | / _ \ '_ \/ _` | (_-<
#  \___|_\___/_.__/\__,_|_/__/
################################################################################

# Define search windows
# p1: search window scale
# p2: region of interest top y axis value
# p3: region of interest bottom y axis value
# p4: search window overlap in number of cells
searchWin = [(2.0, 368, 624, 3),
             (1.5, 368, 560, 2)]

# Set FIFI length and combined heatmap threshold
heatmapFIFOLen = 14
heatmapFIFOThres = 12

# Init FIFO
heatmapFIFO = collections.deque(maxlen=heatmapFIFOLen)

# Init classifier parameters
svc = None
X_scaler =None
orient = None
pix_per_cell = None
cell_per_block = None
spatial_size = None
hist_bins = None
color_space = None

# Init debug flag
setDebug = None

# Init image flag
procImg = None




################################################################################
#  __  __      _      
# |  \/  |__ _(_)_ _  
# | |\/| / _` | | ' \ 
# |_|  |_\__,_|_|_||_|
################################################################################

files = sys.argv[1:]

for index, file in enumerate(files):

    if index == 0:
        if file == 'debug':
            setDebug = True
            continue
        else:
            setDebug = False

    if file.endswith('pkl'):
        dist_pickle = pickle.load(open(file, "rb"))
        svc = dist_pickle["svc"]
        X_scaler = dist_pickle["scaler"]
        orient = dist_pickle["orient"]
        pix_per_cell = dist_pickle["pix_per_cell"]
        cell_per_block = dist_pickle["cell_per_block"]
        spatial_size = dist_pickle["spatial_size"]
        hist_bins = dist_pickle["hist_bins"]
        color_space = dist_pickle["color_space"]

    elif file.endswith('mp4'):
        procImg = False
        if setDebug is True:
            outputExt = '_detect_debug.mp4'
        else:
            outputExt = '_detect.mp4'
        video_output = file.replace('.mp4',outputExt)

        clip1 = VideoFileClip(file)
        clip = clip1.fl_image(processPipeline)
        clip.write_videofile('output_images/' + video_output, audio=False)

    elif file.endswith('jpg'):
        procImg = True
        img = mpimg.imread(file)
        fName = file.replace('test_images/','')
        if setDebug is True:
            enableDebugImg = True
            debugImgFile = 'output_images/' + fName
        finalImg = processPipeline(img)
        if setDebug is True:
            outputExt = '_detect_debug.jpg'
        else:
            outputExt = '_detect.jpg'
        cv2.imwrite('output_images/' + fName.replace('.jpg', outputExt),
                    cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
