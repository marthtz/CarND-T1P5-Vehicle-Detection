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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from skimage.feature import hog
from scipy.ndimage.measurements import label
import collections

from tools import *




################################################################################
#  _    _             ___ _            
# | |  (_)_ _  ___   / __| |__ _ ______
# | |__| | ' \/ -_) | (__| / _` (_-<_-<
# |____|_|_||_\___|  \___|_\__,_/__/__/
#
# Defines a class to receive the characteristics of each line detection
################################################################################
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #average x values of the fitted line over the last n iterations
        self.bestXFitted = None
        #polynomial coefficients averaged over the last n iterations
        self.bestFit = None
        #polynomial coefficients for the most recent fit
        self.currentFit = [np.array([False])]
        self.currentXFitted = None
        #radius of curvature of the line in some units
        self.radiusCurvature = None
        #x values for detected line pixels
        self.allX = None  
        #y values for detected line pixels
        self.allY = None





################################################################################
#                       ___                     
# __ __ ___ _ __ _ _ __|_ _|_ __  __ _ __ _ ___ 
# \ V  V / '_/ _` | '_ \| || '  \/ _` / _` / -_)
#  \_/\_/|_| \__,_| .__/___|_|_|_\__,_\__, \___|
#                 |_|                 |___/     
#
# Warp an image and return the warped image plus the inverse transormation matrix
################################################################################
def warpImage(img, debug=False):
    global enableDebugVid
    global enableDebugImg
    global debugImgFile

    img_size = (img.shape[0], img.shape[1])

    # P1 P2
    # P4 P3
    tl = [588, 455]
    tr = [694, 455]
    br = [1100, img_size[0]-1]
    bl = [200, img_size[0]-1]

    if (debug is True) and (enableDebugImg is True):
        img2 = copy.copy(img)
        cv2.line(img2, (tl[0], tl[1]), (tr[0], tr[1]), [255, 0, 0], 2)
        cv2.line(img2, (tl[0], tl[1]), (bl[0], bl[1]), [255, 0, 0], 2)
        cv2.line(img2, (bl[0], bl[1]), (br[0], br[1]), [255, 0, 0], 2)
        cv2.line(img2, (tr[0], tr[1]), (br[0], br[1]), [255, 0, 0], 2)
        cv2.imwrite(debugImgFile.replace('.jpg','_original_lines.jpg'),
                    cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    # Set transform points
    src = np.float32([tl, tr, br, bl])

    midPoint = img_size[1]//2
    offset = midPoint//2
    tl = [offset, 0]
    tr = [img.shape[1]-1-offset, 0]
    br = [img.shape[1]-1-offset, img.shape[0]-1]
    bl = [offset, img.shape[0]-1]

    dst = np.float32([tl, tr, br, bl])

    # Warp image
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    if enableDebugImg is True:
        if debug is True:
            warped2 = copy.copy(warped)
            cv2.line(warped2, (tl[0], tl[1]), (tr[0], tr[1]), [255, 0, 0], 2)
            cv2.line(warped2, (tl[0], tl[1]), (bl[0], bl[1]), [255, 0, 0], 2)
            cv2.line(warped2, (bl[0], bl[1]), (br[0], br[1]), [255, 0, 0], 2)
            cv2.line(warped2, (tr[0], tr[1]), (br[0], br[1]), [255, 0, 0], 2)
            cv2.imwrite(debugImgFile.replace('.jpg','_warped_lines.jpg'),
                        cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB))
        else:
            warped2 = np.dstack((warped, warped, warped))*255
            cv2.imwrite(debugImgFile.replace('.jpg','_binary_warped.jpg'),
                        cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB))

    return warped, Minv






################################################################################
#          _        _   _  _ _____   _____      _         ___  __
#  ___ ___| |___ __| |_| || / __\ \ / / __| ___| |__  ___| \ \/ /
# (_-</ -_) / -_) _|  _| __ \__ \\ V /\__ \/ _ \ '_ \/ -_) |>  < 
# /__/\___|_\___\__|\__|_||_|___/ \_/ |___/\___/_.__/\___|_/_/\_\
#
# Do a HSV color selection and sobel X on a given image and return a binary
# masked image
################################################################################
def selectHSVSobelX(img):
    global enableDebugVid
    global enableDebugImg
    global debugImgFile

    # Grayscale image for sobel in x axis
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    thresh_min = 50
    thresh_max = 225
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    # Color select on HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Yellow mask
    low = np.array([0, 70, 130])
    high = np.array([25, 255, 255])
    maskY = cv2.inRange(hsv, low, high)

    # White mask
    low = np.array([0, 0, 175])
    high = np.array([40, 25, 255])
    maskW = cv2.inRange(hsv, low, high)
    maskLane = cv2.bitwise_or(maskY, maskW)

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(maskLane == 255) | (sxbinary == 255)] = 1

    if (enableDebugVid is True) or (enableDebugImg is True):
        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, maskLane))

    if enableDebugImg is True:
        cv2.imwrite(debugImgFile.replace('.jpg','_original_thresholded.jpg'),
                    cv2.cvtColor(color_binary, cv2.COLOR_BGR2RGB))

    if enableDebugVid is True:
        return color_binary, combined_binary
    else:
        return combined_binary





################################################################################
#   __ _          _                  _    _          
#  / _(_)_ _  ___| |   __ _ _ _  ___| |  (_)_ _  ___ 
# |  _| | ' \/ -_) |__/ _` | ' \/ -_) |__| | ' \/ -_)
# |_| |_|_||_\___|____\__,_|_||_\___|____|_|_||_\___|
#
# Find a lane line on a given binary image
################################################################################
def findLaneLine(binary_warped, side, slidingWindow, fit=None):
    global ploty
    global enableDebugVid
    global enableDebugImg
    global debugImgFile

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Set the width of the windows +/- margin
    margin = 100

    if slidingWindow is True:
        # Set window parameter
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Take a histogram of the bottom third of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//3:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        if side == 'left':
            x_base = np.argmax(histogram[:midpoint])
        else:
            x_base = np.argmax(histogram[midpoint:]) + midpoint

        # Reset base and indices
        # Current positions to be updated for each window
        x_current = x_base
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & \
                         (nonzeroy < win_y_high) & \
                         (nonzerox >= win_x_low) & \
                         (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            lane_inds.append(good_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)
    else:
        # Use look ahead filter
        lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - margin)) & \
                     (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin)))

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    try:
        # Fit a second order polynomial to each
        fit = np.polyfit(y, x, 2)

        # Generate x and y values for plotting
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

        if side == 'left':
            # Left line color is red
            lineColor = [255, 0, 0]
        else:
            # Right line color is blue
            lineColor = [0, 0, 255]
        # Add fitted line
        out_img[y, x] = lineColor
        if (enableDebugVid is True) or (enableDebugImg is True):
            # Add polynom to image
            for index, item in enumerate(fitx):
                cv2.circle(out_img,
                           (int(fitx[index]), int(ploty[index])),
                           2,
                           (255, 255, 0 ),  # yellow
                           1,
                           8)
    except:
        fit = None
        fitx = None

    if slidingWindow is False and fit != None:
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([fitx-margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx+margin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        window_img = np.zeros_like(out_img)
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))

        # Combine images
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    if enableDebugImg is True:
        cv2.imwrite(debugImgFile.replace('.jpg','_{}_binary_line.jpg'.format(side)),
                    cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))

    return (fit, fitx, x, y, out_img)





################################################################################
#           _    ___                 
#  __ _ ___| |_ / __|  _ _ ___ _____ 
# / _` / -_)  _| (_| || | '_\ V / -_)
# \__, \___|\__|\___\_,_|_|  \_/\___|
# |___/                              
#
# Get curvature of a 2nd order polynom
################################################################################
def getCurve(leftx, lefty, rightx, righty):
    global ploty
    global ym_per_pix
    global xm_per_pix

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature (signed)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0])

    return left_curverad, right_curverad





################################################################################
#           _    ___   __  __         _   
#  __ _ ___| |_ / _ \ / _|/ _|___ ___| |_ 
# / _` / -_)  _| (_) |  _|  _(_-</ -_)  _|
# \__, \___|\__|\___/|_| |_| /__/\___|\__|
# |___/                                   
#
# Get vehicle offset from the lane center
################################################################################
def getOffset(leftx, rightx, imageWidth):
    global xm_per_pix

    midPoint = imageWidth / 2
    carMidPoint = (rightx + leftx) // 2
    pixelOffset = midPoint - carMidPoint
    meterOffset = xm_per_pix * pixelOffset

    return meterOffset





################################################################################
#               _ _    _           _   ___            
#  _  _ _ _  __| (_)__| |_ ___ _ _| |_|_ _|_ __  __ _ 
# | || | ' \/ _` | (_-<  _/ _ \ '_|  _|| || '  \/ _` |
#  \_,_|_||_\__,_|_/__/\__\___/_|  \__|___|_|_|_\__, |
#                                               |___/ 
#
# Defines a class to receive the characteristics of each line detection
################################################################################
def undistortImg(img):
    # Open pre-calculated values for undistortion
    with open(r"cam_cal_pickle.p", "rb") as input_file:
        dist_pickle = pickle.load(input_file)

    undistImg = cv2.undistort(img,
                              dist_pickle["mtx"],
                              dist_pickle["dist"],
                              None,
                              dist_pickle["mtx"])

    if enableDebugImg is True:
        cv2.imwrite(debugImgFile.replace('.jpg','_undistorted.jpg'),
                    cv2.cvtColor(undistImg, cv2.COLOR_BGR2RGB))


    return undistImg





################################################################################
#     _                 _                  
#  __| |_ _ __ ___ __ _| |   __ _ _ _  ___ 
# / _` | '_/ _` \ V  V / |__/ _` | ' \/ -_)
# \__,_|_| \__,_|\_/\_/|____\__,_|_||_\___|
#
# Draw calculated lane on original image
################################################################################
def drawLane(img, warped, left_fitx, right_fitx, Minv, leftX, leftY, rightX, rightY):
    global ploty

    # Create an empty image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp2 = copy.copy(color_warp)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    color_warp2[leftY, leftX] = [255, 0, 0]
    color_warp2[rightY, rightX] = [0, 0, 255]

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    newwarp2 = cv2.warpPerspective(color_warp2, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    result = cv2.addWeighted(result, 0.8, newwarp2, 1, 0)

    return result





################################################################################
#          _    _ _____        _   
#  __ _ __| |__| |_   _|____ _| |_ 
# / _` / _` / _` | | |/ -_) \ /  _|
# \__,_\__,_\__,_| |_|\___/_\_\\__|
#                                  
# Adds text to an image
################################################################################
def addText(img, imgText, textSize):
    # Set offsets for each line
    y0, dy = 50, 50
    # Add all passed text
    for i, line in enumerate(imgText):
        y = y0 + i*dy
        cv2.putText(img,
                    line,
                    (50,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    textSize,
                    (255,255,255),
                    2,
                    cv2.LINE_AA)
    return img





################################################################################
#       _           _ _          
#  _ __(_)_ __  ___| (_)_ _  ___ 
# | '_ \ | '_ \/ -_) | | ' \/ -_)
# | .__/_| .__/\___|_|_|_||_\___|
# |_|    |_|                     
#
# Image processing pipeline
################################################################################
def pipeline(img):
    global enableDebugVid
    global frameCounter
    global leftLine
    global rightLine
    global bestFitWeight
    global bestFitWeight2
    global straightLineThreshold
    global ploty

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


    # A) Inits for vehicle detection
    ############################################################################
    heatInit = np.zeros_like(img[:,:,0]).astype(np.float)
    detect_imglist = []
    box_listlist = []


    # 1. Increment frame counter
    ############################################################################
    frameCounter += 1

    # 2. Undistort camera image
    ############################################################################
    undistImg = undistortImg(img)


    # B) Find cars in image - only support non debug version
    ############################################################################
    for sW in searchWin:
        scale = sW[0]
        ystart = sW[1]
        ystop = sW[2]
        cells_overlap = sW[3]
        box_list = findCars(img, ystart, ystop, scale,
                            cells_overlap, svc, X_scaler, orient,
                            pix_per_cell, cell_per_block,
                            spatial_size, hist_bins, color_space, False)
        box_listlist += box_list

   # C) Add heat to each box in box list
    ############################################################################
    heat = addHeat(heatInit, box_listlist)
    heatmapFIFO.append(heat)
    heatmapLabel = sum(heatmapFIFO)
    heatmapLabel[heatmapLabel <= heatmapFIFOThres] = 0


    # 3. Do color selecton HSV and do sobel on x and return a binary thresholded
    #    image
    ############################################################################
    if enableDebugVid is True:
        combinedColorBinImg_dbg, combinedBinImg = selectHSVSobelX(undistImg)
    else:
        combinedBinImg = selectHSVSobelX(undistImg)

    # 4. Warp binary thresholded image
    ############################################################################
    binary_warped, Minv = warpImage(combinedBinImg)
    # Do some basic variable init (only once per video)
    if frameCounter == 1:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    # 4a. Warp original image for debug purpose
    ############################################################################
    if (enableDebugVid is True) or (enableDebugImg is True):
        warpedImg_dbg, Minv_unused = warpImage(undistImg, debug=True)

    # 5. Detect left lane line
    ############################################################################
    if leftLine.detected is False:
        # If no line has benn detection before do a sliding window search
        (leftLine.currentFit,
        leftLine.currentXFitted,
        leftLine.allX,
        leftLine.allY,
        warpedLaneImgL) = findLaneLine(binary_warped,
                                       side='left',
                                       slidingWindow=True)
    else:
        # Line has been detected before
        # Search new line within a margin of previous line
        (leftLine.currentFit,
        leftLine.currentXFitted,
        leftLine.allX,
        leftLine.allY,
        warpedLaneImgL) = findLaneLine(binary_warped,
                                       side='left',
                                       slidingWindow=False,
                                       fit=leftLine.bestFit)

    # 6. Detect right lane line
    ############################################################################
    if rightLine.detected is False:
        # Line has been detected before
        # Search new line within a margin of previous line
        (rightLine.currentFit,
        rightLine.currentXFitted,
        rightLine.allX,
        rightLine.allY,
        warpedLaneImgR) = findLaneLine(binary_warped,
                                       side='right',
                                       slidingWindow=True)
    else:
        # If no line has benn detection before do a sliding window search
        (rightLine.currentFit,
        rightLine.currentXFitted,
        rightLine.allX,
        rightLine.allY,
        warpedLaneImgR) = findLaneLine(binary_warped,
                                       side='right',
                                       slidingWindow=False,
                                       fit=rightLine.bestFit)

    # 7. Get curvature values for detected lane lines
    #    Catch no lines detected first
    ############################################################################
    if (leftLine.currentFit == None or leftLine.currentXFitted == None or
        rightLine.currentFit == None or rightLine.currentXFitted == None):
        # This catches the case when no lines have been detected
        leftLine.detected = False
        rightLine.detected = False

        if enableDebugVid is True:
            leftCurveValue = -1
            rightCurveValue = -1
            maxCurve = -1
            minCurve = -1
            maxDev = -1
            leftRange = -1
            rightRange = -1
    else:
        # 8. Some lines have been detected - get their curvature
        #    And catch the very first call (1st image of video)
        ########################################################################
        (leftCurveValue,
        rightCurveValue) = getCurve(leftLine.allX,
                                    leftLine.allY,
                                    rightLine.allX,
                                    rightLine.allY)

        if leftLine.bestFit == None:
            # Catch very first call
            leftLine.bestFit = leftLine.currentFit
            leftLine.bestXFitted = leftLine.currentXFitted
            rightLine.bestFit = rightLine.currentFit
            rightLine.bestXFitted = rightLine.currentXFitted
            leftLine.radiusCurvature = leftCurveValue
            rightLine.radiusCurvature = rightCurveValue

        # 9.  Do sanity check - check if detected lines are parallel
        ########################################################################
        # In case signs of lef/right curvature are different, the curvatures
        # have to be both very large to be considered straight
        # Define max difference in curvature as 50% of the larger curvature
        maxCurve = max(leftCurveValue, rightCurveValue)
        minCurve = min(leftCurveValue, rightCurveValue)
        #maxDev = maxCurve // 2
        maxDev = abs(maxCurve // 2)

        # Get cuvature range in which the smaller one has to fit in
        leftRange = (maxCurve - maxDev)
        rightRange = (maxCurve + maxDev)
        if np.sign(leftCurveValue) != np.sign(rightCurveValue):
            if(abs(leftCurveValue) > straightLineThreshold) and (abs(rightCurveValue) > straightLineThreshold):
                # Lines are parallel - set lane lines detected
                leftLine.detected = True
                rightLine.detected = True
            else:
                if (minCurve > leftRange) and (minCurve < rightRange):
                    # Lines are parallel - set lane lines detected
                    leftLine.detected = True
                    rightLine.detected = True
                else:
                    # Lines are not parallel - set lane lines not detected
                    leftLine.detected = False
                    rightLine.detected = False
        else:
            if (minCurve > leftRange) and (minCurve < rightRange):
                # Lines are parallel - set lane lines detected
                leftLine.detected = True
                rightLine.detected = True
            else:
                # Lines are not parallel - set lane lines not detected
                leftLine.detected = False
                rightLine.detected = False


    # 10. Do another sanity check if lane lines curvature is not too different
    #     to a revious fit
    ############################################################################
    # curveSanityThreshold = 0.35

    # # Sanity check on curvature left
    # leftCurveFactor = (leftCurveValue - leftLine.radiusCurvature) / leftLine.radiusCurvature
    # if np.absolute(leftCurveFactor) <= curveSanityThreshold:
    #     leftLine.detected = True
    # else:
    #     leftLine.detected = False
    # # Sanity check on curvature right
    # rightCurveFactor = (rightCurveValue - rightLine.radiusCurvature) / rightLine.radiusCurvature
    # if np.absolute(rightCurveFactor) <= curveSanityThreshold:
    #     rightLine.detected = True
    # else:
    #     rightLine.detected = False


    # 11. If lane lines are successfully detected update polynom and curavure
    #     with respect to previous detections (too smooth the values)
    ############################################################################
    if leftLine.detected is True:
        bestXFitted = leftLine.bestFit[0]*ploty**2 + leftLine.bestFit[1]*ploty + leftLine.bestFit[2]
        leftLine.bestXFitted = bestXFitted * bestFitWeight + leftLine.currentXFitted * (1-bestFitWeight)
        leftLine.radiusCurvature = leftLine.radiusCurvature * bestFitWeight2 + leftCurveValue *  (1-bestFitWeight2)
        leftLine.bestFit = np.polyfit(ploty, leftLine.bestXFitted, 2)

    if rightLine.detected is True:
        bestXFitted = rightLine.bestFit[0]*ploty**2 + rightLine.bestFit[1]*ploty + rightLine.bestFit[2]
        rightLine.bestXFitted = bestXFitted * bestFitWeight + rightLine.currentXFitted * (1-bestFitWeight)
        rightLine.radiusCurvature = rightLine.radiusCurvature * bestFitWeight2 + rightCurveValue *  (1-bestFitWeight2)
        rightLine.bestFit = np.polyfit(ploty, rightLine.bestXFitted, 2)

    # 12. Get vehicle offset to the center of the detected lane
    ############################################################################
    offsetValue = getOffset(leftLine.bestXFitted[-1],
                            rightLine.bestXFitted[-1],
                            binary_warped.shape[1])

    # 13. Draw detected lane on original image
    ############################################################################
    lanesImg = drawLane(undistImg,
                        binary_warped,
                        leftLine.bestXFitted,
                        rightLine.bestXFitted,
                        Minv,
                        leftLine.allX,
                        leftLine.allY,
                        rightLine.allX,
                        rightLine.allY)

    # 14. Add text to final image
    ############################################################################
    combinedRadius = int((abs(leftLine.radiusCurvature) + abs(rightLine.radiusCurvature))//2)
    if combinedRadius > straightLineThreshold:
        radiusText = 'inf - straight lane'
    else:
        radiusText = '{}m'.format(combinedRadius)
    curveText = 'Radius of curvature = {}'.format(radiusText)
    if offsetValue < 0:
        positionText = 'left'
    else:
        positionText = 'right'
    offsetText = 'Vehicle is {:.2f}m {} of center'.format(abs(offsetValue), positionText)
    finalImg = addText(lanesImg, [curveText, offsetText], textSize=1.5)

    # 14a. Generate debug image with additional text and intermediate images
    #      of the processing pipeline
    ############################################################################
    if enableDebugVid is True:
        # Add debug text to image
        combinedColorBinImg_dbg = addText(combinedColorBinImg_dbg,
                                       ['Left curvature = {}m'.format(int(leftLine.radiusCurvature)),
                                       'Right curvature = {}m'.format(int(rightLine.radiusCurvature)),
                                       'Current left curvature = {}m'.format(int(leftCurveValue)),
                                       'Curreny tight curvature = {}m'.format(int(rightCurveValue)),
                                       'Left detect: {}'.format(leftLine.detected),
                                       'Right detect: {}'.format(rightLine.detected),
                                       'Max curve: {}'.format(maxCurve),
                                       'Min curve: {}'.format(minCurve),
                                       'Left curve range: {}'.format(leftRange),
                                       'Right curve range: {}'.format(rightRange),
                                       'Frame: {}'.format(frameCounter)],
                                       textSize=1
                                       )

        # Replace final image with debug output
        finalImg = debugImg(finalImg, combinedColorBinImg_dbg, warpedImg_dbg, binary_warped, warpedLaneImgL, warpedLaneImgR)

    # D. Find final boxes from heatmap using label function
    ############################################################################
    labels = label(heatmapLabel)
    finalImg = drawLabeledBBoxes(np.copy(finalImg), labels)

    # Return processed video image
    return finalImg





################################################################################
#     _     _              ___            
#  __| |___| |__ _  _ __ _|_ _|_ __  __ _ 
# / _` / -_) '_ \ || / _` || || '  \/ _` |
# \__,_\___|_.__/\_,_\__, |___|_|_|_\__, |
#                    |___/          |___/ 
#
# Creates a debug image with extra info and intermediate images of the
# processing pipeline
################################################################################
def debugImg(finalImg, combinedColorBinImg_dbg, warpedImg_dbg, binary_warped, warpedLaneImgL, warpedLaneImgR):

    # Debug image is same size as original image
    # Structure is 2 images in the upper half and 4 images in the lower half

    dbgImgX = int(finalImg.shape[1])
    dbgImgY = int(finalImg.shape[0])

    # Upper 2 imgs
    finalImgX = dbgImgX // 2
    finalImgY = dbgImgY // 2

    # Lower 4 imgs
    warpImgX = dbgImgX // 4
    warpImgY = dbgImgY // 2

    # Create an RGB image from a binary one
    binary_warped2 = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Init debug image
    dbgImg = np.zeros((dbgImgY, dbgImgX, 3), dtype=np.uint8)

    # Final image
    dbgImg[0:finalImgY, 0:finalImgX] = cv2.resize(finalImg, (finalImgX, finalImgY), interpolation=cv2.INTER_AREA)
    # Original image thresholded
    dbgImg[0:finalImgY, finalImgX:dbgImgX] = cv2.resize(combinedColorBinImg_dbg, (finalImgX, finalImgY), interpolation=cv2.INTER_AREA)
    # Warped original image
    dbgImg[finalImgY:dbgImgY, 0:warpImgX] = cv2.resize(warpedImg_dbg, (warpImgX, warpImgY), interpolation=cv2.INTER_AREA)
    # Thresholded warped binary image
    dbgImg[finalImgY:dbgImgY, warpImgX:2*warpImgX] = cv2.resize(binary_warped2, (warpImgX, warpImgY), interpolation=cv2.INTER_AREA)
    # Left lane detection
    dbgImg[finalImgY:dbgImgY, 2*warpImgX:3*warpImgX] = cv2.resize(warpedLaneImgL, (warpImgX, warpImgY), interpolation=cv2.INTER_AREA)
    # Right lane detection
    dbgImg[finalImgY:dbgImgY, 3*warpImgX:4*warpImgX] = cv2.resize(warpedLaneImgR, (warpImgX, warpImgY), interpolation=cv2.INTER_AREA)

    # Return final debug image
    return dbgImg





################################################################################
#   ___ _     _          _    
#  / __| |___| |__  __ _| |___
# | (_ | / _ \ '_ \/ _` | (_-<
#  \___|_\___/_.__/\__,_|_/__/
################################################################################

# Placeholder for y plot values
ploty = None

# Init movie frame counter
frameCounter = None

# Define the relation between previous fits and current fit
bestFitWeight = 0.6
bestFitWeight2 = 0.8

# Global instance for each lane line
leftLine = None
rightLine = None

# Threshold for radius of curvature to be treated as straight line
straightLineThreshold = 4000

# Define conversions in x and y from pixels space to meters
ym_per_pix = 40/720 # meters per pixel in y dimension
xm_per_pix = 3.7/660 # meters per pixel in x dimension

# Debug output to show intermediate pipeline steps
enableDebugVid = False
enableDebugImg = False
debugImgFile   = None





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
    # Init globals
    frameCounter = 0
    leftLine = Line()
    rightLine = Line()

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

    if file.endswith('mp4'):
        if setDebug is True:
            enableDebugVid = True
            outputExt = '_lane_vehicle_debug.mp4'
        else:
            outputExt = '_lane_vehicle.mp4'
        video_output = file.replace('.mp4',outputExt)

        clip1 = VideoFileClip(file)

        clip = clip1.fl_image(pipeline)
        clip.write_videofile('output_images/' + video_output, audio=False)

    elif file.endswith('jpg'):
        img = mpimg.imread(file)
        fName = file.replace('test_images/','')
        if setDebug is True:
            enableDebugImg = True
            debugImgFile = 'output_images/' + fName
        finalImg = pipeline(img)
        cv2.imwrite('output_images/' + fName.replace('.jpg','_lane_vehicle.jpg'),
                    cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
