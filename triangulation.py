import sys
import cv2 
import numpy as np 
import time 

def find_depth(circle_right, circle_left, frameR, frameL, baseline, f, alpha):

    #Convert focal length from mm to pixel
    height_right, width_right, depth_right = frameR.shape 
    height_left, width_left, depth_left = frameL.shape 

    if width_right == width_left:
        f_pixel = (width_right*0.5) / np.tan(alpha*0.5*np.pi/180)

    else:
        print('Left and Right camera frames do not have the same pixel width')

    x_right = circle_right[0]
    x_left = circle_left[0]

    # Calculate disparity (the difference in coordinates of similar features within two stereo images)
    disparity = x_left - x_right
    
    #Calculate Depth in cm along z axis
    zDepth = (baseline*f_pixel)/disparity

    return abs(zDepth)