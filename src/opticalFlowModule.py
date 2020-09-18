import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import subprocess
import os

def opticalFlow(im1, im2, H_W):
    outputPath = os.path.abspath(".")+"/"
    proc = subprocess.Popen(["./cpp/opticalFlow", im1, im2, outputPath, str(H_W)])
    proc.wait()
    output = cv.imread(outputPath+"mask.png")
    os.remove(outputPath+"img.png")
    os.remove(outputPath+"mask.png")
    return output



def lucas_kanade(inputImage1, inputImage2, H_W):
    im1 = cv.imread(inputImage1)
    im2 = cv.imread(inputImage2)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 2000,
                        qualityLevel = 0.2,
                        minDistance = 5,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    im1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(im1_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(im1)

    im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (c,d),(a,b), (255,255,255), 2)
        mask = cv.circle(mask,(a,b),5, (127,127,127),-1)
    img = cv.add(im1,mask)
    mask = cv.resize(mask, (H_W, H_W) , interpolation = cv.INTER_AREA)
    return mask
