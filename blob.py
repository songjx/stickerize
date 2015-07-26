#!/usr/bin/env python
# generate blobbed binary image for sticker-friendly outlines

import cv2
import numpy as np
np.set_printoptions(threshold='nan') # print whole arrays, for debugging

#imgName = 'plastic-domed-sight'
#imgName = 'door-closed-sign'
imgName = '10-ton-set'

origImg = cv2.imread('test-images/'+imgName+'.png',-1) # load image as is (with alpha channel)
print origImg.shape
print origImg.dtype
h_orig,w_orig,c_orig = origImg.shape

if origImg.shape[2] == 4: # if image has an alpha channel
    imgAlpha = origImg[:,:,3].copy()
    comparisonImg = np.zeros((h_orig,2*w_orig,c_orig),origImg.dtype) # blank img for side by side comparison
    comparisonImg[:,0:w_orig] = origImg 
    for c in range(3):
        comparisonImg[:,w_orig:,c] = imgAlpha # set pixels white where origImg wasn't transparent
    comparisonImg[:,w_orig:,3] = 255
    cv2.imshow('comparison, '+imgName,comparisonImg)
else: 
    print "original image has no alpha channel"

cv2.waitKey(0) # wait indefinitely for keystroke
cv2.destroyAllWindows()

