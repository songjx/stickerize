#!/usr/bin/env python
# generate blobbed binary image for sticker-friendly outlines

import cv2
import numpy as np
import os
np.set_printoptions(threshold='nan') # print whole arrays, for debugging

imgDir = './test-images'
walk = os.walk(imgDir)

for roots, dirs, files in walk:
    for img in files:
        imgPath = os.path.join(os.path.relpath(roots),img)
        imgName = os.path.basename(img)
        # load image as is (with alpha channel)
        origImg = cv2.imread(imgPath,-1) 
        h_orig,w_orig,c_orig = origImg.shape
        # blank img for side by side comparison
        comparisonImg = np.zeros(
            (h_orig,2*w_orig,c_orig),
            origImg.dtype) 
        comparisonImg[:,0:w_orig] = origImg 
        rightHalf = comparisonImg[:,w_orig:,:]

        # if image has an alpha channel
        if origImg.shape[2] == 4: 
            imgAlpha = origImg[:,:,3].copy()
            filledAlpha = imgAlpha[:,:].copy()

            # fill holes
            _, contours,hier = cv2.findContours(
                    filledAlpha,
                    cv2.RETR_CCOMP,
                    cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(filledAlpha,[cnt],0,200,-1)

            # set rightHalf pixels gray for filledAlpha and white for imgAlpha
            for c in range(3):
                rightHalf[:,:,c] = cv2.bitwise_or(imgAlpha,filledAlpha)
        else: 
            print imgName + " has no alpha channel"
            for c in range(3):
                rightHalf[:,:,c] = 255
        
        cv2.imshow('comparison, '+imgName,comparisonImg)
        #cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()

