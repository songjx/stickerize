#!/usr/bin/env python
# generate blobbed binary image for sticker-friendly outlines

import cv2
import numpy as np
import os
import math
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

        # make all-opaque alpha channel if absent
        if origImg.shape[2] < 4:
            dummyAlpha = 255 * np.ones(
                    (h_orig,w_orig,1),
                    origImg.dtype)
            origImg = cv2.merge((origImg,dummyAlpha))
        
        # pad image
        w_pad = int(10 + .5 * math.sqrt(max(h_orig, w_orig)))
        paddedImg = cv2.copyMakeBorder(
                origImg,
                w_pad,w_pad,w_pad,w_pad,
                cv2.BORDER_CONSTANT,
                value=(255,255,255,0))
        h_padded,w_padded,_ = paddedImg.shape

        # extract alpha channel
        imgAlpha = paddedImg[:,:,3].copy()

        # create image for side by side comparison
        comparisonImg = np.zeros(
            (h_padded,2*w_padded,4),
            origImg.dtype) 
        comparisonImg[:,0:w_padded] = paddedImg 
        rightHalf = comparisonImg[:,w_padded:,:]

        # created dilated alpha image
        dilAlpha = imgAlpha[:,:].copy()
        _, contours,hier = cv2.findContours(
                dilAlpha,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(dilAlpha,[cnt],0,150,2*w_pad)

        # fill holes in dilated alpha image
        filledAlpha = dilAlpha[:,:].copy()
        _, contours,hier = cv2.findContours(
                filledAlpha,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(filledAlpha,[cnt],0,200,-1)

        # merge results into rightHalf image
        for c in range(3):
            rightHalf[:,:,c] = cv2.bitwise_or(
                    imgAlpha,
                    cv2.bitwise_xor(dilAlpha,filledAlpha))
        _,rightHalf[:,:,3] = cv2.threshold(rightHalf[:,:,0],0,255,cv2.THRESH_BINARY)

        # save rightHalf alpha channel as blob
        blob = rightHalf[:,:,3]
       
        # show original and blobbed images side by side
        cv2.imshow('comparison, '+imgName,comparisonImg)
        #cv2.waitKey(0)

        # write padded images and blobs to file
        cv2.imwrite(os.path.join('./comparison-images',imgName),comparisonImg)
        cv2.imwrite(os.path.join('./padded-images',imgName),paddedImg)
        cv2.imwrite(os.path.join('./blobs',imgName),blob)

cv2.waitKey(0)
cv2.destroyAllWindows()

