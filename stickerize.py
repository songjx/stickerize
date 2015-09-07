import cv2
import math
import os
import numpy as np

class singleSticker:
    def __init__(self, img_path):
        self.orig_img = cv2.imread(img_path,-1)
        self.paths = {'orig_img': img_path}
        self.info = {'name': os.path.basename(img_path)}
        self.info.update(dict(zip(['h_orig', 'w_orig', 'c_orig'], self.orig_img.shape)))

        # make all-opaque alpha channel if absent
        if self.info['c_orig'] == 3:
            self.dummy_alpha = 255 * np.ones(
                    (self.info['h_orig'],self.info['w_orig'],1),
                    self.orig_img.dtype)
            self.orig_img = cv2.merge((orig_img,dummy_alpha))

    def pad_img(self, padded_dir = './padded-images'):
        """Pad the image and save it."""
        self.paths['padded_img'] = os.path.join(padded_dir,self.info['name'])
        self.info['w_pad'] = \
                int(10 + .5 * math.sqrt(max(self.info['h_orig'], self.info['w_orig'])))
        self.padded_img = cv2.copyMakeBorder(
                self.orig_img,
                self.info['w_pad'],
                self.info['w_pad'],
                self.info['w_pad'],
                self.info['w_pad'],
                cv2.BORDER_CONSTANT,
                value=(255,255,255,0))
        self.info['h_padded'], self.info['w_padded'], _ = self.padded_img.shape
        cv2.imwrite(os.path.join(padded_dir, self.info['name']), self.padded_img)

    def make_blob(self, blob_dir = './blobs', comparison_dir = './comparison-images'):
        """Extract, condition, compare, and save binary blob from padded image."""
        # set paths
        self.paths['blob'] = blob_dir
        self.paths['comparison'] = comparison_dir

        # extract alpha channel
        self.img_alpha = self.padded_img[:,:,3].copy()
        
        # dilate alpha image
        self.dil_alpha = self.img_alpha[:,:].copy()
        _,contours,hier = cv2.findContours(
                self.dil_alpha,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(self.dil_alpha, [cnt], 0, 150, 2*self.info['w_pad'])

        # fill holes in dilated alpha image
        self.filled_alpha = self.dil_alpha[:,:].copy()
        _,contours,hier = cv2.findContours(
                self.filled_alpha,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours: 
            cv2.drawContours(self.filled_alpha, [cnt], 0, 200, -1)

        # illustrate blob changes in different shades of gray
        self.pretty_blob = np.zeros(
                (self.info['h_padded'], self.info['w_padded'],4), 
                self.orig_img.dtype)
        for c in range(3):
            self.pretty_blob[:,:,c] = cv2.bitwise_or(
                    self.img_alpha,
                    cv2.bitwise_xor(self.dil_alpha, self.filled_alpha))
        _, self.pretty_blob[:,:,3] = cv2.threshold(
                self.pretty_blob[:,:,0], 0, 255, cv2.THRESH_BINARY)
        
        # make side-by-side comparison image
        self.comp_img = np.zeros((
                self.info['h_padded'], 2*self.info['w_padded'],4), 
                self.orig_img.dtype)
        self.comp_img[:,0:self.info['w_padded']] = self.padded_img
        self.comp_img[:,self.info['w_padded']:,:] = self.pretty_blob
        cv2.imwrite(os.path.join(comparison_dir, self.info['name']), self.comp_img)

        # write blob to file
        self.blob = self.pretty_blob[:,:,3]
        cv2.imwrite(os.path.join(blob_dir, self.info['name']), self.blob)

    def make_svg():
        """Generate and save svg of single sticker."""
        pass

def sticker_sheet(stickers):
    """Generate a sticker sheet."""
    pass
