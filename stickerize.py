import cv2
import math
import os
import numpy as np
import subprocess as sbp
import matplotlib.pyplot as plt

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
            self.orig_img = cv2.merge((self.orig_img,self.dummy_alpha))

    def pad_img(self, padded_dir = './padded-images'):
        """Pad the image and save it."""
        self.paths['padded_img'] = os.path.join(padded_dir,self.info['name'])
        
        # dilating width
        self.info['w_dil'] = \
                int(10 + .4 * math.sqrt(max(self.info['h_orig'], self.info['w_orig'])))

        # kernel width, for later closing
        self.info['w_kernel'] = int(self.info['w_dil'] * 2)

        # padding width
        self.info['w_pad'] = self.info['w_dil'] + 2*self.info['w_kernel']
        
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
        self.paths['blob'] = os.path.join(blob_dir, self.info['name'])
        self.paths['comparison'] = os.path.join(comparison_dir, self.info['name'])

        # extract alpha channel from padded image
        self.img_alpha = self.padded_img[:,:,3].copy()
        
        # dilate alpha image
        self.dil_alpha = self.img_alpha[:,:].copy()
        _,contours,hier = cv2.findContours(
                self.dil_alpha,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(self.dil_alpha, [cnt], 0, 255, 2*self.info['w_dil'])

        # perform closing operation on dilated alpha image
        self.closed_alpha = self.dil_alpha[:,:].copy()
        self.kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.info['w_kernel'],self.info['w_kernel'])) 
        self.closed_alpha = cv2.morphologyEx(
                self.closed_alpha,
                cv2.MORPH_CLOSE,
                self.kernel)

        # fill holes in closed alpha image
        self.filled_alpha = self.closed_alpha[:,:].copy()
        _,contours,hier = cv2.findContours(
                self.filled_alpha,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours: 
            cv2.drawContours(self.filled_alpha, [cnt], 0, 255, -1)

        # illustrate blob changes in different shades of gray
        self.pretty_blob = np.zeros(
                (self.info['h_padded'], self.info['w_padded'],4), 
                self.orig_img.dtype)
        for c in range(3): # set RBG channels
            self.pretty_blob[:,:,c] = cv2.bitwise_or(
                    self.img_alpha,cv2.bitwise_or(
                            (128./255*self.dil_alpha).astype('uint8'), 
                            cv2.bitwise_xor(
                                (110./255*self.closed_alpha).astype('uint8'), 
                                (30./255*self.filled_alpha).astype('uint8'))))

        # (set A channel)
        _, self.pretty_blob[:,:,3] = cv2.threshold(
                self.pretty_blob[:,:,0], 0, 255, cv2.THRESH_BINARY)
        
        # make side-by-side comparison image
        self.comp_img = np.zeros((
                self.info['h_padded'], 2*self.info['w_padded'],4), 
                self.orig_img.dtype)
        self.comp_img[:,0:self.info['w_padded']] = self.padded_img
        self.comp_img[:,self.info['w_padded']:,:] = self.pretty_blob
        cv2.imwrite(self.paths['comparison'], self.comp_img)

        # write blob to file
        self.blob = self.pretty_blob[:,:,3]
        cv2.imwrite(self.paths['blob'], self.blob)
        
    def make_svg(self, pnm_dir = './pnm-blobs', svg_dir = './svgs'):
        """Generate and save svg of single sticker."""
        self.paths['pnm'] = os.path.join(
                pnm_dir, os.path.splitext(self.info['name'])[0]+'.pnm')
        self.paths['svg'] = os.path.join(
                svg_dir, os.path.splitext(self.info['name'])[0]+'.svg')

        # convert blob to pnm because potrace is picky
        sbp.call(['convert',
                self.paths['blob'],
                self.paths['pnm']])

        # potrace pnm
        sbp.call(['potrace',
                '--svg',
                '--invert',
                '--color=#FFFFFF',
                '-o'+self.paths['svg'],
                self.paths['pnm']])


def plots(stickers, plots_dir = './plots'):
    """Generate some sticker statistics."""
    padded_dims = np.zeros([len(stickers),2])
    for i, sticker in enumerate(stickers):
        padded_dims[i,:] = [sticker.info['h_padded'], sticker.info['w_padded']]

    # scatter plot
    plt.scatter(padded_dims[:,1], padded_dims[:,0])
    plt.title('padded image dimensions')
    plt.xlabel('width (px)')
    plt.ylabel('height (px)')
    plt.savefig(os.path.join(plots_dir,'padded_dims.pdf'))
    #plt.show()

def sticker_sheet(stickers):
    """Generate a sticker sheet."""
    pass
