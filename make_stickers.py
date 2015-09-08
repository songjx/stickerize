#!/usr/bin/env python
# make some stickers

import os
import stickerize as stk

img_dir = './test-images'
walk = os.walk(img_dir)
stickers = []

for roots, dirs, files, in walk:
    for img in files:
        stickers.append(stk.singleSticker(os.path.join(img_dir,img)))
        #print os.path.join(img_dir,img)

for sticker in stickers:
    sticker.pad_img()
    sticker.make_blob()
    sticker.make_svg()

stk.plots(stickers)
