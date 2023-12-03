import os

import sys
# sys.path.append('..')

import cv2
import numpy as np


def load_list(text_file):
    if not os.path.isfile(text_file):
        print("[LinkError] Can NOT find file", text_file)
        return None
    reader = open(text_file, 'r')
    data = [t.replace('\n', '') for t in reader.readlines()]
    reader.close()
    return data


def load_dict(json_file):
    if not os.path.isfile(json_file):
        print("[LinkError] Can NOT find file", json_file)
        return None
    import json
    reader = open(json_file, 'r')
    data = json.loads(reader.read())
    reader.close()
    return data

  
def resize_image(im, max_side_len=2400):
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len)/resize_h if resize_h>resize_w else float(max_side_len)/resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h*ratio)
    resize_w = int(resize_w*ratio)

    resize_h = resize_h if resize_h%32==0 else (resize_h//32-1)*32
    resize_w = resize_w if resize_w%32==0 else (resize_w//32-1)*32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)



