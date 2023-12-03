#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import time
import copy
import shutil
import argparse

from glob import glob
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict 

import random
import numpy as np
import pandas as pd

import cv2
from scipy.misc import toimage
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image, ImageDraw, ImageFont


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)


class TemplateDetector:

    _defaults = {
        "threshold": 0.83,
        "templates_dir": os.path.join(dir_path, "Templates/{}/"),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name `{}`".format(n)

    def __init__(self, **kwargs):

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("\n\n\n[TemplateDetector] Initializing\n\n\n")

        assert "process" in kwargs.keys(), "Arguments MUST contain `process`"

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.process = self.process.split('_')[0]

        self.templates_dir = self.templates_dir.format(self.process)
        self.templates_pool = self.generate()
        self.templates_pool = OrderedDict(sorted(self.templates_pool.items()))

    def generate(self):

        print("Loading templates ...")
        templates_pool = dict()
        for template_path in os.listdir(self.templates_dir):
            print("\t-->", template_path)
            template_name = template_path.split('.png')[0]
            templates_pool[template_name] = cv2.imread(self.templates_dir+template_path, 0)
        return templates_pool

    def locate(self, image, visual_check=False):

        image = np.asarray(image)
        
        button_bboxes = {}
        button_confidences = {}

        # Visual check
        if visual_check:
            canvas = deepcopy(image)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        idx = 0
        for template, template_gray in self.templates_pool.items():
            idx += 1
            # template = "leftClick_" + template
            template = template.split("_")[0]
            if template in button_bboxes.keys():
            	continue
            matches = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            max_conf = np.max(matches)
            if max_conf < self.threshold:
                continue
            candidates = np.where(matches>max_conf*0.97)
            topleft = np.max(candidates, axis=1)
            H, W = template_gray.shape[:2]
            top, left = list(topleft)
            button_bboxes[template] = [top, left, top+H, left+W]
            button_confidences[template] = max_conf
            # print('\t', template, button_bboxes[template])

            # Visual check
            if visual_check:
                cv2.rectangle(canvas, (left, top), (left+W, top+H), (255,0,0), 2)
                cv2.putText(canvas, template, (left+W+3, top+H-3), 0, 0.7, (0,0,255))

        # Visual check
        if visual_check:
            cv2.imwrite(os.path.join(dir_path, 'Environments\\test.png'), canvas)
            _ = input("Press any key to continue ")

        return button_bboxes, button_confidences

