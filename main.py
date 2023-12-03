#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import ntpath

import time
import datetime

from tqdm import tqdm
from glob import glob
from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np

from Agents.agent import *
from Labeller.labeller import *
from Utilities.Template_Matching_Utils import crop_defect_zone


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)


# Read the configuration file
with open(dir_path+'\\RL_config_thorough.json', 'r') as reader:
    Config = json.load(reader)

with open(dir_path+"\\defect_code.json", 'r') as reader:
    Defects_Code = json.load(reader)


class SCS_Operator:

    def __init__(self):

        self.mode = "setup"
        self.defect_zone = (50, 180, 900, 1020)
        self.current_defect_feature = np.zeros(768, dtype=int)

        self.Configurer = Agent(process=Config['Process'],
                                use_text=Config['UseText'],
                                use_icon=Config['UseIcon'],
                                optimizer=Config['OptimizerQN'],
                                mode='inference')
        self.Labeller = Labeller()

    def query_RL(self, current_state):
        current_contexts, \
        button_centers = self.Configurer.extract_contexts(current_state)
        action_batch = self.Configurer.query(current_contexts)
        action_id = action_batch[0]
        action = self.Configurer.actions_list[action_id]
        return action, button_centers

    def propose_action(self, current_state):

        if self.mode == "setup":
            action, button_centers = self.query_RL(current_state)
            if any(a in action for a in ['click', 'Click']):
                action = icon2position(action, button_centers)
            elif any(a in action for a in ['label', "Label"]):
                self.mode = "label"
        if self.mode == "label":
            self.previous_defect_feature = self.current_defect_feature
            self.current_defect_feature = np.asarray(
                current_state.crop(self.defect_zone).histogram()
            )
            if compare_features(self.current_defect_feature,
                                self.previous_defect_feature) < 97:
                self.mode = "setup"
                action, button_centers = self.query_RL(current_state)
                if any(a in action for a in ['click', 'Click']):
                    action = icon2position(action, button_centers)
            else:
                defect_type = self.Labeller.label_defect(
                    np.asarray(current_state.crop(self.defect_zone))
                )
                action = "strikeKey_" + Defects_Code[defect_type]
        return action


def icon2position(action, centers):
        action_cmd, button_name = action.split('_')[:2]
        action_cmd += "_h{}_w{}"
        if action in centers.keys():
            h, w = centers[action]
        else:
            h, w = 0, 0
        return action_cmd.format(h, w)


def compare_features(feature1, feature2):
    if len(feature1) != len(feature2):
        print("features MUST share the same shape")
        print(len(feature1))
        print(len(feature2))
        return None
    return np.sqrt(
        np.mean((feature1.flatten()-feature2.flatten())**2)
    )


if __name__ == "__main__":

    Operator = SCS_Operator()
    Operator.Configurer.be_ready()

    while True:
        image_path = input("Insert image path: ")
        current_state = Image.open(image_path)
        action = Operator.propose_action(current_state)

        # with open(image_path+'.txt', 'w') as writer:
        #     writer.write(action)
        print('\t-->', action)
