#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
import datetime

from tqdm import tqdm
from glob import glob
from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np
import matplotlib.pyplot as plt

import math
import operator
from functools import reduce

from Agents.agent import *


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)


# Read the configuration file
with open('RL_config.json', 'r') as reader:
    Config = json.load(reader)

# Assign training hyper-parameters with configuration
PROCESS = Config['Process']
useText = Config['UseText']
useIcon = Config['UseIcon']
MAX_MOVES = Config['MaxMovesPerEpisode']
optimizerQN = Config['OptimizerQN']


agent = Agent(process=PROCESS,
              use_text=useText,
              use_icon=useIcon,
              optimizer=optimizerQN,
              mode='inference')

""" Test the network """
test_idx = 0
testing = True
while testing:
    test_idx += 1
    print("\n\t\t[TEST]", test_idx)

    _ = input("Press any key to test ")

    # Get 1st new observation
    agent.be_ready()

    current_moves = 0
    while agent.on_duty or current_moves>MAX_MOVES:
        current_moves += 1
        print("\nMOVE {} -------".format(current_moves))
        # _ = input("Press any key to take action ")
        raw_current_state = agent.observe()

        start = time.time()
        current_contexts, button_centers = agent.extract_contexts(raw_current_state)
        stop_det = time.time()
        print("Computational Time for Detection:", stop_det-start)

        action_batch = agent.query(current_contexts)
        stop_dec = time.time()
        print("Computational Time for Decision:", stop_dec-stop_det)

        action_id = action_batch[0]
        action = agent.actions_list[action_id]

        """ AGENT practices an ACTION to the ENVIRONMENT """
        try:
            action_args = dict()
            if action in button_centers.keys():
                action_args['click_x'] = button_centers[action][1]
                action_args['click_y'] = button_centers[action][0]
            if 'label' in action:
                action_args['image'] = raw_current_state
            agent.does(action, **action_args)
        except KeyError:
            print("\n".join(
                "{} - {} - {}".format(idx, button, position) \
                for idx, (button, position) in enumerate(button_centers.items())
            ))
            print("Agent chooses {} not existing in this state!".format(action))

        time.sleep(0.19)
