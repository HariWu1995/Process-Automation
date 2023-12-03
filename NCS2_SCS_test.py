#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
with open('RL_config_thorough.json', 'r') as reader:
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


try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin

model_xml = r"pb_models\\D3RQN_SCS.xml"
model_bin = r"pb_models\\D3RQN_SCS.bin"
plugin = IEPlugin(device="MYRIAD")
net = IENetwork(model=model_xml, weights=model_bin)
exec_net = plugin.load(network=net)


""" Test the network """
test_idx = 0
testing = True
while testing:
    test_idx += 1
    print("\n\t\t[TEST]", test_idx)

    _ = input("Press any key to test ")

    cell_state = np.zeros([1, agent.lstm_units])
    hidden_state = np.zeros([1, agent.lstm_units])

    # Get 1st new observation
    agent.be_ready()

    current_moves = 0
    while agent.on_duty or current_moves>MAX_MOVES:
        current_moves += 1
        print("\nMOVE {} -------".format(current_moves))

        image_path = input("Insert path to image: ")
        raw_current_state = Image.open(image_path)
        current_contexts, button_centers = agent.extract_contexts(raw_current_state)

        ### CKPT model ###
        # action_ckpt, post_contexts, LSTM_state = agent.query(current_contexts, return_all=True)
        # action_id = action_ckpt[0]
        # action = agent.actions_list[action_id]
        # ckpt_lstm_state = LSTM_state.eval(session=agent.sess)
        # print(LSTM_state.c, LSTM_state.h)

        ### IR model ###
        outputs = exec_net.infer(inputs={
            "primeQN_preContexts": [current_contexts],
            "LSTMCellZeroState/zeros": cell_state.reshape((1,12)),
            "LSTMCellZeroState/zeros_1": hidden_state.reshape((1,12))
        })
        contexts_ir = outputs["primeQN_Contexts"]
        ir_lstm_state = np.hstack([outputs["add"], outputs["add_1"]])
        cell_state = ir_lstm_state[0, :]
        hidden_state = ir_lstm_state[1, :]

        # action_ir = agent.sess.run(
        #     [agent.primeQN.Qbest],
        #     feed_dict={
        #         agent.primeQN.rnn: contexts_ir,
        #         agent.primeQN.trainLength: 1,
        #         agent.primeQN.batch_size: 1
        #     }
        # )

        ### Compare result ###
        # print("Action\n\tCKPT: {}\n\tIR: {}".format(action_ckpt, action_ir[0]))
        # print("LSTM hidden state Difference", LSTM_state.h-ir_lstm_state[1, :])
        # print("LSTM cell state Difference", LSTM_state.c-ir_lstm_state[0, :])


        """ AGENT practices an ACTION to the ENVIRONMENT """
        # try:
        #     action_args = dict()
        #     if action in button_centers.keys():
        #         action_args['click_x'] = button_centers[action][1]
        #         action_args['click_y'] = button_centers[action][0]
        #     if 'label' in action:
        #         action_args['image'] = raw_current_state
        #     print("Agent does", action)
        # except KeyError:
        #     print("\n".join(
        #         "{} - {} - {}".format(idx, button, position) \
        #         for idx, (button, position) in enumerate(button_centers.items())
        #     ))
        #     print("Agent chooses {} not existing in this state!".format(action))

