#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
import json
import time
import datetime

from tqdm import tqdm
from glob import glob
# from copy import deepcopy

import cv2
from PIL import Image
import numpy as np
from scipy.spatial import distance


from Agents.agent import *


# Read the configuration file
with open('RL_config_UTL.json', 'r') as reader:
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

model_xml = "test_IR\\D3RQN_UTL.xml"
model_bin = "test_IR\\D3RQN_UTL.bin"

plugin = IEPlugin(device="CPU")
net = IENetwork(model=model_xml, weights=model_bin)
exec_net = plugin.load(network=net)
print("Inputs:", net.inputs)
print("Outputs:", net.outputs)
_ = input("Press any key to continue ")

# use_device = 'cpu'
# exec_net = cv2.dnn.readNet(model_bin, model_xml)
# if use_device.lower() in ["ncs", "vpu", "myriad"]:
#     BACKEND = cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE
#     TARGET = cv2.dnn.DNN_TARGET_MYRIAD
# elif use_device.lower() in ["cuda", "gpu"]:
#     BACKEND = cv2.dnn.DNN_BACKEND_CUDA
#     TARGET = cv2.dnn.DNN_TARGET_CUDA
# else:
#     BACKEND = cv2.dnn.DNN_BACKEND_OPENCV
#     TARGET = cv2.dnn.DNN_TARGET_CPU
# exec_net.setPreferableBackend(BACKEND)
# exec_net.setPreferableTarget(TARGET) 
# outputLayers = exec_net.getUnconnectedOutLayersNames()
# print("output layers:", outputLayers)
# inputLayers = exec_net.getLayerInputs()
# print("input layers:", inputLayers)


layer_preContexts = "import/primaryQN/preContextsFlattened"
layer_hiddenCellIn = "import/primaryQN/LSTMCellZeroState/zeros"
layer_hiddenStateIn = "import/primaryQN/LSTMCellZeroState/zeros_1"

layer_Contexts = "import/primaryQN/Contexts"
layer_hiddenCellOut = "import/primaryQN/LSTM_hidden_cell_output"
layer_hiddenStateOut = "import/primaryQN/LSTM_hidden_state_output"


""" Test the network """
test_idx = 0
testing = True
while testing:
    test_idx += 1
    print("\n\t\t[TEST]", test_idx)

    CMD = input("Press any key to test ")
    if CMD in ['c', 'q', "Cancel", "Quit"]:
        testing = False
        continue

    # Get 1st new observation
    agent.be_ready()

    time_keeper = dict()
    time_keeper['ckpt'] = []
    time_keeper['ir'] = []

    cell_state_ir = np.zeros([1, agent.lstm_units], dtype=float)
    hidden_state_ir = np.zeros([1, agent.lstm_units], dtype=float)

    current_moves = 0
    RESET = False
    while agent.on_duty and current_moves<MAX_MOVES and not RESET:

        cmd = input("\n\n\nInsert path to image: ")
        if cmd in ['c', 'q', "Cancel", "Quit"]:
            RESET = True
            continue
        elif os.path.isfile(cmd):
            image_path = cmd
        else:
            continue

        current_moves += 1
        print("\nMOVE {} -------".format(current_moves))
        
        raw_current_state = Image.open(image_path)
        current_contexts, button_centers = agent.extract_contexts(raw_current_state)

        
        ### CKPT model ###
        t1 = time.time()
        action_ckpt, post_contexts, ckpt_state = agent.query(current_contexts, return_all=True)
        action_id = action_ckpt[0]
        t2 = time.time()


        ### IR model ###
        outputs = exec_net.infer(inputs={
            layer_preContexts: current_contexts.reshape((1,1,248)),
            layer_hiddenCellIn: cell_state_ir, 
            layer_hiddenStateIn: hidden_state_ir, 
        })
        # current_contexts = np.reshape(current_contexts, [agent.lstm_units, 1])
        # exec_net.setInput(current_contexts, layer_preContexts)
        # exec_net.setInput(inputs_blob)
        # outputs = exec_net.forward(outputLayers)
        contexts_ir = outputs[layer_Contexts]
        cell_state_ir = outputs[layer_hiddenCellOut]
        hidden_state_ir = outputs[layer_hiddenStateOut]
        qbest_ir = agent.sess.run(
            agent.primeQN.Qbest, 
            feed_dict={
                agent.primeQN.rnn: contexts_ir.reshape((1, agent.lstm_units)),
                agent.primeQN.trainLength: 1,
                agent.primeQN.batch_size: 1
            }
        )
        t3 = time.time()


        ### Record computational time ###
        time_keeper['ckpt'].append(t2-t1)
        time_keeper['ir'].append(t3-t2)


        ### Compare result ###
        print("Action\n\tCKPT: {}\n\tIR: {}".format(action_ckpt, qbest_ir))
        print("Post-Contexts Difference:", distance.euclidean(post_contexts, contexts_ir))
        print("LSTM hidden state Difference:", distance.euclidean(ckpt_state.h, hidden_state_ir))
        print("LSTM cell state Difference:", distance.euclidean(ckpt_state.c, cell_state_ir))


        """ AGENT practices an ACTION to the ENVIRONMENT """
        print("Agent does", agent.actions_list[action_id])


with open(model_xml.replace('xml', 'log'), 'w') as f_handler:
    logger = csv.writer(f_handler, delimiter=',')
    logger.writerow(['step', 'ckpt', 'ir'])
    for idx, (t_ckpt, t_ir) in enumerate(zip(time_keeper['ckpt'], time_keeper['ir'])):
        logger.writerow([idx+1, t_ckpt, t_ir])

