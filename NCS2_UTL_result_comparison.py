import sys
import os
import time
from copy import deepcopy

import numpy as NP
import tensorflow as TF
import keras.backend as K

import cv2
import math
from scipy.spatial import distance

try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin


layer_preContexts = "import/primaryQN/preContexts"
layer_hiddenCellIn = "import/primaryQN/LSTMCellZeroState/zeros" 
layer_hiddenStateIn = "import/primaryQN/LSTMCellZeroState/zeros_1" 

layer_Qbest = "import/primaryQN/Qbest"
layer_Qvalues = "import/primaryQN/add"
layer_hiddenCellOut = "import/primaryQN/LSTM_hidden_cell_output" #"import/primaryQN/IR_converter/cell_state"
layer_hiddenStateOut = "import/primaryQN/LSTM_hidden_state_output" #"import/primaryQN/IR_converter/hidden_state"


model_xml = r"pb_models\\D3RQN_UTL.xml"
model_bin = r"pb_models\\D3RQN_UTL.bin"

plugin = IEPlugin(device="GPU")
net = IENetwork(model=model_xml, weights=model_bin)

print("net.inputs:\n\t", net.inputs)
print("net.outputs:\n\t", net.outputs)

print(net.inputs[layer_hiddenCellIn].shape)
print(net.inputs[layer_hiddenStateIn].shape)
print(net.inputs[layer_preContexts].shape)

print(net.outputs[layer_Qvalues].shape)
print(net.outputs[layer_Qbest+'/Squeeze'].shape)
print(net.outputs[layer_hiddenCellOut].shape)
print(net.outputs[layer_hiddenStateOut].shape)

_ = input("Press any key to continue ")

# input_blob = next(iter(net.inputs))
# print(input_blob)
exec_net = plugin.load(network=net)

# Read the configuration file 
import json
with open('RL_config_UTL.json', 'r') as reader:
    Config = json.load(reader)

from Agents.agent import *
agent = Agent(process="UTL_ir", #Config['Process'], 
              use_text=Config['UseText'], 
              use_icon=Config['UseIcon'],
              mode='inference')
_ = input("Press any key to continue ")

K.clear_session()
TF.reset_default_graph()

gFile = TF.gfile.FastGFile("./pb_models/D3RQN_UTL.pb", 'rb')
graph_def = TF.GraphDef()

# Parses a serialized binary message into the current message.
graph_def.ParseFromString(gFile.read())
gFile.close()

sess_pb = TF.Session()
sess_pb.graph.as_default()
TF.import_graph_def(graph_def)
    
for t in sess_pb.graph.get_operations():
    print(t.name)

# Define input & output tensors
preContexts_tensor = sess_pb.graph.get_tensor_by_name('import/{}:0'.format(layer_preContexts))
LSTM_cell_input_tensor = sess_pb.graph.get_tensor_by_name('import/{}:0'.format(layer_hiddenCellIn))
LSTM_hidden_input_tensor = sess_pb.graph.get_tensor_by_name('import/{}:0'.format(layer_hiddenStateIn))

action_tensor = sess_pb.graph.get_tensor_by_name('import/{}:0'.format(layer_Qbest))
LSTM_cell_output_tensor = sess_pb.graph.get_tensor_by_name('import/{}:0'.format(layer_hiddenCellOut))
LSTM_hidden_output_tensor = sess_pb.graph.get_tensor_by_name('import/{}:0'.format(layer_hiddenStateOut))

print(preContexts_tensor)
print(LSTM_cell_input_tensor)
print(LSTM_hidden_input_tensor)

_ = input("Press any key to continue ")


from PIL import Image

test_idx = 0
testing = True
while testing:
    test_idx += 1
    print("\n\t\t[TEST]", test_idx)

    _ = input("Press any key to test ")

    # Get 1st new observation
    agent.be_ready()

    init_state = NP.zeros([1, agent.lstm_units], dtype=float)
    
    cell_state_ir = deepcopy(init_state)
    hidden_state_ir = deepcopy(init_state)

    cell_state_pb = deepcopy(init_state)
    hidden_state_pb = deepcopy(init_state)

    current_moves = 0
    while agent.on_duty:
        current_moves += 1
        print("\nMOVE {} -------".format(current_moves))
        cmd = input("Insert path to image: ")
        if cmd in ['c', 'q', "Cancel", "Quit"]:
            break
        else:
            image_path = cmd

        if not os.path.isfile(image_path):
            continue

        cell_state_ir, hidden_state_ir = cell_state_pb, hidden_state_pb

        raw_current_state = Image.open(image_path)
        current_contexts, _ = agent.extract_contexts(raw_current_state)
        action_ckpt, _, ckpt_state, \
        hidden_state_ckpt, cell_state_ckpt = agent.query(current_contexts, return_all=True)
        NP.savetxt("__TEST__\\ckpt_lstm_h.txt", ckpt_state.h, fmt='%1.2e')
        NP.savetxt("__TEST__\\ckpt_lstm_c.txt", ckpt_state.c, fmt='%1.2e')
        NP.savetxt("__TEST__\\ckpt_cell.txt", cell_state_ckpt, fmt='%1.2e')
        NP.savetxt("__TEST__\\ckpt_hidden.txt", hidden_state_ckpt, fmt='%1.2e')
        # print("\n\n\n[CKPT]\n\t{}\n\t{}\n\t{}".format(
        #     action_ckpt, ckpt_state.c, ckpt_state.h)
        # )

        action_pb, cell_state_pb, hidden_state_pb = sess_pb.run(
            [action_tensor, LSTM_cell_output_tensor, LSTM_hidden_output_tensor],
            feed_dict={
                preContexts_tensor: [current_contexts],
                LSTM_cell_input_tensor: cell_state_pb, 
                LSTM_hidden_input_tensor: hidden_state_pb, 
            }
        )
        NP.savetxt("__TEST__\\pb_hidden.txt", hidden_state_pb, fmt='%1.2e')
        NP.savetxt("__TEST__\\pb_cell.txt", cell_state_pb, fmt='%1.2e')
        # print("\n\n\n[Frozen]\n\t{}\n\t{}\n\t{}".format(
        #     action_pb, cell_state_pb, hidden_state_pb)
        # )

        outputs = exec_net.infer(inputs={
            layer_preContexts: [current_contexts],
            layer_hiddenCellIn: cell_state_ir, 
            layer_hiddenStateIn: hidden_state_ir, })
        qbest_ir = outputs[layer_Qbest+'/Squeeze']
        action_ir = NP.argmax(outputs[layer_Qvalues])
        cell_state_ir = outputs[layer_hiddenCellOut]
        hidden_state_ir = outputs[layer_hiddenStateOut]
        NP.savetxt("__TEST__\\ir_hidden.txt", hidden_state_ir, fmt='%1.2e')
        NP.savetxt("__TEST__\\ir_cell.txt", cell_state_ir, fmt='%1.2e')
        # print("\n\n\n[NCS2]\n\t{}\n\t{}\n\t{}".format(
        #     action_ir, cell_state_ir, hidden_state_ir)
        # )

        print("\n\n\nAction\n\tCKPT: {}\n\tPB: {}\n\tIR_Q: {}\n\tIR_A: {}".format(
            action_ckpt, action_pb, qbest_ir, action_ir))
        # print("\n\n\nLSTM Cell state\n\tCKPT-PB: {}\n\tPB-IR: {}\n\tIR-CKPT: {}".format(
        #     distance.euclidean(ckpt_state.c, cell_state_pb), 
        #     distance.euclidean(cell_state_pb, cell_state_ir), 
        #     distance.euclidean(cell_state_ir, ckpt_state.c)))
        # print("\n\n\nLSTM Hidden state\n\tCKPT-PB: {}\n\tPB-IR: {}\n\tIR-CKPT: {}".format(
        #     distance.euclidean(ckpt_state.h, hidden_state_pb), 
        #     distance.euclidean(hidden_state_pb, hidden_state_ir), 
        #     distance.euclidean(hidden_state_ir, ckpt_state.h)))
        print("\n\n\nCell state\n\tCKPT-PB: {}\n\tPB-IR: {}\n\tIR-CKPT: {}".format(
            distance.euclidean(cell_state_ckpt, cell_state_pb), 
            distance.euclidean(cell_state_pb, cell_state_ir), 
            distance.euclidean(cell_state_ir, cell_state_ckpt)))
        print("\n\n\nHidden state\n\tCKPT-PB: {}\n\tPB-IR: {}\n\tIR-CKPT: {}".format(
            distance.euclidean(hidden_state_ckpt, hidden_state_pb), 
            distance.euclidean(hidden_state_pb, hidden_state_ir), 
            distance.euclidean(hidden_state_ir, hidden_state_ckpt)))
        # print("\n\n\nLSTM Cell state\n\tCKPT-PB: {}\n\tPB-IR: {}\n\tIR-CKPT: {}".format(
        #     ckpt_state.c-cell_state_pb, 
        #     cell_state_pb-cell_state_ir, 
        #     cell_state_ir-ckpt_state.c))
        # print("\n\n\nLSTM Hidden state\n\tCKPT-PB: {}\n\tPB-IR: {}\n\tIR-CKPT: {}".format(
        #     ckpt_state.h-hidden_state_pb, 
        #     hidden_state_pb-hidden_state_ir, 
        #     hidden_state_ir-ckpt_state.h))
        # print("\n\n\nCell state\n\tCKPT-PB: {}\n\tPB-IR: {}\n\tIR-CKPT: {}".format(
        #     cell_state_ckpt-cell_state_pb, 
        #     cell_state_pb-cell_state_ir, 
        #     cell_state_ir-cell_state_ckpt))
        # print("\n\n\nHidden state\n\tCKPT-PB: {}\n\tPB-IR: {}\n\tIR-CKPT: {}".format(
        #     hidden_state_ckpt-hidden_state_pb, 
        #     hidden_state_pb-hidden_state_ir, 
        #     hidden_state_ir-hidden_state_ckpt))

        # cell_state_pb, hidden_state_pb = hidden_state_pb, cell_state_pb
        # cell_state_ir, hidden_state_ir = hidden_state_ir, cell_state_ir
        # hidden_state_ckpt, cell_state_ckpt = cell_state_ckpt, hidden_state_ckpt














