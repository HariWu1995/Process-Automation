

import numpy as NP
import tensorflow as TF
import keras.backend as K

# RL
import sys, os, cv2, time
import numpy as np, math
from argparse import ArgumentParser

try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin


model_xml = r"pb_models\\D3RQN_SCS.xml"
model_bin = r"pb_models\\D3RQN_SCS.bin"

# Test model
plugin = IEPlugin(device="MYRIAD")
net = IENetwork(model=model_xml, weights=model_bin)

print("net.inputs", net.inputs)
print("net.outputs", net.outputs)

print(net.inputs["LSTMCellZeroState/zeros_1"].shape)
print(net.inputs["LSTMCellZeroState/zeros"].shape)
print(net.inputs["primeQN_preContexts"].shape)

print(net.outputs["primeQN_Contexts"].shape)
print(net.outputs["add"].shape)
print(net.outputs["add_1"].shape)

# input_blob = next(iter(net.inputs))
print(input_blob)
exec_net = plugin.load(network=net)

# Read the configuration file 
import json
with open('RL_config_thorough.json', 'r') as reader:
    Config = json.load(reader)

# from Agents.agent import *
# agent = Agent(process=Config['Process'], 
#               use_text=Config['UseText'], 
#               use_icon=Config['UseIcon'],
#               mode='inference')

K.clear_session()
TF.reset_default_graph()

gFile = TF.gfile.FastGFile("./pb_models/D3RQN_SCS.pb", 'rb')
graph_def = TF.GraphDef()

# Parses a serialized binary message into the current message.
graph_def.ParseFromString(gFile.read())
gFile.close()

sess_pb = TF.Session()
sess_pb.graph.as_default()
TF.import_graph_def(graph_def)

# logger = TF.summary.FileWriter("./logs/SCS_IR")
# logger.add_graph(sess_pb.graph)

# print("\n\n\nDONE")
# quit()


# Define input & output tensors
preContexts_tensor = sess_pb.graph.get_tensor_by_name('import/primeQN_preContexts:0')
LSTM_cell_input_tensor = sess_pb.graph.get_tensor_by_name('import/LSTMCellZeroState/zeros:0')
LSTM_hidden_input_tensor = sess_pb.graph.get_tensor_by_name('import/LSTMCellZeroState/zeros_1:0')

Contexts_tensor = sess_pb.graph.get_tensor_by_name('import/primeQN_Contexts:0')
LSTM_cell_output_tensor = sess_pb.graph.get_tensor_by_name('import/add_1:0')
LSTM_hidden_output_tensor = sess_pb.graph.get_tensor_by_name('import/add:0')

from PIL import Image

test_idx = 0
testing = True
while testing:
    test_idx += 1
    print("\n\t\t[TEST]", test_idx)

    _ = input("Press any key to test ")

    # Get 1st new observation
    agent.be_ready()
    
    cell_state = np.zeros([1, agent.lstm_units])
    hidden_state = np.zeros([1, agent.lstm_units])

    lstm_cell_state = np.zeros([1, agent.lstm_units])
    lstm_hidden_state = np.zeros([1, agent.lstm_units])

    current_moves = 0
    while agent.on_duty or current_moves>MAX_MOVES:
        current_moves += 1
        print("\nMOVE {} -------".format(current_moves))
        image_path = input("Insert path to image: ")

        raw_current_state = Image.open(image_path)
        current_contexts, _ = agent.extract_contexts(raw_current_state)
        action_batch_ckpt, post_contexts, LSTM_state = agent.query(current_contexts, return_all=True)
        print("\n\n\n[CKPT]\n\t{}\n\t{}\n\t{}\n\t{}".format(
        	action_batch_ckpt, post_contexts, LSTM_state[0], LSTM_state[1])
        )

        contexts, lstm_cell_state, lstm_hidden_state = sess_pb.run(
            [Contexts_tensor, LSTM_cell_output_tensor, LSTM_hidden_output_tensor],
            feed_dict={
                preContexts_tensor: [current_contexts],
                LSTM_cell_input_tensor: lstm_cell_state.reshape((1,12)),
                LSTM_hidden_input_tensor: lstm_hidden_state.reshape((1,12)),
            }
        )
        pb_lstm_state = NP.hstack([lstm_hidden_state, lstm_cell_state])
        lstm_cell_state = pb_lstm_state[0, :]
        lstm_hidden_state = pb_lstm_state[1, :]
        print("\n\n\n[Frozen]\n\t{}\n\t{}\n\t{}".format(contexts, lstm_cell_state, lstm_hidden_state))

        outputs = exec_net.infer(inputs={
            "primeQN_preContexts": [current_contexts],
            "LSTMCellZeroState/zeros": cell_state.reshape((1,12)),
            "LSTMCellZeroState/zeros_1": hidden_state.reshape((1,12))
        })
        ir_lstm_state = NP.hstack([outputs["add"], outputs["add_1"]])
        cell_state = ir_lstm_state[0, :]
        hidden_state = ir_lstm_state[1, :]
        print("\n\n\n[NCS2]\n\t{}\n\t{}\n\t{}".format(outputs["primeQN_Contexts"], cell_state, hidden_state))















