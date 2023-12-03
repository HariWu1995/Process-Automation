#!/usr/bin/env python
# coding: utf-8

from PIL import Image, ImageDraw
from copy import deepcopy

import numpy as NP
import tensorflow as TF
import keras.backend as K


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param 
        session: 
                The TensorFlow session to be frozen.
        keep_var_names: 
                A list of variable names that should not be frozen,
                or None to freeze all the variables in the graph.
        output_names:
                Names of the relevant graph outputs.
        clear_devices:
                Remove the device directives from the graph for better portability.
    @return 
        The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants as vars2consts
    
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in TF.global_variables()).difference(keep_var_names or []))
        print("\n\n\nVariables to be frozen:\n\t", "\n\t".join(v for v in freeze_var_names))

        output_names = output_names or []
        output_names += [v.op.name for v in TF.global_variables()]
        print("\n\n\nOutput tensors:\n\t", "\n\t".join(v for v in output_names))

        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = vars2consts(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
    

from Agents.agent_ir import *


def phase_1():

    K.clear_session()
    TF.reset_default_graph()
    
    for t in agent.sess.graph.get_operations():
        if t.name.startswith('primaryQN'):
            print(t.values())
            # print(t.name)

    frozen_graph = freeze_session(agent.sess,
                                  output_names=['primaryQN/Qbest'])
    TF.train.write_graph(frozen_graph, "pb_models", "D3RQN_phase1.pb", as_text=False)
    TF.train.write_graph(frozen_graph, "pb_models", "D3RQN_phase1.pbtxt", as_text=True)

    logger = TF.summary.FileWriter(".\\logs\\phase_1")
    logger.add_graph(frozen_graph) #agent.sess.graph

    return True


def phase_2():

    K.clear_session()
    TF.reset_default_graph()

    gFile = TF.gfile.FastGFile("./pb_models/D3RQN_phase1.pb", 'rb')
    graph_def = TF.GraphDef()

    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(gFile.read())
    gFile.close()

    sess_pb = TF.Session()
    sess_pb.graph.as_default()
    TF.import_graph_def(graph_def)

    # # Override placeholders with constants
    override_graph_def = TF.GraphDef()
    for node in graph_def.node:
        node_name = node.name
        if any(nn in node_name for nn in ["batchSize", "traceLength"]):
            override_node = TF.constant(1, name=node_name)
            override_node_def = override_node.op.node_def
            print("Convert \n\t{} to \n\t{}".format(node, override_node_def))
            override_graph_def.node.extend([override_node_def])
        elif node_name in ['primaryQN/LSTMCellZeroState/zeros', 
                           'primaryQN/LSTMCellZeroState/zeros_1']:
            override_node = TF.placeholder(dtype=TF.float32, shape=[1, agent.lstm_units], name=node_name)
            override_node_def = override_node.op.node_def
            print("Convert \n\t{} to \n\t{}".format(node, override_node_def))
            override_graph_def.node.extend([override_node_def])
        else:
            override_graph_def.node.extend([deepcopy(node)])

    with TF.gfile.GFile("./pb_models/D3RQN_phase2.pb", "wb") as FileHandler:
        FileHandler.write(override_graph_def.SerializeToString())

    logger = TF.summary.FileWriter(".\\logs\\phase_2")
    logger.add_graph(override_graph_def)

    return True


"""
python "C:/Program Files (x86)/IntelSWTools/openvino_2020.2.117/deployment_tools/model_optimizer/mo_tf.py" \
-m "D:/HARI/__RL_Dev__/__Backend_RL__/pb_models/D3RQN_IR.pb" \
-o "D:/HARI/__RL_Dev__/__Backend_RL__/pb_models" \
--data_type FP32 \
--input "primaryQN/preContexts[1 134]" \
--output primaryQN/Qbest \
--log_level DEBUG
"""


def phase_3():
    
    K.clear_session()
    TF.reset_default_graph()

    gFile = TF.gfile.FastGFile("./pb_models/D3RQN_phase2.pb", 'rb')
    graph_def = TF.GraphDef()

    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(gFile.read())
    gFile.close()

    sess_pb = TF.Session()
    sess_pb.graph.as_default()

    # Import a serialized TF `GraphDef` protocol buffer & place into the current `Graph`.
    TF.import_graph_def(graph_def)

    # for t in sess_pb.graph.get_operations():
    #     print(t.values())

    # Define input & output tensors
    preContexts_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/preContexts:0')
    LSTM_cell_input_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/LSTMCellZeroState/zeros:0')
    LSTM_hidden_input_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/LSTMCellZeroState/zeros_1:0')

    action_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/Qbest:0')
    LSTM_cell_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/rnn_transfer/while/Exit_3:0')
    LSTM_hidden_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/rnn_transfer/while/Exit_4:0')

    mask = TF.zeros(shape=[1, agent.lstm_units], dtype=tf.float32, name="import/primaryQN/mask")
    LSTM_cell_output_tensor = TF.add(LSTM_cell_tensor, mask, name='import/primaryQN/LSTM_hidden_cell_output')
    LSTM_hidden_output_tensor = TF.add(LSTM_hidden_tensor, mask, name='import/primaryQN/LSTM_hidden_state_output')

    frozen_graph = freeze_session(sess_pb,
                                  output_names=['import/primaryQN/Qbest', 
                                                'import/primaryQN/LSTM_hidden_cell_output',
                                                'import/primaryQN/LSTM_hidden_state_output'])
    TF.train.write_graph(frozen_graph, "pb_models", "D3RQN_phase3.pb", as_text=False)
    TF.train.write_graph(frozen_graph, "pb_models", "D3RQN_phase3.pbtxt", as_text=True)

    logger = TF.summary.FileWriter(".\\logs\\phase_3")
    logger.add_graph(frozen_graph) 

    return True


def phase_4():
    
    K.clear_session()
    TF.reset_default_graph()

    override_graph_def = TF.GraphDef()
    with tf.variable_scope("primaryQN"):
        blob_batch = tf.placeholder(shape=[1, 1, agent.lstm_units, 1], 
                                    dtype=tf.float32, 
                                    name='inputs_batch')
        # blob = tf.reshape(blob_batch, shape=[1, agent.lstm_units, 3], name='inputs')

        # contexts, cell_state, hidden_state = tf.split(blob, [1,1,1], axis=2, name='split_inputs')
        contexts = tf.reshape(blob_batch, [1, agent.lstm_units], name='preContexts')
        # cell_state = tf.reshape(cell_state, [1, agent.lstm_units], name='LSTMCellZeroState/zeros')
        # hidden_state = tf.reshape(hidden_state, [1, agent.lstm_units], name='LSTMCellZeroState/zeros_1')

    sess = tf.Session()
    sess.graph.as_default()

    pre_graph_def = sess.graph.as_graph_def()
    for pre_node in pre_graph_def.node:
        override_graph_def.node.extend([pre_node])

    
    K.clear_session()
    TF.reset_default_graph()

    gFile = TF.gfile.FastGFile("./pb_models/D3RQN_phase2.pb", 'rb')
    graph_def = TF.GraphDef()

    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(gFile.read())
    gFile.close()

    sess_pb = TF.Session()
    sess_pb.graph.as_default()

    # Import a serialized TF `GraphDef` protocol buffer & place into the current `Graph`.
    TF.import_graph_def(graph_def)

    # # Override placeholders with constants
    for node in graph_def.node:
        node_name = node.name
        if node_name in ["primaryQN/preContexts", 
                         "primaryQN/LSTMCellZeroState/zeros", 
                         "primaryQN/LSTMCellZeroState/zeros_1"]:
            continue
        # elif node_name in ['primaryQN/rnn_transfer/while/Exit_3', 'primaryQN/rnn_transfer/while/Exit_4']:
        #     override_graph_def.node.extend([deepcopy(node)])
        #     identiy_node = TF.NodeDef()
        #     identiy_node.name = "LSTM_hidden_cell_output" if node_name=='primaryQN/rnn_transfer/while/Exit_3' \
        #                    else "LSTM_hidden_state_output"
        #     identiy_node.op = "Identity"
        #     identiy_node.input.extend([node_name])
        #     identiy_node.attr["T"].type = 1
        #     print("Add", identiy_node)
        #     override_graph_def.node.extend([identiy_node])
        else:
            override_graph_def.node.extend([node])

    with TF.gfile.GFile("./pb_models/D3RQN_phase4.pb", "wb") as FileHandler:
        FileHandler.write(override_graph_def.SerializeToString())

    logger = TF.summary.FileWriter(".\\logs\\phase_4")
    logger.add_graph(override_graph_def) 

    return True


def test():
    import cv2
    from PIL import Image

    gFile = TF.gfile.FastGFile("./pb_models/D3RQN_UTL.pb", 'rb')
    graph_def = TF.GraphDef()

    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(gFile.read())
    gFile.close()

    sess_pb = TF.Session()
    sess_pb.graph.as_default()

    # Import a serialized TF `GraphDef` protocol buffer & place into the current `Graph`.
    TF.import_graph_def(graph_def)

    # Define input & output tensors
    preContexts_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/preContexts:0')
    LSTM_cell_input_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/LSTMCellZeroState/zeros:0')
    LSTM_hidden_input_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/LSTMCellZeroState/zeros_1:0')

    action_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/Qbest:0')
    LSTM_cell_output_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/IR_converter/cell_state:0')
    LSTM_hidden_output_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/IR_converter/hidden_state:0')

    test_idx = 0
    testing = True
    while testing:
        test_idx += 1
        print("\n\t\t[TEST]", test_idx)

        _ = input("Press any key to test ")

        # Get 1st new observation
        agent.be_ready()

        cell_state_pb = NP.zeros([1, agent.lstm_units])
        hidden_state_pb = NP.zeros([1, agent.lstm_units])

        current_moves = 0
        while agent.on_duty or current_moves>MAX_MOVES:
            current_moves += 1
            print("\nMOVE {} -------".format(current_moves))
            image_path = input("Insert path to image: ")

            raw_current_state = Image.open(image_path)
            current_contexts, _ = agent.extract_contexts(raw_current_state)
            action_ckpt, _, ckpt_state, hidden_state_ckpt, cell_state_ckpt = agent.query(current_contexts, return_all=True)
            # print("[CKPT]\n\t", action_ckpt, '\n\t', LSTM_state[0], '\n\t', LSTM_state[1])

            action_pb, cell_state_pb, hidden_state_pb = sess_pb.run(
                [action_tensor, LSTM_cell_output_tensor, LSTM_hidden_output_tensor],
                feed_dict={
                    preContexts_tensor: [current_contexts],
                    LSTM_cell_input_tensor: cell_state_pb, 
                    LSTM_hidden_input_tensor: hidden_state_pb, 
                }
            )
            # print("[Frozen]\n\t", action_batch_pb, '\n\t', lstm_cell_state, '\n\t', lstm_hidden_state)
            
            print("Difference b/w CKPT {} & PB {}:".format(action_ckpt, action_pb))
            print(cell_state_pb-ckpt_state.c)
            print(hidden_state_pb-ckpt_state.h)
            print(cell_state_pb-cell_state_pb)
            print(hidden_state_pb-hidden_state_ckpt)

    return True



if __name__ == "__main__":

    PROCESS = 'IR'
    agent = Agent(process=PROCESS, 
                  use_text=False, 
                  use_icon=False,
                  mode='inference')
    # phase_1()
    # phase_2()
    # phase_3()
    phase_4()
    # test()
        



