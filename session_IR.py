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
    

def phase_1():

    K.clear_session()
    TF.reset_default_graph()
    
    for t in sess.graph.get_operations():
        if t.name.startswith('primaryQN'):
            print(t.values())
            # print(t.name)

    frozen_graph = freeze_session(sess,
                                  output_names=['primaryQN/Qbest', 
                                                'primaryQN/rnn_transfer/while/Exit_3',
                                                'primaryQN/rnn_transfer/while/Exit_4'])
    TF.train.write_graph(frozen_graph, "test_IR", "D3RQN_phase1.pb", as_text=False)
    TF.train.write_graph(frozen_graph, "test_IR", "D3RQN_phase1.pbtxt", as_text=True)

    logger = TF.summary.FileWriter(".\\test_IR\\phase_1")
    logger.add_graph(frozen_graph) 

    return True


def phase_2():

    K.clear_session()
    TF.reset_default_graph()

    gFile = TF.gfile.FastGFile("./test_IR/D3RQN_phase1.pb", 'rb')
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
            override_node = TF.placeholder(dtype=TF.float32, shape=[1, LSTM_UNITS], name=node_name)
            override_node_def = override_node.op.node_def
            print("Convert \n\t{} to \n\t{}".format(node, override_node_def))
            override_graph_def.node.extend([override_node_def])
        else:
            override_graph_def.node.extend([deepcopy(node)])

    with TF.gfile.GFile("./test_IR/D3RQN_phase2.pb", "wb") as FileHandler:
        FileHandler.write(override_graph_def.SerializeToString())

    logger = TF.summary.FileWriter(".\\test_IR\\phase_2")
    logger.add_graph(override_graph_def)

    return True


def phase_3():
    
    K.clear_session()
    TF.reset_default_graph()

    gFile = TF.gfile.FastGFile("./test_IR/D3RQN_phase2.pb", 'rb')
    graph_def = TF.GraphDef()

    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(gFile.read())
    gFile.close()

    sess_pb = TF.Session()
    sess_pb.graph.as_default()

    # Import a serialized TF `GraphDef` protocol buffer & place into the current `Graph`.
    TF.import_graph_def(graph_def)

    # for t in sess_pb.graph.get_operations():
    #     # print(t.values())
    #     print(t.name)

    # Define input & output tensors
    # preContexts_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/preContextsFlattened:0')
    # LSTM_cell_input_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/LSTMCellZeroState/zeros:0')
    # LSTM_hidden_input_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/LSTMCellZeroState/zeros_1:0')

    # action_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/Qbest:0')
    LSTM_cell_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/rnn_transfer/while/Exit_3:0')
    LSTM_hidden_tensor = sess_pb.graph.get_tensor_by_name('import/primaryQN/rnn_transfer/while/Exit_4:0')

    mask = TF.zeros(shape=[1, LSTM_UNITS], dtype=tf.float32, name="import/primaryQN/mask")
    LSTM_cell_output_tensor = TF.add(LSTM_cell_tensor, mask, name='import/primaryQN/LSTM_hidden_cell_output')
    LSTM_hidden_output_tensor = TF.add(LSTM_hidden_tensor, mask, name='import/primaryQN/LSTM_hidden_state_output')

    frozen_graph = freeze_session(sess_pb,
                                  output_names=['import/primaryQN/Qbest', 
                                                'import/primaryQN/LSTM_hidden_cell_output',
                                                'import/primaryQN/LSTM_hidden_state_output'])
    TF.train.write_graph(frozen_graph, "test_IR", "D3RQN_phase3.pb", as_text=False)
    TF.train.write_graph(frozen_graph, "test_IR", "D3RQN_phase3.pbtxt", as_text=True)

    logger = TF.summary.FileWriter(".\\test_IR\\phase_3")
    logger.add_graph(frozen_graph) 

    return True


if __name__ == "__main__":
    
    from Agents.d3rqn_ir import *

    TF.reset_default_graph()
    sess = TF.Session()

    N_ACTIONS = 15
    LSTM_UNITS = 248

    Model = D3RQN(n_actions=N_ACTIONS, lstm_units=LSTM_UNITS)
    vars_list = {
        v.name[:-2]: v for v in TF.global_variables() 
    }
    saver = TF.train.Saver(var_list=vars_list)
    checkpoint_path = "D:/HARI/__RL_Dev__/__Backend_RL__/Agents/__IR__/d3rqn.ckpt"
    saver.restore(sess, checkpoint_path)

    phase_1()
    phase_2()
    phase_3()
    # phase_4()

    _ = input("Press any key to continue ")
    # test()
        

"""
python "C:/Program Files (x86)/IntelSWTools/openvino_2020.2.117/deployment_tools/model_optimizer/mo_tf.py"
-m "D:/HARI/__RL_Dev__/__Backend_RL__/test_IR/D3RQN_UTL.pb"
-o "D:/HARI/__RL_Dev__/__Backend_RL__/test_IR"
--data_type FP32
--input "import/primaryQN/preContextsFlattened[1 1 248],import/primaryQN/LSTMCellZeroState/zeros[1 248],import/primaryQN/LSTMCellZeroState/zeros_1[1 248]"
--output import/primaryQN/Contexts,import/primaryQN/LSTM_hidden_cell_output,import/primaryQN/LSTM_hidden_state_output
"""


