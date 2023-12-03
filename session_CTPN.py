#!/usr/bin/env python
# coding: utf-8


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
        freeze_var_names = list(set(v.op.name for v in TF.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in TF.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = vars2consts(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
    

from ctpn import *

# ctpn = CTPN()

# for t in ctpn.sess.graph.get_operations(): 
#     print(t.name)

# frozen_graph = freeze_session(ctpn.sess,
#                               output_names=['bbox_pred/Reshape_1', 'cls_pred/Reshape_1', 'cls_prob'])
# TF.train.write_graph(frozen_graph, "pb_models", "CTPN.pb", as_text=False)
# TF.train.write_graph(frozen_graph, "pb_models", "CTPN.pbtxt", as_text=True)


K.clear_session()
TF.reset_default_graph()

gFile = TF.gfile.FastGFile("./pb_models/CTPN.pb", 'rb')
graph_def = TF.GraphDef()

# Parses a serialized binary message into the current message.
graph_def.ParseFromString(gFile.read())
gFile.close()

sess = TF.Session()
sess.graph.as_default()

# Import a serialized TensorFlow `GraphDef` protocol buffer
# and place into the current default `Graph`.
TF.import_graph_def(graph_def)


from IPython.display import clear_output, Image, display, HTML


def strip_consts(graph_def, max_const_size=32):

    """Strip large constant values from graph_def."""
    strip_def = TF.GraphDef()

    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)

            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>"%size, encoding='utf-8')
    return strip_def


def show_graph(graph_def, max_const_size=32):

    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(NP.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))

    display(HTML(iframe))
    


# show_graph(TF.get_default_graph().as_graph_def())

# for t in sess.graph.get_operations():
#     print(t.values())


input_tensor = sess.graph.get_tensor_by_name('import/input_image:0')
# size_tensor = sess.graph.get_tensor_by_name('import/input_im_info:0')
bbox_tensor = sess.graph.get_tensor_by_name('import/bbox_pred/Reshape_1:0')
class_tensor = sess.graph.get_tensor_by_name('import/cls_pred/Reshape_1:0')
score_tensor = sess.graph.get_tensor_by_name('import/cls_prob:0')

import cv2
from PIL import Image

image_path = "D:\\HARI\\__RL_Dev__\\__Main_UTL__\\Environments\\states\\full_process\\s_02.png"
image_original = Image.open(image_path)
image = NP.asarray(image_original)
H, W = image.shape[:2]
im = image[:,:,0:3]

img = resize_image(im)
nH, nW, c = img.shape[:3]
im_info = NP.array([nH, nW, c]).reshape([1, 3])

bbox_pred_val, cls_prob_val = sess.run(
    [bbox_tensor, score_tensor],
    feed_dict={
        input_tensor: [img],
        # input_im_info: im_info
    }
)

# print("Bbox:\n", bbox_pred_val.shape)
# np.savetxt("pb_models\\ctpn_bbox_pb.txt", bbox_pred_val.flatten(), fmt='%1.2e')
# print("Prob:\n", cls_prob_val.shape)
# np.savetxt("pb_models\\ctpn_prob_pb.txt", cls_prob_val.flatten(), fmt='%1.2e')


# ctpn.locate(image_original)




