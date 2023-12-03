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
    


import sys
sys.path.insert(1, "D:\\HARI\\__RL_Dev__\\__Main_UTL__")
from yolo_v3 import YOLO

# yolo = YOLO()
# yolo.yolo_model.summary()
# print('[YOLO] Input tensors:', yolo.yolo_model.inputs)
# print('[YOLO] Output tensors:', yolo.yolo_model.outputs)

# for t in K.get_session().graph.get_operations():
#     print(t.name)

# frozen_graph = freeze_session(K.get_session(),
#                               # output_names=[out.op.name for out in yolo.yolo_model.outputs])
#                               output_names=['concat_11', 'concat_12', 'concat_13'])
# TF.train.write_graph(frozen_graph, "pb_models", "YOLO_utl.pb", as_text=False)
# TF.train.write_graph(frozen_graph, "pb_models", "YOLO_utl.pbtxt", as_text=True)


K.clear_session()
TF.reset_default_graph()

gFile = TF.gfile.FastGFile("./pb_models/YOLO_utl.pb", 'rb')
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

for t in sess.graph.get_operations():
    print(t.values())


input_tensor = sess.graph.get_tensor_by_name('import/input_1:0')
size_tensor = sess.graph.get_tensor_by_name('import/Placeholder_366:0')
bbox_tensor = sess.graph.get_tensor_by_name('import/concat_11:0')
score_tensor = sess.graph.get_tensor_by_name('import/concat_12:0')
class_tensor = sess.graph.get_tensor_by_name('import/concat_13:0')

import cv2
from PIL import Image
from yolo_utils.yolo3.utils import letterbox_image

image_path = "D:\\HARI\\__RL_Dev__\\__Main_UTL__\\Environments\\states\\full_process\\s_01.png"
image = Image.open(image_path)

boxed_image = letterbox_image(image, (416, 416))
image_data = NP.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = NP.expand_dims(image_data, 0)

out_boxes, out_scores, out_classes = sess.run(
    [bbox_tensor, score_tensor, class_tensor],
    feed_dict={
        input_tensor: image_data,
        size_tensor: (image.size[1], image.size[0]),
    }
)

print("Bboxes:", out_boxes.shape)
print("Scores:", out_scores.shape)
print("Classes:", out_classes.shape)


from post_processing import non_max_suppression

print(out_boxes.shape)

idx = non_max_suppression(out_boxes, 0.55, out_scores)
print(out_boxes.shape)
print(idx)
out_boxes, out_scores, out_classes = out_boxes[idx], out_scores[idx], out_classes[idx]
out_boxes = NP.round(out_boxes).astype(NP.int64)

print("Bboxes:", out_boxes)
print("Scores:", out_scores)
print("Classes:", out_classes)






