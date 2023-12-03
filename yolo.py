import sys
# sys.path.append("..")


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import load_model, Model
from keras.layers import Input

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from tensorflow.compat.v1.keras.backend import set_session

import tensorflow as tf
from keras import backend as K

from PIL import Image

from yolo_utils.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo_utils.yolo3.utils import letterbox_image

from pre_processing import *
from post_processing import *


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)
sys.path.insert(1, dir_path)

mother_dir, _ = os.path.realpath(dir_path).rsplit("\\", 1)
sys.path.insert(1, mother_dir)


class YOLO(object):

    _defaults = {
        "iou": 0.43,
        "score": 0.2,
        "gpu_num": 1,
        "gpu_memory": 0.4,
        "model_path": os.path.join(dir_path, "yolo_utils\\models\\yolo_{}.h5"),
        "classes_path": os.path.join(dir_path, 'yolo_utils\\config\\classes_{}.txt'),
        "anchors_path": os.path.join(dir_path, 'yolo_utils\\config\\yolo_anchors.txt'),
        "model_image_size": (416, 416),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name `{}`".format(n)

    def __init__(self, **kwargs):

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("\n\n\n[YOLO_v3] Initializing\n\n\n")

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.process = self.process.split('_')[0]

        self.model_path = self.model_path.format(self.process)
        self.classes_path = self.classes_path.format(self.process)

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        # Reset the graph in case we have to load a model many times
        K.clear_session()
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory
        self.sess = tf.Session(config=config, graph=tf.get_default_graph())
        K.set_session(self.sess)
        # tf.keras.backend.set_session(self.sess)
        print("\n\n\n[YOLO_v3] Session is created\n\n\n")

        self.boxes, self.scores, self.classes = self.generate()

        print("\n\n\n[YOLO_v3] Initialized successfully\n\n\n")

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):

        # check if model is h5 file or not 
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
            print("\n\n\n[YOLO_v3] Loaded model\n\n\n")
        except Exception as e:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            print("\n\n\n[YOLO_v3] Loading weights ...\n\n\n")
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes+5), \
                'Mismatch between model and given anchor and class sizes'

        # temp model to extract features
        # layer_name = 'leaky_re_lu_65'
        # self.feature_maps = self.yolo_model.get_layer(layer_name).output
        # self.feature = K.function([self.yolo_model.input], [self.feature_maps])

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num > 1:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output, 
            self.anchors,
            len(self.class_names), 
            self.input_image_shape,
            score_threshold=self.score, 
            iou_threshold=self.iou
        )
        return boxes, scores, classes
        
    def locate(self, image):
        """
        Input:
            + image: image need detecting
        Output:
            + centers: a dictionary with 
                            keys: buttons detected
                            values: centers of all buttons
            + confidences: a dictionary with 
                            keys: buttons detected
                            values: confidence of all buttons
        """
        # image = Image.open(image_path)

        # resize image to 416,416 to make input for Yolov3-416
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width%32),
                              image.height - (image.height%32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # take output from model
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: (image.size[1], image.size[0]),
                # K.learning_phase(): 0
            }
        )

        # apply nms into result, to this step, it take 0.7 seconds
        idx = non_max_suppression(out_boxes, 0.55, out_scores)
        out_boxes, out_scores, out_classes = out_boxes[idx], out_scores[idx], out_classes[idx]
        out_boxes = np.round(out_boxes).astype(np.int64)

        # take position of each label
        bboxes = {}
        confidences = {}
        for box, score, cl in zip(out_boxes, out_scores, out_classes):
            predicted_class = self.class_names[cl]
            if predicted_class in bboxes.keys():
                if score > confidences[predicted_class]:
                    bboxes[predicted_class] = box[:4]
                    confidences[predicted_class] = score
            else:
                bboxes[predicted_class] = box[:4] # top, left, bottom, right
                confidences[predicted_class] = score
        return bboxes, confidences
        
    def close_session(self):
        self.sess.close()

