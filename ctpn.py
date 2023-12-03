import os
import sys
import time
import shutil

import json
from copy import deepcopy

import cv2
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image, ImageFilter, ImageDraw

from ctpn_utils.nets import model_train as model
from ctpn_utils.rpn_msr.proposal_layer import proposal_layer
from ctpn_utils.text_connector.detectors import TextDetector


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)
sys.path.insert(1, dir_path)

mother_dir, _ = os.path.realpath(dir_path).rsplit("\\", 1)
sys.path.insert(1, mother_dir)


class CTPN(object):

    _defaults = {
        "gpu_name": '0',
        "gpu_memory": 0.3,
        "model_path": os.path.join(dir_path, "ctpn_utils\\model\\"),
        "labels_path": os.path.join(dir_path, "ctpn_utils\\config\\contexts_{}.json"),
        "classes_path": os.path.join(dir_path, "ctpn_utils\\config\\classes_{}.txt"),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.process = self.process.split('_')[0]

        self.labels_path = self.labels_path.format(self.process)
        self.classes_path = self.classes_path.format(self.process)

        self.labels = loadJson(self.labels_path)
        self.class_names = ['leftClick_'+v for k,v in self.labels.items()]
        self.class_names = list(set(self.class_names))
        writer = open(self.classes_path, 'w')
        writer.write("\n".join(self.class_names))
        writer.close()

        self.bbox_pred, self.cls_pred, self.cls_prob = self.generate()
        print(self.bbox_pred)
        print(self.cls_pred)
        print(self.cls_prob)

        print("\n\n\n[CTPN] Sucessfully initialized!\n\n\n")

    def generate(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_name
        with tf.get_default_graph().as_default():
            # generate model
            self.input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            self.input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = model.model(self.input_image)
            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            # divide GPU's RAM
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory
            self.sess =  tf.Session(config=config)

            # load checkpoint
            ckpt_state = tf.train.get_checkpoint_state(self.model_path)
            model_path = os.path.join(
                self.model_path,
                os.path.basename(ckpt_state.model_checkpoint_path)
            )
            saver.restore(self.sess, model_path)

        return bbox_pred, cls_pred, cls_prob

    def locate(self, image, keep_labels_only=True, use_focus=False, focus_zone=None, visual_check=False):

        canvas = image.copy()

        # required image from opencv
        image = np.asarray(image)
        H, W = image.shape[:2]
        im = image[:,:,0:3]

        # resize image to get 1 shape is 750
        # img = cv2.resize(im, (750, 750), interpolation=cv2.INTER_LINEAR)
        img = resize_image(im) # (1280, 768) --> (1008, 608)
        nH, nW, c = img.shape[:3]
        im_info = np.array([nH, nW, c]).reshape([1, 3])

        bbox_pred_val, cls_prob_val = self.sess.run(
            [self.bbox_pred, self.cls_prob],
            feed_dict={
                self.input_image: [img],
                # self.input_im_info: im_info
            }
        )
        # print("Bbox:\n", bbox_pred_val.shape)
        # np.savetxt("pb_models\\ctpn_bbox_ckpt.txt", bbox_pred_val.flatten(), fmt='%1.2e')
        # print("Prob:\n", cls_prob_val.shape)
        # np.savetxt("pb_models\\ctpn_prob_ckpt.txt", cls_prob_val.flatten(), fmt='%1.2e')

        textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
        scores = textsegs[:, 0]
        textsegs = textsegs[:, 1:5]

        textdetector = TextDetector(DETECT_MODE='H')
        boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
        boxes = np.array(boxes, dtype=np.int)
        # example of 1 row in boxes: 176,56,400,56,400,69,176,69,0.99732864

        # convert position from resized_image to real image
        boxes[:,0] = boxes[:,0] * W/nW
        boxes[:,2] = boxes[:,2] * W/nW
        boxes[:,1] = boxes[:,1] * H/nH
        boxes[:,5] = boxes[:,5] * H/nH

        if visual_check:
            drawer = ImageDraw.Draw(canvas)
            for bb in boxes:
                left, top, right, _, _, bottom, _, _, _ = bb
                drawer.rectangle([left, top, right, bottom], outline='red')

            plt.imshow(canvas)
            plt.show()

        centers, confidences = read_image(im, boxes, keep_labels_only, self.labels, use_focus, focus_zone)
        return centers, confidences

    def close_session(self):
        self.sess.close()


def resize_image(img):
    # resize image to get the suitable input for ctpn model
    img_size = img.shape
    img_size_min = np.min(img_size[0:2])
    img_size_max = np.max(img_size[0:2])

    img_scale = float(600) / float(img_size_min)
    if np.round(img_scale*img_size_max) > 1200:
        img_scale = float(1200) / float(img_size_max)
    new_h = int(img_size[0]*img_scale)
    new_w = int(img_size[1]*img_scale)

    new_h = new_h if new_h//16==0 else (new_h//16+1)*16
    new_w = new_w if new_w//16==0 else (new_w//16+1)*16

    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    print("[CTPN_resize] {}x{} --> {}x{}".format(img_size[0], img_size[1], new_h, new_w))
    return new_img


import pytesseract as pt
pt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def read_image(image, boxes, keep_labels_only=False, labels=None, use_focus=False, focus_zone=None):

    H, W = image.shape[:2]

    bboxes = {}
    confidences = {}

    for idx, box in enumerate(boxes):
        # each row get 9 variable - 4 box points and the confidence of that box
        left, top, right, _, _, bottom, _, _, _ = box

        if use_focus:
            if top < focus_zone[0]-3 or bottom > focus_zone[2]+3 \
            or left < focus_zone[1]-7 or right > focus_zone[3]+7:
                continue

        # expand image and binarize the input of tesseract
        roi = image[
            max(top-11, 3):min(bottom+11, H-3),
            max(left-13, 7):min(right+13, W-7)
        ]
        # border_size = 403
        # if roi.shape[0] < border_size:
        #     roi = cv2.copyMakeBorder(roi,
        #                              (border_size-roi.shape[0])//2, (border_size-roi.shape[0])//2,
        #                              0, 0,
        #                              cv2.BORDER_CONSTANT,#BORDER_REPLICATE)
        #                              value=[255,255,255])
        # if roi.shape[1] < border_size:
        #     roi = cv2.copyMakeBorder(roi,
        #                              0, 0,
        #                              (border_size-roi.shape[1])//2, (border_size-roi.shape[1])//2,
        #                              cv2.BORDER_CONSTANT,#BORDER_REPLICATE)
        #                              value=[255,255,255])
        roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 11, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        roi = cv2.resize(roi, None, fx=1.13, fy=1.19, interpolation=cv2.INTER_LINEAR)
        # plt.imshow(roi)
        # plt.show()

        result = pt.image_to_data(roi, output_type=pt.Output.DICT)
        words = []
        confidence = 0
        n_words = 0
        for c, t in zip(result['conf'], result['text']):
            if c != '-1':
                confidence += int(c)
                n_words += 1
                words += [t]
        if n_words == 0:
            continue

        text = " ".join(w for w in words if w!=' ')
        text = re.sub(r"[,.;@#?!&$]+\| *", " ", text)
        text = text.replace('  ', ' ')

        score = float(confidence)/n_words/100
        # print(text, score)

        if not keep_labels_only:
            label = text
            max_sim = 1.0
        else:
            max_sim = -1
            for k, v in labels.items():
                sim = compare_strings(text, v)
                if sim > max_sim:
                    max_sim = sim
                    label_id = k

            if max_sim < 0.61:
                continue

            label = "leftClick_" + labels[label_id]

        # Suppose that FOCUS ZONE has eliminated all buttons share the same name, except the last one
        confidences[label] = [max_sim*score]
        bboxes[label] = [top, left, bottom, right]

    return bboxes, confidences


def loadJson(json_path):
    with open(json_path, 'r') as l:
        labels = json.loads(l.read())
    return labels


import re
from difflib import SequenceMatcher

def compare_strings(a, b):

    # Set lower-case and Stick every word into one
    a = a.lower().replace(' ', '')
    b = b.lower().replace(' ', '')

    # Remove all number
    a = re.sub('[0-9]', '', a)
    b = re.sub('[0-9]', '', b)

    # Remove all characters inside brackets
    a = re.sub(r'\([^)]*\)', '', a)
    b = re.sub(r'\([^)]*\)', '', b)

    # Remove special characters
    a = re.sub(r"[,.:;@#?!&$]+\| *", '', a)
    b = re.sub(r"[,.:;@#?!&$]+\| *", '', b)

    sim = SequenceMatcher(None, a, b).ratio()
    return sim
