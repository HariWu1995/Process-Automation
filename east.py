import sys
# sys.path.append("..")

import os
import time
# from concurrent import futures
from multiprocessing import Pool, cpu_count

import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

import math
import numpy as np
from skimage import feature
from scipy import spatial

import tensorflow as tf

from east_utils import locality_aware_nms as nms_locality
from east_utils import model

from pre_processing import *
from post_processing import *

from crnn import CRNN


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)
sys.path.insert(1, dir_path)

mother_dir, _ = os.path.realpath(dir_path).rsplit("\\", 1)
sys.path.insert(1, mother_dir)


class EAST(object):

    _defaults = {
        "gpu_num": 1,
        "gpu_memory": 0.3,
        "model_path": 'east_utils\\models\\east_icdar2015_resnet_v1_50_rbox\\',
        "labels_path": 'east_utils\\config\\labels_{}.json',
        "words_path": 'east_utils\\config\\words_{}.txt',
        "phrases_path": 'east_utils\\config\\phrases_{}.txt',
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name `{}`".format(n)

    def __init__(self, **kwargs):

        print("\n\n\n[EAST] Initializing\n\n\n")

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.process = self.process.split('_')[0]

        self.labels_path = self.labels_path.format(self.process)
        self.words_path = self.words_path.format(self.process)
        self.phrases_path = self.phrases_path.format(self.process)

        # self.labels = load_dict(os.path.join(dir_path, self.labels_path))
        self.words = load_list(os.path.join(dir_path, self.words_path))
        self.phrases = load_list(os.path.join(dir_path, self.phrases_path))

        self.total_classes = self.words+self.phrases
        self.class_names = ['leftClick_'+c for c in self.total_classes]

        self.f_score, self.f_geometry = self.generate()
        # print(self.f_score)
        # print(self.f_geometry)
        self.crnn = CRNN()

        # workers = cpu_count()
        # self.executor = Pool(processes=workers)
        print("\n\n\n[EAST] Initialized successfully\n\n\n")

    def generate(self, restrict_gpu=False):
        # Tell program which GPU we use
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # Reset the graph in case we have to load a model many times
        tf.reset_default_graph()
        with tf.get_default_graph().as_default():
            # simple syntax
            self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            f_score, f_geometry = model.model(self.input_images, is_training=False)
            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            if restrict_gpu:
                config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory
            self.sess =  tf.Session(config=config)
            print("\n\n\n[EAST] Session is created\n\n\n")

            # Load model
            ckpt_state = tf.train.get_checkpoint_state(os.path.join(dir_path, self.model_path))
            model_path = os.path.join(dir_path, self.model_path, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(self.sess, model_path)

        print("\n\n\n[EAST] Model is loaded\n\n\n")

        return f_score, f_geometry

    def locate(self, image, use_focus=False, focus_zone=None, visual_check=False):

        canvas = image.copy()

        image = np.asarray(image)[:,:,:3]
        image = image[:,:,::-1]
        H, W = image.shape[:2]
        im_resized, (ratio_h, ratio_w) = resize_image(image)
        score, geometry = self.sess.run(
            [self.f_score, self.f_geometry],
            feed_dict={
                self.input_images: [im_resized]
            }
        )

        # print("Score:\n", score.shape)
        # np.savetxt("pb_models\\east_score_ckpt.txt", score.flatten(), fmt='%1.2e')
        # print("Geometry:\n", geometry.shape)
        # np.savetxt("pb_models\\east_geometry_ckpt.txt", geometry.flatten(), fmt='%1.2e')

        boxes = merge_boxes(score_map=score, geo_map=geometry)
        image = image[:,:,::-1]

        if boxes is None:
            return {}, {}
        else:
            boxes = boxes[:,:8].reshape((-1, 4, 2))
            boxes[:,:,0] /= ratio_w
            boxes[:,:,1] /= ratio_h

        # centers = {}
        # confidences = {}
        temp_labels = []
        temp_positions = []
        temp_confidences = []

        for i, box in enumerate(boxes):
            # top left, top right, bottom right, bottom left
            box = sort_poly(box.astype(np.int32))

            # get bigger RoI for better recognition
            top = max(0, box[0,1]-5)
            left = max(0, box[0,0]-5)
            right = min(W, box[2,0]+5)
            bottom = min(H, box[2,1]+5)

            if use_focus:
                if top < focus_zone[0]-3 or bottom > focus_zone[2]+3 \
                or left < focus_zone[1]-7 or right > focus_zone[3]+7:
                    continue

            mini_box = np.copy(image[top:bottom,
                                     left:right])
            mini_box = Image.fromarray(mini_box).convert('L')

            # Run CRNN
            name, score = self.crnn.eval(mini_box)

            # push into list and wait for processing later.
            temp_labels.append(name)
            temp_positions.append([left, top, right, bottom])
            temp_confidences.append(float(score))

        if visual_check:
            drawer = ImageDraw.Draw(canvas)
            for bb in temp_positions:
                drawer.rectangle(bb, outline='red')

            plt.imshow(canvas)
            plt.show()

        # Post-processing
        bboxes, confidences = postprocess_special_cases(
            temp_labels, temp_positions, temp_confidences,
            self.total_classes
        )
        return bboxes, confidences

    def close_session(self):
        self.sess.close()
