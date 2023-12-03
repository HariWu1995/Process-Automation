import os
import sys

import time

import math
import cv2
import numpy as np

from argparse import ArgumentParser

try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin


m_input_size = 416

yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

classes = 4
coords = 4
num = 3
anchors = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]

LABELS = ("leftClick_Program", "leftClick_Configure",
            "leftClick_Back", "leftClick_Run")

label_text_color = (49, 124, 139)
label_text_color = (0, 0, 255)
label_background_color = (0, 0, 255)
box_color = (255, 0, 0)
box_thickness = 2


# def build_argparser():
#     parser = ArgumentParser()
#     parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
#                                                 Sample will look for a suitable plugin for device specified (CPU by default)", default="CPU", type=str)
#     return parser


def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence

    def info(self):
        print("xmax : ",self.xmax," | ymax : ", self.ymax)
        print("xmin : ",self.xmin," | ymin : ", self.ymin)


def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval


def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

    out_blob_h = blob.shape[2]
    out_blob_w = blob.shape[3]

    side = out_blob_h
    anchor_offset = 0
    # print("side : ", side)

    if len(anchors) == 18:   ## YoloV3
        # print("Anchors YOLO")
        if side == yolo_scale_13:
            # print("13")
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            # print("26")
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            # print("52")
            anchor_offset = 2 * 0

    elif len(anchors) == 12: ## tiny-YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 3
        elif side == yolo_scale_26:
            anchor_offset = 2 * 0

    else:                    ## ???
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    side_square = side * side
    output_blob = blob.flatten()

    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if (scale < threshold):
                continue
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            # print("height, width ",height, width)
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * output_blob[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)
    return objects

# Path IR Model
model_xml = r"../Model IR/UTL_yolo.xml" #<--- MYRIAD
model_bin = r"../Model IR/UTL_yolo.bin"

model_xml = r"../Model IR/frozen_darknet_yolov3_model_UTL.xml" #<--- MYRIAD
model_bin = r"../Model IR/frozen_darknet_yolov3_model_UTL.bin"

#Load plugin
plugin = IEPlugin(device="MYRIAD")
if "CPU" in "Option":
    plugin.add_cpu_extension("lib/libcpu_extension.so")

# Read Model IR
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))

# Load model
exec_net = plugin.load(network=net)


# path_image = "n_01.png"
path_image = "../input/utl/p_12.png"
# path_image = r"C:\Users\BlackHole\OneDrive - Viralint Pte Ltd\Projects\Github\UTL on Stick\Main Source\UTL on Stick\YOLOv3\input\scs\1.png"
path_image = r"C:\Users\BlackHole\OneDrive - Viralint Pte Ltd\Projects\Github\UTL on Stick\Main Source\UTL on Stick\YOLOv3\input\utl\p_03.png"

# while cap.isOpened():
t1 = time.time()

## Uncomment only when playing video files
#cap.set(cv2.CAP_PROP_POS_FRAMES, framepos)

from PIL import Image, ImageDraw, ImageFont

image_raw = Image.open(path_image)
# image_raw = image_raw.crop((0, 0, 1600, 960))
# print(image_raw.size)


#Configure camera
camera_width = image_raw.size[0]
camera_height = image_raw.size[1]
fps = ""
framepos = 0
frame_count = 0
vidfps = 0
skip_frame = 0
elapsedTime = 0
new_w = int(camera_width * m_input_size/camera_width)
new_h = int(camera_height * m_input_size/camera_height)
print("new_h, new_w : ",new_h, new_w)


font = ImageFont.truetype(font='font\\FiraMono-Medium.otf', size=20)

new_image_size = (camera_width - (camera_width % 32), camera_height - (camera_height % 32))

size = (416, 416)
iw, ih = new_image_size
print("new_image_size ",iw, ih)
w, h = size
scale = min(w/iw, h/ih)
nw = int(iw*scale)
nh = int(ih*scale)

image = image_raw.resize((nw,nh), Image.BICUBIC)
boxed_image = Image.new('RGB', size, (128,128,128))
boxed_image.paste(image, ((w-nw)//2, (h-nh)//2))

image_data = np.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = image_data.transpose((2,0,1))  # Change data layout from HWC to CHW
prepimg = np.expand_dims(image_data, 0)  # Add batch dimension.
print(type(prepimg))
print(prepimg.shape)

# resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
# canvas = np.full((m_input_size, m_input_size, 3), 128)
# canvas[(m_input_size-new_h)//2:(m_input_size-new_h)//2 + new_h,(m_input_size-new_w)//2:(m_input_size-new_w)//2 + new_w,  :] = resized_image
# prepimg = canvas
# prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
# prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
#                                             # N: number of images in the batch
#                                             # H: height of the image
#                                             # W: width of the image
#                                             # C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)
outputs = exec_net.infer(inputs={input_blob: prepimg})

print(len(outputs))
# print(outputs)
# print(outputs['detector/yolo-v3/Conv_14/BiasAdd/YoloRegion'].shape)
# print(outputs['detector/yolo-v3/Conv_22/BiasAdd/YoloRegion'].shape)
# print(outputs['detector/yolo-v3/Conv_6/BiasAdd/YoloRegion'].shape)

objects = []

for output in outputs.values():
    ParseYOLOV3Output(output, new_h, new_w, camera_height, camera_width, 0.7, objects)

# print(objects)

# # Filtering overlapping boxes
objlen = len(objects)
print(objlen)
for i in range(objlen):
    if (objects[i].confidence == 0.0):
        continue
    for j in range(i + 1, objlen):
        if (IntersectionOverUnion(objects[i], objects[j]) >= 0.2):
            objects[j].confidence = 0


print("number box : ", objlen)
print("number objects : ", len(objects))

# Drawing boxes
for obj in objects:
    if obj.confidence < 0.8 :
        continue
    # print("Confidence : ",obj.confidence)

    label = obj.class_id
    confidence = obj.confidence
    if confidence > 0.2:
        label_text = LABELS[label] + "\n (" + "{:.1f}".format(confidence * 100) + "%)"
        # print(label_text)
        # obj.info()
        thickness = (image_raw.size[0] + image_raw.size[1])//600
        # print("thickness", thickness)
        draw = ImageDraw.Draw(image_raw)
        label_size = draw.textsize(label_text)

        # top, left, bottom, right = box
        top = obj.ymin
        left = obj.xmin
        bottom = obj.ymax
        right = obj.xmax
        # print("top, left, bottom, right : ", top, left, bottom, right)


        if top - label_size[1] >= 0:
            # text_origin = np.array([left, top - label_size[1]])
            text_origin = np.array([left - 120, top - 20])
        else:
            text_origin = np.array([left - 25, top + 2])

        # My kingdom for a good redistributable image drawing library.
        # for i in range(thickness):
        #     draw.rectangle([left + i, top + i, right - i, bottom - i], outline=box_color)
        draw.rectangle(((left, top), (right, bottom)), outline=box_color, width=box_thickness)
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)])
        draw.text(text_origin, label_text, fill=label_text_color, font=font)

        # cv2.rectangle(image, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), box_color, box_thickness)
        # cv2.putText(image, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)

image_raw.show()

# imcv = np.asarray(image_raw)
# opencvImage = cv2.cvtColor(imcv, cv2.COLOR_RGB2BGR)
# cv2.imshow("Image", opencvImage)
# cv2.waitKey(1)

elapsedTime = time.time() - t1
fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)
print("Fpt : ", fps)

## frame skip, video file only
#skip_frame = int((vidfps - int(1/elapsedTime)) / int(1/elapsedTime))
#framepos += skip_frame
import time
time.sleep(5)
# cv2.destroyAllWindows()
