import sys
sys.path.append('..')

import cv2
import numpy as np

import lanms_utils
from east_utils.icdar import restore_rectangle
from east_utils import locality_aware_nms as nms_locality

import pytesseract 
from pytesseract.pytesseract import tesseract_cmd

from difflib import SequenceMatcher


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def merge_boxes(score_map, geo_map, 
                score_map_thresh=0.8, 
                box_thresh=0.01, 
                nms_thresh=0.2):
    '''
    :function
        restore text boxes from score-map and geo-map
    :param 
        score_map:
        geo_map:
        score_map_thresh: threshold for score map
        box_thresh: threshold for boxes
        nms_thresh: threshold for non-maximum suppression
    :return:
    '''
 
    if len(score_map.shape) == 4:
        score_map = score_map[0,:,:,0]
        geo_map = geo_map[0,:,:,]

    # filter the score map
    xy_text = np.argwhere(score_map>score_map_thresh)
 
    # sort the text boxes via the y-axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    
    # restore
    text_box_restored = restore_rectangle(xy_text[:,::-1]*4, 
                                          geo_map[xy_text[:, 0], 
                                                  xy_text[:, 1], 
                                                  :]) # N*4*2
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:,:8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:,0], xy_text[:,1]]
    
    # nms part
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thresh)
    boxes = lanms_utils.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)

    if boxes.shape[0] == 0:
        return None

    # filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32)//4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    
    boxes = boxes[boxes[:, 8]>box_thresh]
    return boxes


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0]-p[1, 0]) > abs(p[0, 1]-p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def resize_image(im, max_side_len=2400):
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len)/resize_h if resize_h>resize_w else float(max_side_len)/resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h*ratio)
    resize_w = int(resize_w*ratio)

    resize_h = resize_h if resize_h%32==0 else (resize_h//32-1)*32
    resize_w = resize_w if resize_w%32==0 else (resize_w//32-1)*32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
   
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    chosen_ones = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = (x2-x1+1) * (y2-y1+1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

        
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        chosen_ones.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w*h) / area[idxs[:last]]
        
        iou = w*h
        out = np.where(iou/area[i]>0.8)
        
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap>max_bbox_overlap)[0], out[0])
            )
        )
    return chosen_ones


def compare_strings(a, b):
    
    import re
    from difflib import SequenceMatcher

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
    a = re.sub(r"[,.:;@#?!&$]+\/| *", '', a)
    b = re.sub(r"[,.:;@#?!&$]+\/| *", '', b)

    sim = SequenceMatcher(None, a, b).ratio()
    return sim


def read_word(image):
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
    return text


def two_becomes_one(box1, box2):
    return [min(box1[0], box2[0]), 
            min(box1[1], box2[1]), 
            max(box1[2], box2[2]),
            max(box1[3], box2[3])]


def postprocess_special_cases(temp_labels, temp_positions, temp_confidences, labels):

    # print("\n\n\nPost-Processing ...\n")
    bboxes = dict()
    confidences = dict()

    # work on array as a pro-gramer
    temp_positions = np.array(temp_positions)
    temp_confidences = np.array(temp_confidences)

    # Split positions into useful data
    lefts, tops, rights, bottoms = temp_positions.T

    # Assign centers
    centers_h = np.array([(tops+bottoms)//2])
    centers_w = np.array([(lefts+rights)//2])
    centers_point = np.concatenate((centers_h, centers_w), axis=0).T

    # fake index for this part only 
    indices = np.arange(0, len(temp_confidences), 1)
    
    processed_list = []
    for n, (l,t,r,b) in enumerate(zip(lefts, tops, rights, bottoms)):

        # Ignore who are processed
        if n in processed_list:
            continue
    
        # choose words are in same line
        rows = np.where(np.abs(t-tops)<13) and \
               np.where(np.abs(b-bottoms)<13)
        boxes_in_line = temp_positions[rows]
        indices_left = indices[rows]
        # print("Same Line\n", boxes_in_line)

        # sort parallel with index
        _, order = zip(*sorted(zip(boxes_in_line[:,0], indices_left)))
        order = list(order)

        # choose words are close to each other
        temp_rights = rights[order]
        temp_lefts = lefts[order]
        groups = [[order[0]]]
        groups_idx = 0
        for idx in range(len(order)-1):
            if np.abs(lefts[order[idx+1]]-rights[order[idx]]) < 31:
                groups[groups_idx] += [order[idx+1]]
            # elif order[min_distance_idx] not in columns:
            else:
                groups_idx += 1
                groups += [[order[idx+1]]]

        for group in groups:

            # Decode position of merged boxes
            boxes_in_group = temp_positions[rows and group]
            groupLeft = np.min(boxes_in_group[:,0])
            groupTop = np.min(boxes_in_group[:,1])
            groupRight = np.max(boxes_in_group[:,2])
            groupBottom = np.max(boxes_in_group[:,3])
            # print("Group\n", boxes_in_group)

            # add indices into processed-list
            indices_left = indices[group]
            processed_list += indices_left.tolist()

            # merge label by index
            this_label = ' '.join([temp_labels[j] for j in indices_left])
            
            # check result
            max_sim = -1
            for label in labels:
                sim = compare_strings(this_label, label)
                if sim > max_sim:
                    max_sim = sim
                    best_label = 'leftClick_'+label

            if max_sim < 0.997**(len(best_label)-9)-0.13:
                # print('\tNon-Existent Label:', best_label, max_sim)
                continue
            # print('\tLabel:', best_label)

            # score is the mean value of all points' scores, 
            # then is multiplied by similarity of text-recognition
            this_score = np.mean(temp_confidences[rows and group]) * max_sim

            # update dictionaries
            bboxes[best_label] = [groupTop, groupLeft, groupBottom, groupRight]
            confidences[best_label] = this_score

    return bboxes, confidences


def get_extreme_location(all_positons, extreme_type):
    all_positons = np.array(all_positons)
    if extreme_type.lower() == "min":
        extreme_idx = np.argmin(all_positons, axis=0)
    elif extreme_type.upper() == "MAX":
        extreme_idx = np.argmax(all_positons, axis=0)
    else:
        print("[ExtremeTypeError] Extreme can NOT be", extreme_type)
        return None

    return list(all_positons[extreme_idx[0]])

