import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)
# os.environ['SDL_VIDEO_CENTERED'] = '0'

# windows_offset = 0

import sys
import copy
import shutil

import ctypes

import time
import datetime

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
import math
import random

import itertools
from collections import defaultdict, deque

import pygame
from pygame.locals import *


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)


WIDTH, HEIGHT = 1920, 1080

""" Create a Scenario to use for all functions """
screen_resolution = (WIDTH, HEIGHT)
scenario = pygame.display.set_mode(screen_resolution)
scenario.fill([255, 255, 255])  # clear the screen


key_mapping = {
    "f1": pygame.K_F1,
    "f2": pygame.K_F2,
    "f3": pygame.K_F3,
    "f4": pygame.K_F4,
    "f5": pygame.K_F5,
    "f6": pygame.K_F6,
    "f7": pygame.K_F7,
    "space": pygame.K_SPACE,
}

metakey_mapping = {
    "control": [pygame.K_RCTRL, pygame.K_LCTRL, ],
    "shift": [pygame.K_RSHIFT, pygame.K_LSHIFT, ],
    "alt": [pygame.K_RALT, pygame.K_LALT, ]
}
                             
                                                      
class Button(pygame.sprite.Sprite):

    def __init__(self, image, position, activate=True):
        super().__init__()

        self.image = image
        self.position = position
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = self.position
        self.activate = activate

        scenario.blit(self.image, self.position)


def read_key_stroken():
    key_pressed = False
    while not key_pressed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                print('\tKey pressed:', event.key)
                key_pressed = True
                key_striked = event.key #pygame.key.get_pressed() 
    return key_striked


def main_app():

    # Get and display the scene
    scene_path = os.path.join(dir_path, "app_resources/scenes/WaferReview.png")
    scenario.blit(pygame.image.load(scene_path), (0, 0))    

    pygame.mouse.set_visible(True)
    pygame.display.update()

    # Load list of defects
    defect_zone = [40, 170, 855, 855]
    defect_list = glob(
        os.path.join(dir_path, "app_resources/button_WaferReview/defects/*.jpg")
    )
    random.shuffle(defect_list)
    print("Number of defects:", len(defect_list))

    # Load defect till the end
    load_new_defect = True
    clicks_sequence = deque(maxlen=19)
    running = True

    while running and len(defect_list)>0:
        if load_new_defect:
            load_new_defect = False
            current_defect = defect_list.pop()
            object_defect = Button(
                image=pygame.transform.scale(
                    pygame.image.load(current_defect), defect_zone[-2:]
                ),
                position=defect_zone[:2]
            )
            with open(current_defect.replace('jpg', 'txt'), 'r') as file_handler:
                current_label = file_handler.readline().lower()
            print("\n\n\n", current_defect, "-->", current_label)

        # Apply changes
        pygame.display.update()

        # Read which key was stroken
        key_striked = read_key_stroken()
        clicks_sequence.append(key_striked)

        # Check the right key was stroken
        if current_label == "space":
            if clicks_sequence[-1] != key_mapping["space"]:
                continue
        elif len(clicks_sequence) >= 2:
            meta_key, label_key = current_label.split('+')
            if clicks_sequence[-2] not in metakey_mapping[meta_key]:
                continue
            if clicks_sequence[-1] != key_mapping[label_key]:
                continue
        load_new_defect = True

    _ = input("Press ANY key to quit ")

            
if __name__ == "__main__":

    # Initialize the app
    pygame.init()
    # pygame.mixer.init()
    # print(pygame.display.get_window_size())

    main_app()
    









