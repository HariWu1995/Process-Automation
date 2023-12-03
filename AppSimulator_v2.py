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
from collections import defaultdict

import pygame
from pygame.locals import *

from Environments.environment import *


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)


# Read the configuration file 
with open('RL_config_thorough.json', 'r') as reader:
    Config = json.load(reader)

# Assign training hyper-parameters with configuration
PROCESS = Config['Process']
useText = Config['UseText']
useIcon = Config['UseIcon']
arrivalNode = Config['ArrivalNode']
destinationNode = Config['DestinationNode']
WIDTH, HEIGHT = Config['SCREEN_WIDTH'], Config['SCREEN_HEIGHT']

""" Create a Scenario to use for all functions """
screen_resolution = (WIDTH, HEIGHT)
scenario = pygame.display.set_mode(screen_resolution)
scenario.fill([255, 255, 255])  # clear the screen


def print_windows_info():
    from ctypes import POINTER, WINFUNCTYPE, windll
    from ctypes.wintypes import BOOL, HWND, RECT
    try:
        windows_info = pygame.display.get_wm_info()["window"]
        print("[window]", windows_info)

        prototype = WINFUNCTYPE(BOOL, HWND, POINTER(RECT))
        paramflags = (1, "hwnd"), (2, "lprect")

        GetWindowRect = prototype(("GetWindowRect", windll.user32), paramflags)
        rect = GetWindowRect(windows_info)
        print("Window Rect:", rect.top, rect.left, rect.bottom, rect.right)
        return True
    except Exception:
        print("[WindowsInfo] Error")
        return False


def MessageBox(title, text, style=3):
    """  
    Styles:
        0 : OK
        1 : OK     | Cancel
        2 : Abort  | Retry     | Ignore
        3 : Yes    | No        | Cancel
        4 : Yes    | No
        5 : Retry  | No 
        6 : Cancel | Try Again | Continue
        
    To change icon, add value to above number
        16: Stop-sign icon
        32: Question-mark icon
        48: Exclamation-point icon
        64: Information-sign icon
    """
    return ctypes.windll.user32.MessageBoxW(3, text, title, style, 32)


def convert_pil_image(pil_image):
    image_mode = pil_image.mode
    image_size = pil_image.size
    image_data = pil_image.tobytes()

    image = pygame.image.fromstring(image_data, image_size, image_mode)
    return image


def convert_np_image(np_image, color_channels=3):
    np_image = np_image[:,:,:color_channels]

    # resize
    image = cv2.resize(np_image, dsize=(0,0),
                       fx=WIDTH/1920, 
                       fy=HEIGHT/1080, 
                       interpolation=cv2.INTER_CUBIC)
    # rotate clock-wise
    image_90 = np.rot90(image, k=1, axes=(1,0)) # for counter-clockwise: axes=(0,1)
    
    # flip horizontally (left/right direction)
    image_array = np.fliplr(image_90) 

    # flip vertically (up/down direction)
    # image_array = np.flipud(image_90) 
    return pygame.surfarray.make_surface(image_array)


def transform_coordinate(position, old_W=1920, old_H=1080,
                                  new_W=WIDTH, new_H=HEIGHT):
    """
    Corresponding to step-by-step of function `convert_np_image`
    """
    position_w, position_h = position
    # print(position_w, position_h)

    # Apply to rotate clockwise
    M = cv2.getRotationMatrix2D(center=(old_W//2, old_H//2), 
                                angle=-90, scale=1.0)
    cosine = np.abs(M[0, 0])
    sine = np.abs(M[0, 1])
    position_w = int(old_H*sine + old_W*cosine)
    position_h = int(old_H*cosine + old_W*sine)
    # print(position_w, position_h)

    # Apply to resize
    position_w *= (new_W//old_W)
    position_h *= (new_H//old_H)
    # print(position_w, position_h)

    # Apply to flip horizontally
    position_w = new_W - position_w
    # print(position_w, position_h)
    return position_w, position_h
                             
                                                      
class Button(pygame.sprite.Sprite):
    
    def __init__(self, image_array, position, activate=True):
        
        super().__init__()

        self.image = convert_np_image(image_array)
        self.activate = activate
        position_w, position_h = position
        # print(position_w, position_h)
        position_w = int(position_w*WIDTH/1920)
        position_h = int(position_h*HEIGHT/1080)
        # print(position_w, position_h)
        self.position = position_w, position_h
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = self.position
        # print(self.rect)

        scenario.blit(self.image, self.position)


def main_app():

    pygame.mouse.set_visible(True)
    pygame.display.update()

    # Load the Environment
    env = Environment(process=PROCESS, 
                      use_text=useText, 
                      use_icon=useIcon)
    env.load_data()
    env.set_arrival_and_destination(arrival_id=int(arrivalNode), 
                                destination_id=int(destinationNode))
    env.reset()

    running = True
    while running:
        action = None
        appIsLoading = False

        # Get and display the scene
        scene, buttons = env.simulate()
        scenario.blit(convert_np_image(scene), (0, 0))

        # Create list of Buttons in this scene
        button_objects = dict()
        for button, bbox in buttons.items():
            print(button)
            top, left, bottom, right = bbox
            button_image = scene[top:bottom, left:right, :3]
            button_objects[button] = Button(image_array=button_image, 
                                            position=(left, top))
            if any(w in button.lower() for w in ['wait', 'load']):
                appIsLoading = True
                action = button
        
        # Apply changes
        pygame.display.update()

        # Check END-GAME
        print("Game Status:", env.end_simulator)
        if env.end_simulator:
            print('\n\n\nYou MUST press ESC to continue\n\n\n')
            RESET = False
            while not RESET:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            RESET = True
                            env.reset()

        # Wait if app in loading
        # if appIsLoading:
        #     time.sleep(0.7)

        # Read mouse click
        mouse_clicked = False
        key_pressed = False
        timeKeeper = time.time()
        while not (mouse_clicked or key_pressed or \
            (appIsLoading and time.time()-timeKeeper>0.3)):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    mouse_pressed = pygame.mouse.get_pressed()
                    left_pressed, middle_pressed, right_pressed = mouse_pressed
                    mouse_clicked = True
                elif event.type == pygame.KEYDOWN:
                    key_pressed = True
                    keys_striked = pygame.key.get_pressed()

        # Check which button was clicked
        if mouse_clicked:
            for button_name, button_obj in button_objects.items():
                if button_obj.rect.collidepoint(mouse_pos):
                    action = button_name
                    break

        # Check which key was stroken
        if key_pressed:
            print('\tKey pressed:', event.key)
            if event.key == pygame.K_RETURN:
                action = 'strikeKey_Enter'
            if event.key in [
                pygame.K_SPACE, pygame.K_F2,
                pygame.K_F3, pygame.K_F4, 
                pygame.K_F5, pygame.K_F6
            ]:
                action = 'label_Defect'

        if action is None:
            print('\t--> No action')
            continue
        elif any(ss in action for ss in ['wait', 'Wait']):
            time.sleep(0.3)

        # Take action to Environment
        reward = env.next_state(by_action=action)
        print('\t{} --> {}'.format(action, reward))

        # Check if the clicked button is the chosen one
        # icon = 'SnowFlake' if reward > 0 else 'GameOver'
        # icon_path = os.path.join(dir_path, 'app_resources', '{}.png'.format(icon))
        # scenario.blit(pygame.image.load(icon_path), (100, 200))
        # pygame.display.update()
        # time.sleep(0.3)

            
if __name__ == "__main__":

    # Initialize the app
    pygame.init()
    # pygame.mixer.init()

    # print(pygame.display.get_window_size())

    print_windows_info()
    main_app()
    









