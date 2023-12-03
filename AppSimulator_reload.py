import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)
# os.environ['SDL_VIDEO_CENTERED'] = '0'

# windows_offset = 0

import sys
import shutil

import ctypes

import time
import datetime

from glob import glob
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import math
import random

import itertools
from collections import defaultdict, deque

import pygame
from pygame.locals import *


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, file_path)


# Initialize the app
pygame.init()
# pygame.mixer.init()


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


class Button(pygame.sprite.Sprite):

    def __init__(self, image, position, activate=True):
        super().__init__()

        self.image = image
        self.position = position
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = self.position
        self.activate = activate

        scenario.blit(self.image, self.position)


resource_root = "app_resources\\"
scene_root = resource_root + "scenes\\"
button_root = resource_root + "button_{}\\"


"""
Define location of buttons for each scene
"""
button_position = dict()
button_position["StartSelection"] = dict()
button_position["StartSelection"]["LotSearch"] = [1815, 150]

button_position["LotSearch"] = dict()
button_position["LotSearch"]["LotId"] = [795, 400]
button_position["LotSearch"]["LotSearch"] = [1815, 150]

button_position["LotId"] = dict()
button_position["LotId"]["LotId"] = [795, 400]
button_position["LotId"]["LotSearch"] = [1815, 150]

button_position["StartReview"] = dict()
button_position["StartReview"]["Finish"] = [1405, 745]
button_position["StartReview"]["FitSize"] = [ 10, 265]
button_position["StartReview"]["ListView"] = [585, 745]

button_position["FitSize"] = dict()
button_position["FitSize"]["Finish"] = [1020, 790]
button_position["FitSize"]["FitSize"] = [  10, 265]
button_position["FitSize"]["ListView"] = [1015, 745]

button_position["ListView"] = dict()
button_position["ListView"]["Finish"] = [1730, 745]
button_position["ListView"]["FitSize"] = [ 10, 265]
button_position["ListView"]["ListView"] = [910, 745]

button_position["WaferReview"] = dict()
button_position["WaferReview"]["Finish"] = [1730, 745]
button_position["WaferReview"]["FitSize"] = [ 10, 265]
button_position["WaferReview"]["ListView"] = [910, 745]
button_position["WaferReview"]["DefectZone"] = [40, 170] #size=855x855


"""
Define activation of buttons for each scene
"""
action = dict()
action["StartSelection"] = "LeftClick_LotSearch"
action["LotSearch"     ] = "LeftClick_LotId"
action["LotId"         ] =  "PressKey_Enter"
action["StartReview"   ] = "LeftClick_FitSize"
action["FitSize"       ] = "LeftClick_ListView"
action["ListView"      ] =     "Label_Defect"
action["WaferReview"   ] = "LeftClick_Finish"


scenes = [
    # "StartSelection", "LotSearch", "LotId",
    # "StartReview", "FitSize", "ListView", 
    "WaferReview"
]


width, height = 1920, 1080
scenario = pygame.display.set_mode((width, height))
scenario.fill([255, 255, 255])  # clear the screen

# print(pygame.display.get_window_size())


def load_defects(path):
    print("Load a set of defects")
    defects = os.listdir(path)
    random.shuffle(defects)
    defects = deque(defects)
    return defects


def main_app():

    pygame.mouse.set_visible(True)
    pygame.display.update()

    label_keystrikes = [pygame.K_SPACE, pygame.K_F2, pygame.K_F3, 
                        pygame.K_F4, pygame.K_F5, pygame.K_F6]
    load_new_defect = True
    scene_change = True
    scene_idx = -1
    running = True
    defects = None
    while running:
        if scene_change:
            scene_idx = 0 if scene_idx==len(scenes)-1 else scene_idx+1
            scene_change = False

            background_scene = scenes[scene_idx]
            print("[background_scene]", background_scene)

            # Load background
            scenario.blit(
                pygame.image.load(
                    scene_root+"{}.png".format(background_scene)
                ), (0, 0)
            )

            # Create list of Buttons in this scene
            button_objects = dict()
            for button in button_position[background_scene].keys():
                button_folder = deepcopy(button_root).format(background_scene)
                if button == "DefectZone":
                    continue
                else:
                    button_path = button_folder + "{}.png".format(button)
                    button_image = pygame.image.load(button_path).convert()
                button_objects[button] = Button(
                    button_image,
                    button_position[background_scene][button]
                )

        if "DefectZone" in list(button_position[background_scene].keys()):
            if load_new_defect:
                load_new_defect = False
                if defects is None:
                    defects = load_defects(path=button_root.format("WaferReview")+"defects\\")
                if len(defects) <= 0:
                    defects = load_defects(path=button_root.format("WaferReview")+"defects\\")
                current_defect = defects.popleft()
                button_path = button_folder + "defects\\" + current_defect
                button_image = pygame.image.load(button_path).convert()
                button_image = pygame.transform.scale(button_image, (855, 855))
                button_objects["DefectZone"] = Button(
                    button_image,
                    button_position[background_scene][button]
                )

        print("\tLoading menu buttons", \
            " - ".join("{}".format(k) for k in button_objects.keys()))

        # Apply changes
        # pygame.display.flip()
        pygame.display.update()

        # Read mouse click
        mouse_clicked = False
        key_pressed = False
        while not (mouse_clicked or key_pressed):
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
                    key_striked = pygame.key.get_pressed()
        print("\tKey: {} - Mouse: {}".format(key_pressed, mouse_clicked))

        # Check RESET
        if key_pressed:
            try:
                if event.key == pygame.K_ESCAPE:
                    scene_idx = -1
                    scene_change = True
                elif event.key == pygame.K_PRINT:
                    running = False
            except Exception as e:
                print(e)

        # Check right action was taken
        if mouse_clicked:
            if "Click" in action[background_scene]:
                if ("Left" in action[background_scene] and left_pressed) \
                or ("Right" in action[background_scene] and right_pressed) \
                or ("Middle" in action[background_scene] and middle_pressed):
                    for button_name, button_obj in button_objects.items():
                        if button_name in action[background_scene] \
                        and button_obj.rect.collidepoint(mouse_pos):
                            print("Button {} was clicked".format(button_name))
                            scene_change = True
                            break
        elif key_pressed:
            print("Key {} was striked".format(event.key))
            if "Press" in action[background_scene]:
                if "Enter" in action[background_scene] and event.key==pygame.K_RETURN:
                    scene_change = True
            # elif "Label" in action[background_scene]:
            if event.key in label_keystrikes:
                # Load defects
                # defects = load_defects(path=button_root.format("WaferReview")+"defects\\")
                # scene_change = True
                load_new_defect = True

        # Apply changes
        pygame.display.update()


if __name__ == "__main__":
    # print_windows_info()
    main_app()
