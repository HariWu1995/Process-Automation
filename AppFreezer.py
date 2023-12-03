import os
import sys
import copy
import shutil

import time
import datetime

from glob import glob
from tqdm import tqdm

import cx_Freeze


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, file_path)

executables = [
    cx_Freeze.Executable("AppSimulator.py")
]

cx_Freeze.setup(
    name="UTL App Simulator",
    version="1.0.0",
    author="hari.wu.95@gmail.com",
    description="Simulate UTL App for Training D3RQL",
    options={
        "build_exe": {
            "packages": ["pygame"],
            "include_files": ["app_resources/"]
        }
    },
    executables=executables
    )

### [CMD] python AppFreezer.py build ###










