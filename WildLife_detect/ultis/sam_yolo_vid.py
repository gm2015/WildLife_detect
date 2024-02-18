import ultralytics
#from IPython.display import display, Image
#from roboflow import Roboflow
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

# Set the device for GPU acceleration
device = "cuda"

# Check Ultralytics version and setup completion
ultralytics.checks()

# Set the first_run flag to False after the initial run
first_run = False