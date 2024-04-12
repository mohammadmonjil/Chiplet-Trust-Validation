import multiprocessing
import threading
from useful import *
from shape_classes import *
import bisect
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fourier import *
import os

layout_name = "36.png"  
layout_path = os.path.join( os.getcwd(), 'images', layout_name)
image = Image.open(layout_path)
im_col_length, im_row_length  = image.size
num_window_row = 9
num_window_col = 3
win_row_length = int(im_row_length/num_window_row)
win_col_length = int(im_col_length/num_window_col)

# total_windows = num_window_row*num_window_col
adaptive_window = Adaptive_window(layout_path, win_col_length,num_window_col)

while True:
    current_window, win_col_index, start_row, end_row, start_col, end_col, one_row_done, last_row_flag = adaptive_window.get_current_window()
    if last_row_flag:
        break   