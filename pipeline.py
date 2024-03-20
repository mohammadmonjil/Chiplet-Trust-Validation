import numpy as np

import os
# import parameters as params
import multiprocessing

import time
import random

# from components import *
from skimage import measure
import matplotlib.pyplot as plt

from fourier import *
from parallel_task_classes import *

if __name__ == '__main__':

    # Define parameters
    layout_name = "26.png"  # Replace with your image path 
    layout_path = os.path.join( os.getcwd(), 'images', layout_name)
    N_cells = 2
    N_windows = 1
    
    # Establish communication queues.
    cell_queue = multiprocessing.Queue()
    partial_cell_queue = multiprocessing.Queue()
    sorted_cell_queue_layout = multiprocessing.Queue()
    sorted_cell_queue_SEM = multiprocessing.Queue()
    window_event = multiprocessing.Event()
    extractor_finish_event = multiprocessing.Event()
    merger_finish_event = multiprocessing.Event()
    
    cell_extractor_layout = Cell_extractor(cell_queue, partial_cell_queue, layout_path, N_windows, window_event, extractor_finish_event)
    cell_collector_layout = Cell_collector(cell_queue, sorted_cell_queue_layout, N_cells, window_event, extractor_finish_event, merger_finish_event)
    cell_merger_layout = Cell_merger(partial_cell_queue, cell_queue, extractor_finish_event, merger_finish_event)

    cell_validation = Cell_validation(sorted_cell_queue_layout, sorted_cell_queue_SEM)
    
    cell_extractor_layout.start()
    cell_collector_layout.start()
    cell_merger_layout.start()
    cell_validation.start()
    
    cell_extractor_layout.join()
    cell_collector_layout.join()
    cell_merger_layout.join()