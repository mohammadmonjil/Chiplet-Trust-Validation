from fourier import *
import cv2
from PIL import Image
from useful import *
import numpy as np
import os

class shape_packet:

    def __init__(self):
        self.horizontal_threshold = 20
        self.packet_type = 0
        self.bbox = (0,0,0,0)
        self.blob_img = []
        self.centroid = []
        self.fourier = []
        self.num_fourier_coeff = 10

    def set_shape_descriptor(self):
        contour, _ = cv2.findContours(self.blob_img.astype(np.uint8) * 255 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gx,gy = sample_polygon_uniformly(contour[0],100)
        contour_sampled = gx + gy*1.0j
        self.fourier = fourier_descriptors(contour_sampled, self.num_fourier_coeff) 
        self.fourier[0] = 0 # make translation invariant
        self.fourier = self.fourier/ np.sqrt( np.sum( (np.abs(self.fourier))**2) ) # make scale invariant
        self.Ga, self.Gb = make_start_point_invariant(self.fourier)
        # self.blob_img = []
        # self.fourier = []

    def set_centroid(self):
        total_rows, total_cols = self.blob_img.shape

        # Calculate the sum of x and y coordinates
        sum_x = 0
        sum_y = 0
        total_points = 0
    
        for row in range(total_rows):
            for col in range(total_cols):
                if self.blob_img[row][col]:
                    sum_y += col
                    sum_x += row
                    total_points += 1
    
        # Avoid division by zero
        if total_points == 0:
            return None, None
        
        self.centroid = (int(sum_x / total_points + self.bbox[0]), int(sum_y / total_points+self.bbox[1]))

    def __eq__(self, other):
        return self.centroid == other.centroid

    def __lt__(self, other):
        if abs(other.centroid[0] - self.centroid[0]) <= self.horizontal_threshold:
            if self.centroid[1] - other.centroid[1] < 0 :
                return True
            else :
                return False
        elif other.centroid[0] - self.centroid[0] > self.horizontal_threshold:
            return True
        elif self.centroid[0] - other.centroid[0] > self.horizontal_threshold:
            return False


class Window_partial_cells:
    def __init__(self, window_coordinates):
        self.type_4_pkts = []
        self.type_8_pkts = []
        self.window_coordinates = window_coordinates

# class Window:
#     def __init__(self) :
#         pass

class No_window:
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = os.listdir(self.data_path)
        self.file_index = 0
    def get_current_window(self):
        image_path = os.path.join(self.data_path, self.files[self.file_index])
        img = Image.open(image_path)
        current_window = np.array(img)

        if current_window.ndim == 3:
            current_window = my_preprocessing(current_window[:,:,0])
            
        self.file_index = self.file_index + 1

        if self.file_index > len(self.files)-1:    
            return current_window, True
        else:
            return current_window, False

class Fixed_window:
    def __init__(self, path):
        pass

class Adaptive_window:
    def __init__(self, path, num_window_col):
        
        self.image = Image.open(path)
        im_col_length, _  = self.image.size
        self.num_window_col = num_window_col
        self.win_col_length = int(im_col_length/self.num_window_col)

        self.im_col_length, self.im_row_length  = self.image.size
        self.horizontal_hop = 200
        self.th = 50
        self.start_row = 0
        self.end_row = self.start_row + self.horizontal_hop
        self.start_col = 0
        self.end_col = self.start_col + self.win_col_length
        self.is_this_last_rows = False
        self.first_rows_done =  False
        self.cell_width = 0

        self.win_col_index = 0
        self.first_call = True

        self.curret_start_row, self.current_end_row, self.current_last_row_flag = self.get_window_rows()

    def get_current_window(self):
            

        start_col = self.win_col_index * self.win_col_length
        end_col = min((self.win_col_index + 1) * self.win_col_length, self.im_col_length)
        current_window = np.array( self.image.crop(( start_col, self.curret_start_row, end_col, self.current_end_row, )) )[:,:,0]
        current_window = my_preprocessing(current_window)

        temp_win_col_index = self.win_col_index
        temp_curret_start_row = self.curret_start_row
        temp_current_end_row = self.current_end_row
        last_window = False
        
        self.win_col_index = self.win_col_index + 1

        if self.win_col_index > self.num_window_col - 1:
            self.win_col_index = 0
            one_row_done = True
        
            if self.current_last_row_flag:
                last_window = True
            else:
                self.curret_start_row, self.current_end_row, self.current_last_row_flag = self.get_window_rows()

        else:
            one_row_done = False

        return current_window, temp_win_col_index, temp_curret_start_row, temp_current_end_row, start_col, end_col, one_row_done, last_window
        
    def get_window_rows(self):

        if self.first_rows_done:
            self.end_row = self.start_row + self.cell_width
        else:
            self.end_row = self.start_row + self.horizontal_hop
            
        if self.end_row >= self.im_row_length:
            self.end_row = self.im_row_length
            self.is_this_last_rows = True
            return self.start_row, self.end_row, self.is_this_last_rows
            
        while True:
            window_bin = my_preprocessing( np.array( self.image.crop(( self.start_col, self.start_row, self.end_col, self.end_row )) )[:,:,0] )

            if sum( window_bin[self.end_row - self.start_row -1,:] ) == 0:
                window_found = True
                break
            else:                
                self.end_row = self.end_row + 1
                if self.end_row >= self.im_row_length:
                    self.end_row = self.im_row_length
                    self.is_this_last_rows = True
                    break
        
        temp_start_row = self.start_row
        temp_end_row = self.end_row

        # store for later iterations
        self.start_row = self.end_row + 1                            
        # self.end_row = self.start_row + temp_end_row - temp_start_row
        
        if not self.first_rows_done:
            self.first_rows_done = True
            self.cell_width = temp_end_row - temp_start_row
            
        return temp_start_row, temp_end_row, self.is_this_last_rows