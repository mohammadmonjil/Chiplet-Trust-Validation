from fourier import *
import cv2
from PIL import Image
from useful import *
import numpy as np

class shape_packet:

    def __init__(self):
        self.horizontal_threshold = 20
        self.packet_type = 0
        self.bbox = (0,0,0,0)
        self.blob_img = []
        self.centroid = []
        self.fourier = []
        self.num_fourier_coeff = 40

    def set_shape_descriptor(self):
        contour, _ = cv2.findContours(self.blob_img.astype(np.uint8) * 255 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gx,gy = sample_polygon_uniformly(contour[0],50)
        contour_sampled = np.column_stack((gx, gy)).astype(np.int32)
        self.fourier = fourier_descriptors(contour_sampled, self.num_fourier_coeff) 
        self.fourier[0] = 0 # make translation invariant
        self.fourier = self.fourier/ np.sqrt( np.sum( (np.abs(self.fourier))**2) ) #make scale invariant
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

class Adaptive_window:
    def __init__(self, path, win_col_length):
        self.image = Image.open(path)
        _, self.im_row_length  = self.image.size
        self.horizontal_hop = 200
        self.th = 50
        self.start_row = 0
        self.end_row = self.start_row + self.horizontal_hop
        self.start_col = 0
        self.end_col = self.start_col + win_col_length
        self.is_this_last_rows = False
        self.first_rows_done =  False
        self.cell_width = 0
    
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