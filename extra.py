from PIL import Image
from useful import *
import numpy as np

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