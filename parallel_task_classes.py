import multiprocessing
from useful import *
from shape_classes import *
import bisect
import numpy as np
import matplotlib.pyplot as plt

class Cell_extractor(multiprocessing.Process):
    def __init__(self, cell_queue, partial_cell_queue, path, N_windows, window_event, extractor_finish_event):
        multiprocessing.Process.__init__(self)
        self.cell_queue = cell_queue  
        self.partial_cell_queue = partial_cell_queue  
        self.path = path
        self.N_windows = N_windows
        self.window_event = window_event
        self.extractor_finish_event = extractor_finish_event
    
    def run(self):
        self.name = multiprocessing.current_process().name

        image = read_input(self.path)
        im_row_length, im_col_length = image.shape
        num_window_row = 2
        num_window_col = 3
        win_row_length = int(im_row_length/num_window_row)
        win_col_length = int(im_col_length/num_window_col)
        
        total_windows = num_window_row*num_window_col

        i_windows = 0  # track how many windows have been processed
        
        for win_row_index in range(0,num_window_row):
            for win_col_index in range(0,num_window_col):
                # current_window = layout[win_row_index*win_row_length: (win_row_index+1)*win_row_length
                #                             ,win_col_index*win_col_length: (win_col_index+1)*win_col_length]
                start_row = win_row_index * win_row_length
                end_row = min((win_row_index + 1) * win_row_length, im_row_length)
                start_col = win_col_index * win_col_length
                end_col = min((win_col_index + 1) * win_col_length, im_col_length)
        
                current_window = image[start_row:end_row, start_col:end_col]
                
                # plt.imshow(current_window)
                # plt.show()
                img_bin = my_preprocessing(current_window)
                # plt.imshow(img_bin, cmap = 'gray')
                # plt.show()
                # perform prepcossing
                # layout, layout_denoised, layout_bin, th = pre_processing(layout_path)
                blobs_layout = connected_components(img_bin)
                window_coordinates = (win_row_index, win_col_index)
                window_partial_cells = Window_partial_cells( window_coordinates )

                for blob in blobs_layout:
                    blob_img = blob.image
                    # print(blob_img)
                    # plt.imshow(blob_img,cmap = 'gray')
                    # plt.show()
                    (min_row, min_col, max_row, max_col) = blob.bbox
                    # print(blob.bbox)
                    
                    pkt = shape_packet()
                    
                    pkt.bbox = (min_row+start_row, min_col+start_col, max_row+start_row, max_col+start_col) # we preserve coordinates 
                    pkt.blob_img = blob_img
                    # store centroid of shape coordinate in coordinate of original image
                    # print(start_row,start_col)
                    pkt.centroid = ( int( blob.centroid[0]+start_row), int(blob.centroid[1]+start_col)) # centroid from skii_image is wrt window image coordinates
                    # print(pkt.centroid)
                    # print(blob_img.shape)
                    
                    if ( max_col == win_col_length ) :
                        pkt.packet_type = 4
                        window_partial_cells.type_4_pkts.append(pkt)
                        # # print("found type",pkt.packet_type)
                    elif ( min_col == 0 ) :
                        pkt.packet_type = 8
                        window_partial_cells.type_8_pkts.append(pkt)
                        # print("found type",pkt.packet_type)
                    else:
                        # shape_pkts.append(pkt)
                        self.cell_queue.put(pkt)
                        # print("found type",pkt.packet_type)
                
                self.partial_cell_queue.put(window_partial_cells)
                
                i_windows = i_windows + 1

                if(i_windows == self.N_windows):
                    self.window_event.set()
                    i_windows = 0
        
        self.extractor_finish_event.set() # signal end of processing all the windows
        # print(f"{self.name} {self.pid}: all packets sent. Now exiting")


class Cell_collector(multiprocessing.Process):
    def __init__(self, cell_queue, sorted_cell_queue, N_cells, window_event, extractor_finish_event, merger_finish_event): 
        multiprocessing.Process.__init__(self)
        self.cell_queue = cell_queue  
        self.sorted_cell_queue = sorted_cell_queue
        self.N_cells = N_cells
        self.window_event = window_event
        self.extractor_finish_event = extractor_finish_event
        self.merger_finish_event = merger_finish_event
        self.sorted_cell_list = []
    
    def run(self):
        self.name = multiprocessing.current_process().name

        while True:
            # if self.window_event.is_set():
            #     self.window_event.clear()                 
            #     if len(self.sorted_cell_list) == 0:
            #         pass
            #     elif len(self.sorted_cell_list)> 0 and len(self.sorted_cell_list) < self.N_cells:
            #         self.sorted_cell_queue.put(self.sorted_cell_list)
            #         self.sorted_cell_list = []
            #     else:
            #         self.sorted_cell_queue.put(self.sorted_cell_list[:self.N_cells])
            #         del self.sorted_cell_list[:self.N_cells]
            
            # # el
            # if self.extractor_finish_event.is_set():
            #     print(len(self.sorted_cell_list))
            #     self.sorted_cell_queue.put(self.sorted_cell_list)
            #     self.sorted_cell_list = []
            #     self.extractor_finish_event.clear()
                
            #     # self.sorted_cell_queue.put(self.sorted_cell_list[:self.N_cells])
            #     # del self.sorted_cell_list[:self.N_cells]
                         
            
            # el
            if not self.cell_queue.empty():
                pkt = self.cell_queue.get()
                bisect.insort(self.sorted_cell_list , pkt)
                print(len(self.sorted_cell_list))
                if len(self.sorted_cell_list ) == 8:
                    for xx in self.sorted_cell_list:
                        plt.imshow( xx.blob_img )
                        plt.show()  
            # print(f"{self.name}: {self.pid} Cell received with centroid= {pkt.centroid}")
            # plt.imshow( pkt.blob_img )
            # plt.show()

class Cell_merger(multiprocessing.Process):
    def __init__(self, partial_cell_queue, cell_queue, extractor_finish_event, merger_finish_event):
        multiprocessing.Process.__init__(self)
        self.partial_cell_queue = partial_cell_queue  
        self.cell_queue = cell_queue  
        self.extractor_finish_event = extractor_finish_event
        self.merger_finish_event = merger_finish_event
    
    def run(self):
        self.name = multiprocessing.current_process().name

        while True:

            if self.extractor_finish_event.is_set(): 
                if self.partial_cell_queue.empty(): # if queue is empty, signal finish event and terminate the process, else continue
                    self.merger_finish_event.set()
                    break
                else:
                    window_partial_cells = self.partial_cell_queue.get()
                    window_coordinates = window_partial_cells.window_coordinates
                    
                    if window_coordinates[1] == 0:    # Check if it is start of a row 
                        window_x = window_partial_cells
                    else:
                        window_x_1 = window_partial_cells
                        self.merge_cells(window_x, window_x_1)
                        window_x = window_x_1
            else:
                window_partial_cells = self.partial_cell_queue.get()
                window_coordinates = window_partial_cells.window_coordinates
                
                if window_coordinates[1] == 0:    # Check if it is start of a row 
                    window_x = window_partial_cells
                else:
                    window_x_1 = window_partial_cells
                    self.merge_cells(window_x, window_x_1)
                    window_x = window_x_1
            # print(f"{self.name}: {self.pid} Partial cell received with window coordinate= {window_partial_cells.window_coordinates}")
            # plt.imshow( pkt.blob_img )
            # plt.show()
    
    def merge_cells( self, p1, p2 ):
        
        for type_4_pkt in p1.type_4_pkts:
            type_4_img = type_4_pkt.blob_img
            # print(type_4_img)
            type_4_img_rows, type_4_img_cols = type_4_img.shape
            type_4_right_edge_min_row = float('inf')
            type_4_right_edge_max_row = float('-inf')
            # get the min & max row values of the pixels touching the right edge of the image
            for row_index in range(type_4_img_rows):
                if type_4_img[row_index][-1]:  # Check the last column of each row
                    type_4_right_edge_max_row = max(type_4_right_edge_max_row, row_index)
                    type_4_right_edge_min_row = min(type_4_right_edge_min_row, row_index)
            
            # print(" type4 org coordinate",type_4_right_edge_min_row,type_4_right_edge_max_row)
            # plt.imshow(type_4_img)
            # # print(type_4_img)
            # plt.show()
            # print(type_4_pkt.bbox)
                    
            # Shift to original coordinates
                    
            type_4_bbox_min_row, type_4_bbox_min_col, type_4_bbox_max_row, type_4_bbox_max_col = type_4_pkt.bbox
            type_4_right_edge_min_row = type_4_right_edge_min_row + type_4_bbox_min_row
            type_4_right_edge_max_row = type_4_right_edge_max_row + type_4_bbox_min_row
    
            # print(" type4 shifted coordinate",type_4_right_edge_min_row, type_4_right_edge_max_row)
            
            for type_8_pkt in p2.type_8_pkts[:]:
                type_8_img = type_8_pkt.blob_img
                type_8_img_rows, type_8_img_cols = type_8_img.shape
    
                # plt.imshow(type_8_img)
                # plt.show()
                
                type_8_left_edge_min_row = float('inf')
                type_8_left_edge_max_row = float('-inf')
                # get the min & max row values of the pixels touching the right edge of the image
                for row_index in range(type_8_img_rows):
                    if type_8_img[row_index][0]:  # Check the first column of each row
                        type_8_left_edge_max_row = max(type_8_left_edge_max_row, row_index)
                        type_8_left_edge_min_row = min(type_8_left_edge_min_row, row_index)
                
                # print("type 8 org",type_8_left_edge_min_row, type_8_left_edge_max_row)
                # Shift to original coordinates
                type_8_bbox_min_row, type_8_bbox_min_col , type_8_bbox_max_row, type_8_bbox_max_col = type_8_pkt.bbox
                type_8_left_edge_min_row = type_8_left_edge_min_row + type_8_bbox_min_row
                type_8_left_edge_max_row = type_8_left_edge_max_row + type_8_bbox_min_row
    
                # print("type8 shifted",type_8_left_edge_min_row, type_8_left_edge_max_row)
    
                #Check merging condition
                if (np.abs(type_4_right_edge_min_row - type_8_left_edge_min_row) <= 3) and (np.abs(type_4_right_edge_max_row - type_8_left_edge_max_row) <= 3):
                    
                    # print("match found")
    
                    # merge packets
                    merged_pkt =  shape_packet()
    
                    # new bounding box
                    merged_bbox_min_row = min( type_4_bbox_min_row, type_8_bbox_min_row)
                    merged_bbox_max_row = max( type_4_bbox_max_row, type_8_bbox_max_row)
                    merged_bbox_min_col = type_4_bbox_min_col
                    merged_bbox_max_col = type_8_bbox_max_col
    
                    merged_pkt.bbox = (merged_bbox_min_row,merged_bbox_min_col,merged_bbox_max_row,merged_bbox_max_col)
    
                    #new merged image
                    merged_image = np.zeros((merged_bbox_max_row-merged_bbox_min_row, merged_bbox_max_col-merged_bbox_min_col), dtype=bool)
    
                    row_diff = type_4_bbox_min_row - type_8_bbox_min_row
                    if ( row_diff > 0 ):
                        merged_image[ row_diff: row_diff+ type_4_img_rows, : type_4_img_cols] = type_4_img
                        merged_image[ : type_8_img_rows, type_4_img_cols: ] = type_8_img
                    else:
                        merged_image[ : type_4_img_rows, : type_4_img_cols ] = type_4_img
                        merged_image[ np.abs(row_diff): np.abs(row_diff)+ type_8_img_rows,  type_4_img_cols: ] = type_8_img
    
                    merged_pkt.blob_img = merged_image
                    merged_pkt.set_centroid()
                    
                    self.cell_queue.put(merged_pkt)

                    # shape_pkts.append(merged_pkt)
                    
                    # plt.imshow(merged_pkt.blob_img)
                    # plt.show()
                    
                    p2.type_8_pkts.remove(type_8_pkt)
                    
                    break

class Cell_validation(multiprocessing.Process):
    def __init__(self, sorted_cell_queue_layout, sorted_cell_queue_SEM):
        multiprocessing.Process.__init__(self)
        self.sorted_cell_queue_layout = sorted_cell_queue_layout  
        self.sorted_cell_queue_SEM = sorted_cell_queue_SEM  
    
    def run(self):
        self.name = multiprocessing.current_process().name

        while True:
            list = self.sorted_cell_queue_layout.get()
            print(f"{self.name}: {self.pid} list received with length= {len(list)}")
            # for pkt in list:
            #     plt.imshow(pkt.blob_img)
            #     plt.show()