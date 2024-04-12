import multiprocessing
import threading
from useful import *
from shape_classes import *
import bisect
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fourier import *

class Cell_extractor(multiprocessing.Process):
    def __init__(self, type_name, cell_queue, partial_cell_queue, path, row_done_event):
        multiprocessing.Process.__init__(self)
        self.type_name = type_name
        self.cell_queue = cell_queue  
        self.partial_cell_queue = partial_cell_queue  
        self.path = path
        self.row_done_event = row_done_event
    
    def run(self):
        self.name = multiprocessing.current_process().name + self.type_name
        
####################### No Window Direct Image START #########################################
        window = No_window(self.path)

        i_cells = 0

        while True:
            current_window, last_window = window.get_current_window()
            blobs_layout = connected_components(current_window)
            

            for blob in blobs_layout:
                pkt = shape_packet()
                pkt.bbox = blob.bbox
                pkt.blob_img = blob.image
                pkt.centroid = ( int( blob.centroid[0]), int(blob.centroid[1]))
                pkt.set_shape_descriptor()
                self.cell_queue.put(pkt)
                i_cells = i_cells + 1

            self.row_done_event.set()

            if last_window:
                break
        
        self.cell_queue.put(None)
        self.partial_cell_queue.put(None)
        print(f'\n{self.name} Total cells extracted {i_cells}') 

####################### No Window Direct Image END ###################################

####################### Adaptive Window START #########################################
        # image = Image.open(self.path)
        # im_col_length, _  = image.size
        # num_window_col = 3
        # win_col_length = int(im_col_length/num_window_col)
        
        # window = Adaptive_window(self.path, num_window_col)

        # i_cells = 0 # track how many cells have been collected for the current row. Currently counting complete cells and partial cells of type 8.
        
        # while True:
        #     current_window, win_col_index, start_row, _ , start_col, _ , one_row_done, last_window = window.get_current_window()        
        #     blobs_layout = connected_components(current_window)
        #     window_partial_cells = Window_partial_cells( win_col_index )

        #     for blob in blobs_layout:
        #         blob_img = blob.image
        #         (min_row, min_col, max_row, max_col) = blob.bbox
                
        #         pkt = shape_packet()
                
        #         pkt.bbox = (min_row+start_row, min_col+start_col, max_row+start_row, max_col+start_col) # we preserve coordinates 
        #         pkt.blob_img = blob_img
        #         # store centroid of shape coordinate in coordinate of original image
        #         pkt.centroid = ( int( blob.centroid[0]+start_row), int(blob.centroid[1]+start_col)) # centroid from skii_image is wrt window image coordinates

                
        #         if ( max_col == win_col_length ) :
        #             pkt.packet_type = 4
        #             window_partial_cells.type_4_pkts.append(pkt)
        #         elif ( min_col == 0 ) :
        #             pkt.packet_type = 8
        #             window_partial_cells.type_8_pkts.append(pkt)
                    
        #         else:
        #             pkt.set_shape_descriptor()
        #             self.cell_queue.put(pkt)
        #             i_cells = i_cells + 1
            
        #     self.partial_cell_queue.put(window_partial_cells)

        #     if one_row_done:
        #         self.row_done_event.set() # one row of the windows have been processed
        #     if last_window:
        #         break
        
        # self.cell_queue.put(None)
        # self.partial_cell_queue.put(None)
        # print(f'\n{self.name} Total cells extracted {i_cells}')   

####################### Adaptive Window END #########################################

class Cell_collector(multiprocessing.Process):
    def __init__(self, type_name, cell_queue, sorted_cell_queue,
                 row_done_event, collector_done_event): 
        multiprocessing.Process.__init__(self)
        self.type_name = type_name
        self.cell_queue = cell_queue  
        self.sorted_cell_queue = sorted_cell_queue
        self.row_done_event = row_done_event
        self.collector_done_event = collector_done_event
        self.sorted_cell_list = []

    def run(self):
        
        self.name = multiprocessing.current_process().name + self.type_name
        
        total_count = 0
        None_received = 0
        
        while True:

            if not self.cell_queue.empty():
                pkt = self.cell_queue.get()

                if pkt is None:
                    None_received = None_received + 1
                else:
                    bisect.insort(self.sorted_cell_list , pkt)

            if self.row_done_event.is_set():
                self.row_done_event.clear()     

                if not self.cell_queue.empty():
                    pkt = self.cell_queue.get()
                    
                    if pkt is None:
                        None_received = None_received + 1
                    else:
                        bisect.insort(self.sorted_cell_list , pkt)

                cell_count = len(self.sorted_cell_list)
                total_count = total_count + cell_count
                
                if cell_count > 0:
                    self.sorted_cell_queue.put(self.sorted_cell_list[:cell_count])
                    del self.sorted_cell_list[:cell_count]

            if None_received ==2:
                cell_count = len(self.sorted_cell_list)
                total_count = total_count + cell_count

                if cell_count > 0:
                    self.sorted_cell_queue.put(self.sorted_cell_list[:cell_count])
                    del self.sorted_cell_list[:cell_count]
                break
        
        
        self.sorted_cell_queue.put(None)
        self.collector_done_event.set()
        print(f'\n {self.name} is now exiting, total cells collected = {total_count}')

class Cell_merger(multiprocessing.Process):
    def __init__(self, type_name, partial_cell_queue, cell_queue):
        multiprocessing.Process.__init__(self)
        self.type_name = type_name
        self.partial_cell_queue = partial_cell_queue  
        self.cell_queue = cell_queue  
        self.merged_cells = 0
    
    def run(self):
        self.name = multiprocessing.current_process().name + self.type_name

        while True:
            if not self.partial_cell_queue.empty(): 
                window_partial_cells = self.partial_cell_queue.get()

                if window_partial_cells is None:
                    break
                window_coordinates = window_partial_cells.window_coordinates
                
                if window_coordinates == 0:    # Check if it is start of a row 
                    window_x = window_partial_cells
                else:
                    window_x_1 = window_partial_cells
                    self.merge_cells(window_x, window_x_1)
                    window_x = window_x_1
        
        self.cell_queue.put(None)
        print(f'\n{self.name} total cells merged = {self.merged_cells}')
    
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
                    merged_pkt.set_shape_descriptor()
                    self.cell_queue.put(merged_pkt)
                    self.merged_cells = self.merged_cells + 1
                    # print(f'{self.name} merged and sent 1 packet')
                    
                    # shape_pkts.append(merged_pkt)
                    
                    # plt.imshow(merged_pkt.blob_img)
                    # plt.show()
                    
                    p2.type_8_pkts.remove(type_8_pkt)
                    
                    break

class Cell_validation(multiprocessing.Process):

    def __init__(self, sorted_cell_queue_layout, sorted_cell_queue_SEM, 
                                      collector_done_event_layout, collector_done_event_SEM):
        
        multiprocessing.Process.__init__(self)
        self.sorted_cell_queue_layout = sorted_cell_queue_layout  
        self.sorted_cell_queue_SEM = sorted_cell_queue_SEM  
        self.collector_done_event_layout = collector_done_event_layout
        self.collector_done_event_SEM = collector_done_event_SEM

        self.cell_list_SEM = []
        self.cell_list_layout = []
        self.rcv_cells_layout_thread_running = False
        self.rcv_cells_SEM_thread_running = False
        self.rcv_cells_layout_thread = threading.Thread(target= self.rcv_cells, args =(self.sorted_cell_queue_layout, self.cell_list_layout, 
                                                                                      'layout' ) )
        self.rcv_cells_SEM_thread = threading.Thread(target=self.rcv_cells, args =(self.sorted_cell_queue_SEM, self.cell_list_SEM, 
                                                                                   'SEM' ))
        self.threshold = 0.5
        self.rows = 0
        self.report = open('report.txt', 'w')
        self.report.write('Report')
        self.total_cells_layout = 0
        self.total_cells_SEM = 0
        self.cell_chunk = 10

    
    def run(self):
        self.name = multiprocessing.current_process().name
        
        self.rcv_cells_layout_thread.start()
        self.rcv_cells_layout_thread_running = True
        self.rcv_cells_SEM_thread.start()
        self.rcv_cells_SEM_thread_running = True

        while True:
            if not self.rcv_cells_layout_thread_running and not self.rcv_cells_SEM_thread_running:
                N_cells_SEM = len(self.cell_list_SEM)
                N_cells_layout = len(self.cell_list_layout)
                self.total_cells_layout = self.total_cells_layout + N_cells_layout
                self.total_cells_SEM = self.total_cells_SEM + N_cells_SEM

                if N_cells_layout == N_cells_SEM:
                    self.shape_validate( self.cell_list_layout, self.cell_list_SEM)
                break
            else:
                self.validate()
        self.report.write(f'\nTotal cells in layout = {self.total_cells_layout}')
        self.report.write(f'\nTotal cells in SEM = {self.total_cells_SEM}')
        self.report.close()

    def validate(self):
        
        if len(self.cell_list_layout) > self.cell_chunk and len(self.cell_list_SEM) > self.cell_chunk:
            self.total_cells_layout = self.total_cells_layout + self.cell_chunk 
            self.total_cells_SEM = self.total_cells_SEM + self.cell_chunk 
            self.shape_validate( self.cell_list_layout[:self.cell_chunk], self.cell_list_SEM[:self.cell_chunk])
            del self.cell_list_layout[:self.cell_chunk]
            del self.cell_list_SEM[:self.cell_chunk]

    def shape_validate(self, cell_list_layout, cell_list_SEM ):

        for pkt_layout, pkt_SEM in zip(cell_list_layout, cell_list_SEM):
            # pass
            # f_similarity = calculate_similarity(pkt_layout.fourier, pkt_SEM.fourier)
            distance =  min ( complex_l2( pkt_layout.Ga, pkt_SEM.Ga) , complex_l2( pkt_layout.Ga, pkt_SEM.Gb) ,
                             complex_l2( pkt_layout.Gb, pkt_SEM.Ga) , complex_l2( pkt_layout.Gb, pkt_SEM.Gb) )
            # # print(f_similarity)
            # self.report.write(f'\n Distance ={distance}')
            if distance > self.threshold:
                # pass
                self.report.write(f'\nCell modification detected')
                # self.report.write(f'\n Distance ={distance}')
                # plt.imshow(pkt_SEM.blob_img)
                # plt.show()

    def rcv_cells(self, cell_queue, cell_list, type_name):
        
        while True:
            if not cell_queue.empty():
                pkts = cell_queue.get()
                if pkts is None:
                    if type_name == 'layout':
                        self.rcv_cells_layout_thread_running = False
                    elif type_name == 'SEM':
                        self.rcv_cells_SEM_thread_running = False
                    break
                else:
                    cell_list.extend(pkts) 

