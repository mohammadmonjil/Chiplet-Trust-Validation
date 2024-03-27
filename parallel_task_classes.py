import multiprocessing
import threading
from useful import *
from shape_classes import *
import bisect
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from fourier import *

class Cell_extractor(multiprocessing.Process):
    def __init__(self, type_name, cell_queue, partial_cell_queue, path, extractor_done_event, row_done_event, extractor_conn):
        multiprocessing.Process.__init__(self)
        self.type_name = type_name
        self.cell_queue = cell_queue  
        self.partial_cell_queue = partial_cell_queue  
        self.path = path
        self.extractor_done_event = extractor_done_event
        self.row_done_event = row_done_event
        self.extractor_conn = extractor_conn
    
    def run(self):
        self.name = multiprocessing.current_process().name + self.type_name

        # image = read_input(self.path)
        image = Image.open(self.path)
        im_row_length, im_col_length = image.size
        num_window_row = 9
        num_window_col = 3
        win_row_length = int(im_row_length/num_window_row)
        win_col_length = int(im_col_length/num_window_col)
        
        # total_windows = num_window_row*num_window_col

        # i_windows = 0  # track how many windows have been processed
        i_cells = 0 # track how many cells have been collected for the current row. Currently counting complete cells and partial cells of type 8.
        
        for win_row_index in range(0,num_window_row):
            for win_col_index in range(0,num_window_col):
                # current_window = layout[win_row_index*win_row_length: (win_row_index+1)*win_row_length
                #                             ,win_col_index*win_col_length: (win_col_index+1)*win_col_length]
                start_row = win_row_index * win_row_length
                end_row = min((win_row_index + 1) * win_row_length, im_row_length)
                start_col = win_col_index * win_col_length
                end_col = min((win_col_index + 1) * win_col_length, im_col_length)
        
                # current_window = image[start_row:end_row, start_col:end_col]
                current_window = image.crop((start_row, start_col, end_row, end_col))
                # current_window = np.array(current_window)[:,:,0] 
                # plt.imshow(current_window)
                # plt.show()
                # print(current_window)
                img_bin = my_preprocessing( np.array(current_window)[:,:,0]  )
                plt.imshow(img_bin, cmap = 'gray')
                plt.show()
                # print(img_bin)
                # perform prepcossing
                # layout, layout_denoised, layout_bin, th = pre_processing(layout_path)
                blobs_layout = connected_components(img_bin)
                window_coordinates = (win_row_index, win_col_index)
                window_partial_cells = Window_partial_cells( window_coordinates )

                for blob in blobs_layout:
                    blob_img = blob.image
                    (min_row, min_col, max_row, max_col) = blob.bbox
                    
                    pkt = shape_packet()
                    
                    pkt.bbox = (min_row+start_row, min_col+start_col, max_row+start_row, max_col+start_col) # we preserve coordinates 
                    pkt.blob_img = blob_img
                    # store centroid of shape coordinate in coordinate of original image
                    pkt.centroid = ( int( blob.centroid[0]+start_row), int(blob.centroid[1]+start_col)) # centroid from skii_image is wrt window image coordinates

                    
                    if ( max_col == win_col_length ) :
                        pkt.packet_type = 4
                        window_partial_cells.type_4_pkts.append(pkt)
                        i_cells = i_cells + 1
                        # # print("found type",pkt.packet_type)
                    elif ( min_col == 0 ) :
                        pkt.packet_type = 8
                        window_partial_cells.type_8_pkts.append(pkt)
                        
                    else:
                        # shape_pkts.append(pkt)
                        self.cell_queue.put(pkt)
                        i_cells = i_cells + 1
                
                self.partial_cell_queue.put(window_partial_cells)
            
            # send the number of cells
            self.extractor_conn.send(i_cells)
            i_cells = 0 # reset the cell counter to zero for the next row of the windows
            self.row_done_event.set() # one row of the windows have been processed

        self.extractor_done_event.set() # signal end of processing all the windows
        
        # print(f"{self.name} {self.pid}: all packets sent. Now exiting")


class Cell_collector(multiprocessing.Process):
    def __init__(self, type_name, cell_queue, sorted_cell_queue, extractor_done_event, merger_done_event, row_done_event, collector_conn, collector_done_event): 
        multiprocessing.Process.__init__(self)
        self.type_name = type_name
        self.cell_queue = cell_queue  
        self.sorted_cell_queue = sorted_cell_queue
        self.extractor_done_event = extractor_done_event
        self.merger_done_event = merger_done_event
        self.row_done_event = row_done_event
        self.collector_conn = collector_conn
        self.collector_done_event = collector_done_event
        self.sorted_cell_list = []
        self.collect_sort_thread = threading.Thread(target=self.collect_sort)
        self.collect_sort_thread_running = False
    
    def collect_sort(self):

        while True:
            if self.extractor_done_event.is_set() and self.merger_done_event.is_set():
                if not self.cell_queue.empty():
                    pkt = self.cell_queue.get()
                    bisect.insort(self.sorted_cell_list , pkt)
                    # print(f'{self.name} lenght of list {len(self.sorted_cell_list)}')
                else: 
                    self.collect_sort_thread_running = False
                    break
            else:
                if not self.cell_queue.empty():
                        pkt = self.cell_queue.get()
                        bisect.insort(self.sorted_cell_list , pkt)
                        # print(f'{self.name} lenght of list {len(self.sorted_cell_list)}')

        print(f'\n {self.name} Child thread exiting')    

    def run(self):
        
        self.name = multiprocessing.current_process().name + self.type_name
        self.collect_sort_thread.start()
        self.collect_sort_thread_running = True
        
        while True:

            if self.collect_sort_thread_running:
                if self.row_done_event.is_set():
                    self.row_done_event.clear()     
                    cell_count = self.collector_conn.recv()  # Number of cells in the current row
                    # print(f'{self.name} Number of cells in current row = {cell_count}')

                    while len(self.sorted_cell_list) < cell_count: # We wait untill the sorted list has cell_count number of cells
                        pass
                    
                    self.sorted_cell_queue.put(self.sorted_cell_list[:cell_count])
                    del self.sorted_cell_list[:cell_count]
                    
            else: # when the child thread has finished, send whatever cells are in the list to the queue and exit
                
                if len(self.sorted_cell_list) > 0:
                    self.sorted_cell_queue.put(self.sorted_cell_list)
                    self.sorted_cell_list = []
                
                self.collector_done_event.set()
                break

        print(f'\n {self.name} Parent thread exiting')

class Cell_merger(multiprocessing.Process):
    def __init__(self, type_name, partial_cell_queue, cell_queue, extractor_done_event, merger_done_event):
        multiprocessing.Process.__init__(self)
        self.type_name = type_name
        self.partial_cell_queue = partial_cell_queue  
        self.cell_queue = cell_queue  
        self.extractor_done_event = extractor_done_event
        self.merger_done_event = merger_done_event
    
    def run(self):
        self.name = multiprocessing.current_process().name + self.type_name

        while True:

            if self.extractor_done_event.is_set(): 
                if self.partial_cell_queue.empty(): # if queue is empty, signal finish event and terminate the process, else continue
                    # print(f"{self.name} partial cell queue is empty, merger terminating")
                    self.merger_done_event.set()
                    break
                else:
                    # print(f"{self.name} Extractor terminated, ,merger finishing soon")
                    window_partial_cells = self.partial_cell_queue.get()
                    window_coordinates = window_partial_cells.window_coordinates
                    
                    if window_coordinates[1] == 0:    # Check if it is start of a row 
                        window_x = window_partial_cells
                    else:
                        window_x_1 = window_partial_cells
                        self.merge_cells(window_x, window_x_1)
                        window_x = window_x_1
            else:
                if not self.partial_cell_queue.empty(): 
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
                    # print(f'{self.name} merged and sent 1 packet')
                    
                    # shape_pkts.append(merged_pkt)
                    
                    # plt.imshow(merged_pkt.blob_img)
                    # plt.show()
                    
                    p2.type_8_pkts.remove(type_8_pkt)
                    
                    break

class Cell_validation(multiprocessing.Process):

    def __init__(self, sorted_cell_queue_layout, sorted_cell_queue_SEM, collector_done_event_layout, collector_done_event_SEM):
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
                                                                                       self.collector_done_event_layout, 'layout' ) )
        self.rcv_cells_SEM_thread = threading.Thread(target=self.rcv_cells, args =(self.sorted_cell_queue_SEM, self.cell_list_SEM, 
                                                                                   self.collector_done_event_SEM, 'SEM' ))
        self.cell_size = 10
    
    def run(self):
        self.name = multiprocessing.current_process().name
        self.rcv_cells_layout_thread.start()
        self.rcv_cells_layout_thread_running = True
        self.rcv_cells_SEM_thread.start()
        self.rcv_cells_SEM_thread_running = True

        while True:
            # time.sleep(.1)

            # print(f"{self.name}: layout list length= {len(self.cell_list_layout)}")
            # print(f"{self.name}: SEM list length= {len(self.cell_list_SEM)}")

            if not self.rcv_cells_layout_thread_running and not self.rcv_cells_SEM_thread_running:
                
                if len(self.cell_list_layout) > 0 and len(self.cell_list_SEM) > 0: 
                    self.validate( self.cell_list_layout, self.cell_list_SEM)
                    self.cell_list_layout = []
                    self.cell_list_SEM = []

                print(f'{self.name} is now exiting')
                break
            else:
                if len(self.cell_list_layout) > self.cell_size and len(self.cell_list_SEM) > self.cell_size: 
                    self.validate( self.cell_list_layout[:self.cell_size], self.cell_list_SEM[:self.cell_size])
                    del self.cell_list_layout[:self.cell_size]
                    del self.cell_list_SEM[:self.cell_size]


    def validate(self, cell_list_layout, cell_list_SEM ):

        for pkt_layout, pkt_SEM in zip(cell_list_layout, cell_list_SEM):
            # contour = detect_contour(filled_image)
            img = pkt_layout.blob_img.astype(np.uint8) * 255 
            contour, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gx,gy = sample_polygon_uniformly(contour[0],50)
            contour_sampled = np.column_stack((gx, gy)).astype(np.int32)
            f_descriptors_layout = fourier_descriptors(contour_sampled)

            img = pkt_SEM.blob_img.astype(np.uint8) * 255 
            contour, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gx,gy = sample_polygon_uniformly(contour[0],50)
            contour_sampled = np.column_stack((gx, gy)).astype(np.int32)
            f_descriptors_SEM = fourier_descriptors(contour_sampled)

            f_similarity = calculate_similarity(f_descriptors_layout, f_descriptors_SEM)
            
            # print(f'{self.name} similarity value = {f_similarity}')

    def rcv_cells(self, cell_queue, cell_list, collector_done_event, type_name):
        
        while True:
            if collector_done_event.is_set():
                if not cell_queue.empty():
                    cell_list.extend(cell_queue.get())                  
                else:
                    if type_name == 'layout':
                        self.rcv_cells_layout_thread_running = False
                    elif type_name == 'SEM':
                        self.rcv_cells_SEM_thread_running = False
                    break
            else:
                if not cell_queue.empty():
                    cell_list.extend(cell_queue.get()) 