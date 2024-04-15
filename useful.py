import cv2
from skimage.measure import label, regionprops
import numpy as np
import psutil
import csv
import time

def connected_components(img):
    lbl = label(img)
    blobs = regionprops(lbl)
    return blobs
def my_preprocessing(img):
    img_denoised = denoised(img)
    img_bin, th = binarized(img_denoised)
    return img_bin

def read_input(path):
    img = cv2.imread(path)
    img = img[:,:,0]
    print('img shape: ',img.shape)
    return img

def denoised(img):
    denoised_img = cv2.fastNlMeansDenoising(img,None,30,7,30)
#     cv2.imwrite(os.path.join(output_path, 'denoised.jpg'),denoised_img)
    return denoised_img

def binarized(img, threshold = -1):
    th, img_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print(' OT threshold: ',th)
    #print('threshold using: ',th+10)
    th, img_th = cv2.threshold(img, th+10, 255 , cv2.THRESH_BINARY)
    #cv2.imwrite(os.path.join(output_path,'binarized.jpg'),img_th)
    #print('threshold using: ',th)
    return img_th, th
    
def sample_polygon_uniformly(contour, M):
    x = contour[:,0,0]
    y = contour[:,0,1]
    L = path_length(x,y)
    N = len(x)
    delta = L/M
#     print(len(y))

    gx = np.zeros(M)
    gy = np.zeros(M)
    gx[0] = x[0]
    gy[0] = y[0]
    
    i = 0
    k = 1
    alpha = 0
    beta = delta
    
    while (i<N)and(k<M) :
        vax = x[i]
        vay = y[i]
        vbx = x[i+1]
        vby = y[i+1]
        
        lenth_seg = abs( pow((pow((vax-vbx),2) + pow((vay-vby),2)), .5))
        
        while (beta<=(alpha+lenth_seg))and(k<M):
            gx[k] = vax + (beta-alpha)/lenth_seg*(vbx-vax)
            gy[k] = vay + (beta-alpha)/lenth_seg*(vby-vay)
            k = k + 1
            beta = beta + delta
        alpha = alpha + lenth_seg
        i = i+1
        
    return gx, gy

def path_length(x,y):
    N = len(x)
    L = 0
    for i in range(0,N-1):
        L = L + abs( pow((pow((x[i+1]-x[i]),2) + pow((y[i+1]-y[i]),2)), .5))
    return L
    
# def calculate_similarity(descriptors_1, descriptors_2):
#     diff = np.abs(descriptors_1 - descriptors_2)
#     similarity = np.mean(diff)
#     return similarity

# def log_resource_usage(pid_dict, program_done_event):
#     log_cpu = open('log_cpu.txt', mode='w')
#     # log_mem = open('log_mem.csv', mode='w', newline='')
    
#     all_ps_process = []
#     process_names = []

#     for process_name, pid in pid_dict.items():
#         all_ps_process.append( psutil.Process(pid) )
#         process_names.append(process_name)

#     # cpu_writer = csv.DictWriter(log_cpu, fieldnames= process_names)
#     # mem_writer = csv.DictWriter(log_mem, fieldnames= process_names)
    
#     # cpu_writer.writeheader()
#     # mem_writer.writeheader()

#     # cpu_dict = {}
#     # memory_dict = {}
    
#     while not program_done_event.is_set():
#         for process_name, ps_process in zip(process_names, all_ps_process):
#             if psutil.pid_exists( pid_dict[process_name]):
#                 cpu_percent_per_core = ps_process.cpu_percent(interval = None, percpu = True)
#                 string = f'{process_name}: \n'
#                 for i, cpu_percent in enumerate(cpu_percent_per_core):
#                     string += f'core {i} = {cpu_percent}'
#                 # memory_dict[process_name] = ps_process.memory_info().rss / (1024 * 1024)
#             else:
#                 pass

#         # cpu_writer.writerow(cpu_dict)
#         # mem_writer.writerow(memory_dict)
#         time.sleep(0.1)

#     log_cpu.close()
    # log_mem.close()

def log_resource_usage(pid_dict, program_done_event):
    log_cpu = open('log_cpu.csv', mode='w', newline='')
    log_mem = open('log_mem.csv', mode='w', newline='')
    
    all_ps_process = []
    process_names = []

    for process_name, pid in pid_dict.items():
        all_ps_process.append( psutil.Process(pid) )
        process_names.append(process_name)

    cpu_writer = csv.DictWriter(log_cpu, fieldnames= process_names)
    mem_writer = csv.DictWriter(log_mem, fieldnames= process_names)
    
    cpu_writer.writeheader()
    mem_writer.writeheader()

    cpu_dict = {}
    memory_dict = {}
    
    while not program_done_event.is_set():
        for process_name, ps_process in zip(process_names, all_ps_process):
            if psutil.pid_exists( pid_dict[process_name]):
                cpu_dict[process_name] = ps_process.cpu_percent(interval = None)
                memory_dict[process_name] = ps_process.memory_info().rss / (1024 * 1024)
            else:
                cpu_dict[process_name] = 0
                memory_dict[process_name] = 0
                
        cpu_writer.writerow(cpu_dict)
        mem_writer.writerow(memory_dict)
        time.sleep(0.1)

    log_cpu.close()
    log_mem.close()