import cv2
import os
import copy
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


from sklearn.mixture import GaussianMixture as GMM


def pre_processing(path):
    img = read_input(path)
    img_denoised = denoised(img)
    img_bin, th = binarized(img_denoised)
    return img, img_denoised, img_bin, th

def denoised(img):
    denoised_img = cv2.fastNlMeansDenoising(img,None,30,7,30)
#     cv2.imwrite(os.path.join(output_path, 'denoised.jpg'),denoised_img)
    return denoised_img

def binarized(img, threshold = -1):
    th, img_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     print(' OT threshold: ',th)
    #print('threshold using: ',th+10)
    th, img_th = cv2.threshold(img, th+11, 255 , cv2.THRESH_BINARY)
    #cv2.imwrite(os.path.join(output_path,'binarized.jpg'),img_th)
    #print('threshold using: ',th)
    return img_th, th

def read_input(path):
    img = cv2.imread(path)
    img = img[:,:,0]
    return img

def detect_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

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


    
def simple_signal(np_points):
    curve = np.zeros(shape=(2, np_points))
    t = np.linspace(-4, 4, np_points)

    curve[0, :] = 5 * np.cos(t) - np.cos(6 * t)
    curve[1, :] = 15 * np.sin(t) - np.sin(6 * t)

    return curve
    
def calculate_similarity(descriptors_1, descriptors_2):
    diff = np.abs(descriptors_1 - descriptors_2)
    similarity = np.mean(diff)
    return similarity
