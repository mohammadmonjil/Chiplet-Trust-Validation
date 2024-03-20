import cv2
import numpy as np
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture as GMM
# import parameters as params


from components import *
from skimage import measure
import matplotlib.pyplot as plt

from fourier import *
from utility import *

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
    print(' OT threshold: ',th)
    #print('threshold using: ',th+10)
    th, img_th = cv2.threshold(img, th+10, 255 , cv2.THRESH_BINARY)
    #cv2.imwrite(os.path.join(output_path,'binarized.jpg'),img_th)
    #print('threshold using: ',th)
    return img_th, th

def pre_processing(path):
    img = read_input(path)
    img_denoised = denoised(img)
    img_bin, th = binarized(img_denoised)
    return img, img_denoised, img_bin, th

layout_path = "1.png"  # Replace with your image path
layout, layout_denoised, layout_bin, th = pre_processing(layout_path)
blobs_layout = connected_components(layout_bin)


for blob in blobs_layout:
    img = blob.image
    plt.imshow(img)
    plt.show()
