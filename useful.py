import cv2
from skimage.measure import label, regionprops
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