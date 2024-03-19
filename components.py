from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture as GMM

from utility import *
import parameters as params


import numpy as np
import cv2
import copy


def connected_components(img):
    lbl = label(img)
    blobs = regionprops(lbl)
    return blobs



def regions(blobs,image, partial_check=True):
    _,width = image.shape
    bboxes = []
    index = -1 
    top_corner = {}
    max_height = -1
    min_area = 100000
    omitted_bbox = []
    for _,blob in enumerate(blobs):
        #print('bbox: ',blob.bbox)
        bbox_yx = blob.bbox     #y1,x1,y2,x2
        bbox = (bbox_yx[1],bbox_yx[0],bbox_yx[3],bbox_yx[2]) #x1,y1,x2,y2

        if partial_check:
            if bbox[0] == 0 or bbox[2] >= width-1:      #omitting partially present bbox
                omitted_bbox.append(bbox)
                continue 
            area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            #print(area)
            if area < min_area:
                min_area = area
            if min_area <= 10:
                omitted_bbox.append(bbox)
                continue
        index+=1
        top_corner[index] = [bbox[0],bbox[1]]
        if (bbox[3]-bbox[1]) > max_height:
            max_height = bbox[3]-bbox[1]
        bboxes.append(bbox)
        
    return bboxes, top_corner, max_height, min_area, omitted_bbox



#detect the keypoints
def compare(post_bbox, y2_list):
    y1 = post_bbox[1]
    count = 0
    for y2 in y2_list:
        if y1>=y2:    
            count += 1
        else:
            break
    return count



def bbox_sorting(img, bboxes, output_path, th=0.5):     #params --> [[x1,y1,x2,y2]]
    image =copy.deepcopy(img)
    _,width = img.shape
    print('img shape: ',img.shape)
    lines = []
    line_y1 = 0
    max_y2 = -10
    count = -1

    y2_list = []
    bbox_traverse = 0

    bboxes_sorted = []
    r_bboxes = []
    lines.append([0,0,width-1,0])
    #r_bboxes_local = []

    for i in range(0,len(bboxes)-1):
        bbox = bboxes[i]
        post_bbox = bboxes[i+1]
        max_y2 = max(max_y2, bbox[3])
        #bbox_local_coord = [bbox[0], bbox[1]-line_y1, bbox[2], bbox[3]-line_y1]
        #r_bboxes_local.append([int(x) for x in bbox_local_coord])                                         #,x1,y1-d,x2,y2-d
        r_bboxes.append([int(x) for x in bbox])
        y2_list.append(bbox[3])

        bbox_traverse += 1
        if bbox_traverse < 10:                      #inserting 1st 10 dopant regions before 
            continue                                #proceeding to next stage
        #print("check")  
        y2_list.sort() #small to large
        smaller_count = compare(post_bbox, y2_list) #whether next bbox y2 crossed the height of all other r_bboxes
                                                    # if it crosses, how many it crosses
        #print('smaller count: ',smaller_count)

        if smaller_count/len(y2_list) >= th:        #We are assuming, if next bbox crosses 50% of previous
                                                    #boxes, then it is from next row

            line_y1 = max_y2                        # we draw the line of the row from the max height element
                                                    # of previous row

            line = [0,line_y1, width-1, line_y1]    #row separator line
            
            lines.append(line)                 
            ##initialize
            max_y2 = -10
            y2_list.clear()
            bbox_traverse = 0
            ####
            r_bboxes.sort(key=lambda x: int(x[0]))
            bboxes_sorted.append(copy.deepcopy(r_bboxes))
            r_bboxes = []

            image = draw_rectangle(bbox, image,thickness=2)
            image = draw_rectangle(post_bbox, image,color = (255,255,255),thickness=2)
            count+=1
    
    if len(r_bboxes)>0:
        r_bboxes.append([int(x) for x in post_bbox])
        r_bboxes.sort(key=lambda x: int(x[0]))
        bboxes_sorted.append(copy.deepcopy(r_bboxes))

    print('#row count: ',count)
    return lines, image, bboxes_sorted


def components_merging(row_images, row_bboxes_local, output_path):

    output_row_image_path = os.path.join(output_path,"rows")
    if not os.path.exists(output_row_image_path):
        os.makedirs(output_row_image_path)

    updated_list = []
    distances = []
    check = np.zeros(len(row_bboxes_local),dtype=bool)

    for index, (row_image, bboxes) in enumerate(zip(row_images, row_bboxes_local)):
        if index%2==0:
            bboxes_next_row = row_bboxes_local[index+1]
            if len(bboxes) >= len(bboxes_next_row):
                check[index] = True
            else:
                check[index+1] = True

        distance = calculate_distance(bboxes)
        distances.extend(distance)
        threshold = th_selection(distance)
        print('row: ',index,' distances: ',distance, 'threshold: ', threshold)
        updated_bboxes = composite_component_formation(bboxes, distance, threshold=threshold)
        row_image = draw_bboxes(row_image,updated_bboxes)
        cv2.imwrite(os.path.join(output_row_image_path, 'row_'+str(index)+'_merged_bbox.png'),row_image)
        splated_bboxes = splating_bbox(updated_bboxes,row_image)

        if index%2==1:  #every 2nd one
            if check[index]:
                print('row ', index, ' selected')
                updated_list[len(updated_list)-1] = splated_bboxes
            else:
                print('row ', index-1, ' selected')
        else:
            updated_list.append(splated_bboxes)
    #print(check)
    return updated_list



def composite_component_formation(components, distance, threshold=params.merging_threshold):
    comp_list = []
    merge = []
    next = True
    #print('used th: ',threshold)
    for i,d in enumerate(distance):
        if d < threshold:
            merge.extend([components[k] for k in range(i,i+2)])
            next = False
        else:
            if len(merge) > 0:
                merged_comp = merging(merge)
                comp_list.append(merged_comp)
                merge = []
            if next:
                comp_list.append(components[i])
            else:
                next = True
        #print("step ",i," comp: ",comp_list, " merge: ",merge)
    if len(merge) > 0:
        merged_comp = merging(merge)
        comp_list.append(merged_comp)
    if distance[len(distance)-1] >= threshold:
        comp_list.append(components[len(components)-1])
    return comp_list



def cell_generation(cell_rows, merged_cells, output_path):
    for r_n, (row_image, cells) in enumerate(zip(cell_rows,merged_cells)):
        height, width = row_image.shape
        for c_n, comp in enumerate(cells):
            x1, y1, x2, y2 = cropped_coordinates(height, width, comp)
            cell_image = row_image[y1:y2+1, x1:x2+1]
            save_cell_image(cell_image, r_n, c_n, output_path) 