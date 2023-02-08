from imageio import imread, imsave
import numpy as np
from plantcv import plantcv as pcv
import cv2
import os
import glob
import pandas as pd
import os.path

# source https://github.com/nrt-impacts/CSS844-module3-Hollender/blob/main/2021-4-16_2D_Branch_Angle_Estimation_JK.ipynb

def indexer(array):
    to_process = np.where(array == 255)
    ys = to_process[0]
    xs = to_process[1]
    coords = []
    for i in range(len(xs)):
        coord = [xs[i],ys[i]]
        coords.append(coord)
    return coords

## find base

def find_base(tips):
    y_coordinates = []
    for i in range(len(tips)):
        y_coordinates.append(tips[i][1])
    max_index = np.argmax(y_coordinates)
    to_return = tips[max_index]
    return to_return

## find branchpoint with smallest euclidian distance to base
import math
def find_branch_point(array,base):
    to_process = np.where(array == 255)
    ys = to_process[0]
    xs = to_process[1]
    coords = []
    for i in range(len(xs)):
        coord = [xs[i],ys[i]]
        coords.append(coord)
    coords_to_use = []
    for i in range(len(coords)):
        if coords[i] != base:
            coords_to_use.append(coords[i])
    distances = []
    for i in range(len(coords_to_use)):
        distance = math.sqrt((coords_to_use[i][0] - base[0])**2 + (coords_to_use[i][1] - base[1])**2)
        distances.append(distance)
    branch_point_index = np.argmin(distances)
    branch_point_coord = coords_to_use[branch_point_index]        
    return branch_point_coord

# I've got the relevant branch point + the relevant base. Now, find the two tips farthest from the branch point. 

def find_branch_tips(array,branch_point,base):
    to_process = np.where(array == 255)
    ys = to_process[0]
    xs = to_process[1]
    coords = []
    for i in range(len(xs)):
        coord = [xs[i],ys[i]]
        coords.append(coord)
    coords_to_use = []
    for i in range(len(coords)):
        if coords[i] != base:
            coords_to_use.append(coords[i])
    
    # calculate distances of remaining tips to the branch point
    distances = []
    for i in range(len(coords_to_use)):
        distance = math.sqrt((coords_to_use[i][0] - branch_point[0])**2 + (coords_to_use[i][1] - branch_point[1])**2)
        distances.append(distance)
    original_distances = distances
    distances.sort()
    first = distances[0]
    second = distances[1]
    top_index = original_distances.index(first)
    second_index = original_distances.index(second)
    
    branch_tips = [coords_to_use[top_index],coords_to_use[second_index]]
    return branch_tips

# now we've got the base, our branch point, and our two things to compare against. Now
# (1) branchpoint to branch tip 1, and branchpoint to branch tip 2 vectors
# (2) Figure out the angle between them

def branch_angle(branch_tips,branch_point):
    first_branch_tip = branch_tips[0]
    second_branch_tip = branch_tips[1]
    branch_vector_one = [first_branch_tip[0] - branch_point[0],first_branch_tip[1] - branch_point[1]]
    branch_vector_two = [second_branch_tip[0] - branch_point[0],second_branch_tip[1] - branch_point[1]]
    
    unit_vector_1 = branch_vector_one / np.linalg.norm(branch_vector_one)
    unit_vector_2 = branch_vector_two / np.linalg.norm(branch_vector_two)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle_in_radians = np.arccos(dot_product)
    angle_in_degrees = np.degrees(angle_in_radians)
    
    return angle_in_degrees

## Put everything together into one function

def calculate_branch_angle(tip_pts_mask,branch_pts_mask):
    indices = indexer(tip_pts_mask)
    base = find_base(indices)
    branch_point = find_branch_point(branch_pts_mask,base)
    branch_tips = find_branch_tips(tip_pts_mask,branch_point,base)
    degrees = branch_angle(branch_tips,branch_point)
    return degrees

if __name__ == "__main__":
 indir = '/mnt/home/zhengjul/class/frontiers/work/1_segmentation/'  #read in segmented images
 outdir_binary='/mnt/home/zhengjul/class/frontiers/work/2_binary_thresh/' #output binary threshold images
 outdir_plantcv='/mnt/home/zhengjul/class/frontiers/work/3_plantcv_img/' #output plantcv image results
 # 'img','area','angle'
 outfile_tables='/mnt/home/zhengjul/class/frontiers/work/4_tables/tables.csv' #output into one table the mask area and output "root angle" (suspicious, skeletonize is too detailed for this to be correct based on visual inspection)


 # analyze all figures
 with open(outfile_tables_backup, 'r') as f:
  imgs_done = f.read()

 for infile in glob.glob(indir+"*"):
  filename = os.path.basename(infile)
  outfile_binary = outdir_binary + filename
  outfile_plantcv = outdir_plantcv + filename
  #print(filename)

  if not filename in imgs_done: #if filename is not in finishd file
   print(filename)

   # binarize image into black or white
   image, path, filename = pcv.readimage(infile, mode="gray")
   binary_img = pcv.threshold.binary(gray_img=image, threshold=10, max_value=255, object_type='light')
   pcv.print_image(binary_img, outfile_binary)
   print("output binary img")

   # fill in any holes so that skeletonize won't capture too much details
   binary_img = pcv.median_blur(gray_img=binary_img, ksize=5) # blur
   mask = pcv.fill(bin_img=binary_img, size=500) # fill in holes

   # manually crop the mask to focus on the lateral root angle
   cropped_mask = mask[1300:3000, 1300:3000]

   #skeletonize
   skeleton = pcv.morphology.skeletonize(mask=cropped_mask)
   # Adjust line thickness with the global line thickness parameter
   pcv.params.line_thickness = 200 
   pcv.print_image(skeleton, outfile_plantcv)
   print("output skeleton img")

   # Prune the skeleton  
   pruned, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=0, mask=cropped_mask)

   # Identify branch points   
   branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=skeleton, mask=cropped_mask, label="default")

   # Identify tip points   
   tip_pts_mask = pcv.morphology.find_tips(skel_img=skeleton, mask=None, label="default")

   print("prior to calc angle")
   # add results to df as new row
   ret,thresh=cv2.threshold(binary_img,10,255,cv2.THRESH_BINARY_INV)
   area = cv2.countNonZero(thresh)    # mask area
   angle = calculate_branch_angle(tip_pts_mask,branch_pts_mask) # root angle
   print("after calc angle")

   print("save row data "+str(area)+" "+str(angle))
   with open(outfile_tables,"a+") as out:
    out.write(filename+","+str(area)+","+str(angle)+"\n")
