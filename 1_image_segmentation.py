import glob
import math 
import numpy as np
import cv2
from imageio import imread, imsave
import os
import os.path

# https://stackoverflow.com/questions/72062001/remove-everything-of-a-specific-color-with-a-color-variation-tolerance-from-an

def colour_thresh_segmentation(infile, outfile):
 #NOTE we switched from RGB to BGR by using cv2, thus the colours below are BGR
 image = cv2.imread(infile)

 ### removing background ###
 #remove green background
 u_green = np.array([60, 75, 50],np.uint8)
 l_green = np.array([20, 40, 20],np.uint8)
 mask1 = cv2.inRange(image, l_green, u_green)

 old_high = np.array([10,20,20],np.uint8)
 old_low = np.array([0, 10, 0],np.uint8)
 mask2 = cv2.inRange(image, old_low, old_high)

 old_high = np.array([63,95,68],np.uint8)
 old_low = np.array([48, 75, 45],np.uint8)
 mask3 = cv2.inRange(image, old_low, old_high)

 old_high = np.array([75,120,75],np.uint8)
 old_low = np.array([68, 110, 65],np.uint8)
 mask = cv2.inRange(image, old_low, old_high)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (400,400))
 mask4 = cv2.dilate(mask, kernel, iterations=1)

 old_high = np.array([25,40,25],np.uint8)
 old_low = np.array([16, 32, 14],np.uint8)
 mask = cv2.inRange(image, old_low, old_high)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
 mask5 = cv2.dilate(mask, kernel, iterations=1)

 #combine all masks to fully remove background
 combined_mask = mask1 | mask2 | mask3 |mask4 | mask5
 image[combined_mask>0] = [0,0,0]

 ### remove yellow tag only in the top left corner 25% of image ###
 h,w,channel = image.shape
 subsect = image[0:math.floor(h/2),0:math.floor(w/2),:]

 old_high = np.array([90, 190, 215],np.uint8)
 old_low = np.array([60, 170, 180],np.uint8)
 mask = cv2.inRange(subsect, old_low, old_high)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100))
 mask1 = cv2.dilate(mask, kernel, iterations=1)

 old_high = np.array([50, 140, 165],np.uint8)
 old_low = np.array([45, 130, 155],np.uint8)
 mask = cv2.inRange(subsect, old_low, old_high)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100)) # remove within 100pixel region
 mask3 = cv2.dilate(mask, kernel, iterations=1)

 old_high = np.array([85, 210, 240],np.uint8)
 old_low = np.array([70, 195, 220],np.uint8)
 mask = cv2.inRange(subsect, old_low, old_high)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100)) # remove within 100pixel region
 mask4 = cv2.dilate(mask, kernel, iterations=1)

 # grey text on yellow tag
 old_high = np.array([70, 70, 70],np.uint8)
 old_low = np.array([60, 60, 50],np.uint8)
 mask = cv2.inRange(subsect, old_low, old_high)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)) # remove within 100pixel region
 mask5 = cv2.dilate(mask, kernel, iterations=1)

 # white part of yellow tag
 old_high = np.array([250, 250, 250],np.uint8)
 old_low = np.array([210, 210, 210],np.uint8)
 mask = cv2.inRange(subsect, old_low, old_high)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100)) # remove within 100pixel region
 mask6 = cv2.dilate(mask, kernel, iterations=1)

 old_high = np.array([250, 250, 250],np.uint8)
 old_low = np.array([177, 177, 176],np.uint8)
 mask = cv2.inRange(subsect, old_low, old_high)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100)) # remove within 100pixel region
 mask7 = cv2.dilate(mask, kernel, iterations=1)

 #combine all masks to clean out the yellow tag marker in the top left section of the image (25% of image)
 combined_mask = mask1 | mask3 | mask4 | mask5 | mask6 | mask7
 subsect[combined_mask>0] = [0,0,0]
 image[0:math.floor(h/2),0:math.floor(w/2),:] = subsect

 #write out segmented image
 cv2.imwrite(outfile, image)
 return

if __name__ == "__main__":

 indir = '/mnt/home/zhengjul/class/frontiers/images/'
 outdir='/mnt/home/zhengjul/class/frontiers/work/1_segmentation/'
 for infile in glob.glob(indir+"*"):
  outfile = outdir + os.path.basename(infile)
  if not os.path.isfile(outfile):
    print(outfile) 
    colour_thresh_segmentation(infile, outfile)

