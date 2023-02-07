# CSS844-frontiers-proj1-img-processing
Group members: Farnaz and Julia

Image files: Cross sectional images of maize roots (no scale bar)\
Treatment groups: 6, combinations of the following treatments:
1. Nitrogen deficient and normal Nitrogen
2. Biostimulant 1, Biostimulant 2, and control

Python file descriptions:

1_image_segmentation.py \
-- use opencv to perform colour range thresholding on a folder containing images. BGR colour ranges found using eyedropped in ms paint. Outputs segmented images to output folder \
-- 1) Background removed \
-- 2) Top left 25% of image underwent thresholding to remove the yellow name tag \
-- reference: https://stackoverflow.com/questions/72062001/remove-everything-of-a-specific-color-with-a-color-variation-tolerance-from-an 

2_plantcv.py \
-- use plantcv to obtain area (in absolute number of pixels) and root angles (in degrees) from folder containing segmented images. Outputs binarized images to a folder, skeletonized images to another folder, and resulting tables.csv file to a third folder. The `tables.csv` file contains 3 columns: 1. image filename, 2. root area, and 3. root angle
-- 1) binarize segmented image \
-- 2) blur small objects and fill in holes (improves skeletonize) \
-- 3) crop mask to focus on lateral root growth \
-- 4) skeletonize \
-- 5) identify branch and tip points \
-- 6) calculate root angle \
-- 7) calculate area of uncropped, binarized image from step 1 \
-- reference: https://github.com/nrt-impacts/CSS844-module3-Hollender/blob/main/2021-4-16_2D_Branch_Angle_Estimation_JK.ipynb 
