import os
import cv2

path_directory = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/case_1/'
list_images = os.listdir(path_directory)
list_images = [image for image in list_images if image[-4:] == '.TIF']

for image in list_images:
    im = cv2.imread(path_directory+image)
        #Image.open(path_directory+image)
    outfile = image[:-4] + '.jpg'
    cv2.imwrite(path_directory + outfile, im)
