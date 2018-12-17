import cv2
import os
import random
import numpy as np
import shutil
from matplotlib import pyplot as plt

print(os.getcwd())
files_path = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/aerial_photos_plus'
f = os.listdir(files_path)


def adjust_brightness(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)


for element in f[:]:
    img = cv2.imread("".join([files_path, '/', element]))

    rows, cols, channels = img.shape
    # get the rotation matrixes
    rot1 = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1)
    rot2 = cv2.getRotationMatrix2D((cols/2,rows/2), 180, 1)
    rot3 = cv2.getRotationMatrix2D((cols/2,rows/2), 270, 1)
    # rotate the images
    im_rot1 = cv2.warpAffine(img, rot1, (cols, rows))
    im_rot2 = cv2.warpAffine(img, rot2, (cols, rows))
    im_rot3 = cv2.warpAffine(img, rot3, (cols, rows))
    # flip images 
    horizontal_img = cv2.flip(img, 0 )
    vertical_img = cv2.flip(img, 1 )
   
    # save the images 
    cv2.imwrite("".join([files_path, '/', element[:-4], '1', '.jpg']), im_rot1)
    cv2.imwrite("".join([files_path, '/', element[:-4], '_2', '.jpg']), im_rot2)
    cv2.imwrite("".join([files_path, '/', element[:-4], '_3', '.jpg']), im_rot3)
    cv2.imwrite("".join([files_path, '/', element[:-4], '_4', '.jpg']), horizontal_img)
    cv2.imwrite("".join([files_path, '/', element[:-4], '_5', '.jpg']), vertical_img)

    # change brightness
    list_of_images = [img, im_rot1, im_rot2, im_rot3, horizontal_img, vertical_img]
    gammas = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
    for i in range(4):
        choice = random.choice(list_of_images)
        image_brg1 = adjust_brightness(choice, random.choice(gammas))
        cv2.imwrite("".join([files_path, '/', element[:-4], '_', str(i+6), '.jpg']), image_brg1)

    """
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.imshow(horizontal_img)
    plt.subplot(2,2,3)
    plt.imshow(vertical_img)
    plt.subplot(2,2,4)
    plt.imshow(img)

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.imshow(im_rot1)
    plt.subplot(2,2,3)
    plt.imshow(im_rot2)
    plt.subplot(2,2,4)
    plt.imshow(im_rot3)

    plt.show()"""

