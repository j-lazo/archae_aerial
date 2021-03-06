import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from general_functions import paint_image


def build_map(list_of_images, directory, length, height, list_of_values=[]):
    index_list = [image[7:] for image in list_of_images]
    for k in range(height):
        base = length
        for j in range(length):
            for i, image in enumerate(index_list):
                if int(image[:-4]) == j+(base*k):
                    #a = cv2.imread(directory + list_of_images[i])
                    if list_of_values == []:
                        value = 1
                    else:
                        for index, name_result in enumerate(list_of_values[1]):
                            if name_result == list_of_images[i]:
                                value = float(list_of_values[0][index])
                                #if value != 1:
                                #    print(list_of_values[1][index], value)
                                break
                            else:
                                value = 1

                    a = paint_image(directory + list_of_images[i], value)
                    if j > 0:
                        new_im = np.concatenate((new_im, a), axis=1)
                    else:
                        new_im = a

        shape_new_image = np.shape(new_im)
        if shape_new_image[1] != 200*length:
            add_zeros = np.zeros([shape_new_image[0], 200*length - shape_new_image[1], 3])
            new_im = np.concatenate((new_im, add_zeros), axis=1)

        if k > 0:
            whole_image = np.concatenate((new_im, whole_image), axis=-0)

        else:
            whole_image = new_im

        #plt.figure()
        #plt.imshow(whole_image)
        #plt.show()
    return whole_image


"""def main():
    directory = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/Island/Ir_images/'
    list_of_images = os.listdir(directory)
    whole_picture = build_map(list_of_images, directory, 47, 58)
    cv2.imwrite('Birka_RGB.jpg', whole_picture)


if __name__ == '__main__':
    main()
"""