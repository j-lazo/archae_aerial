import general_functions as gf
import generate_map as gm
import os
import cv2


def main():

    real = 'real_values/Real_values_test.csv'
    y_2test, names_test = gf.load_labels(real)
    directory = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/Hjortahammar/All/'
    list_of_images = os.listdir(directory)
    whole_picture = gm.build_map(list_of_images, directory, 15, 18, [y_2test, names_test])
    cv2.imwrite('Hjortahammar_reals.jpg', whole_picture)

    predicted = 'results/Inception_predictions_20181213-18h12m_l2_0.01.csv'
    y_2test, names_test = gf.load_predictions(predicted)
    directory = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/Hjortahammar/All/'
    list_of_images = os.listdir(directory)
    whole_picture = gm.build_map(list_of_images, directory, 15, 18, [y_2test, names_test])
    cv2.imwrite('Hjortahammar_predicted.jpg', whole_picture)


if __name__ == '__main__':
    main()
