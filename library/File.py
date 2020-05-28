"""
Functions to read from files
TODO: move the functions that read label from Dataset into here
"""
import numpy as np


def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P0:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            # print('cam_to_img from File.py ',cam_to_img)
            return cam_to_img


    file_not_found(cab_f)

def get_P(cab_f):
    for line in open(cab_f):
        if 'P_rect_00' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            return_matrix = np.zeros((3,4))
            return_matrix = cam_P.reshape((3,4))
            # print('get_P_from File.py ',return_matrix)
            return return_matrix
    # print('get_calibration_cam_to_image_from File.py ',get_calibration_cam_to_image)
    # try other type of file
    return get_calibration_cam_to_image




def file_not_found(filename):
    print("\nError! Can't read calibration file, does %s exist?"%filename)
    exit()
