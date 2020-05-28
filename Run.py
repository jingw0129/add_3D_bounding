"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""


from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo
import pickle
import os
import time
import math
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse
import cv2
# ######################################################
# model_pkl = []
# try:
#     with open(os.path.dirname(__file__) + 'weights/back_up/epoch_10.pkl','rb') as f:
#
#         # loop indefinitely
#         model_pkl.append(pickle.load(f))  # add each item from the file to a list
# except EOFError:                             # the exception is used to break the loop
#     pass
# print(model_pkl)
# ##############################################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="Kitti/testing/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):
        # the math! returns X, the corners used for constraint
        location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)
        # print('location, X ',location, X)
        orient = alpha + theta_ray

        if img_2d is not None:
            plot_2d_box(img_2d, box_2d)

        plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes
        vertices = plot_3d_box(img, cam_to_img, orient, dimensions, location)
        # print('vertices: ',vertices)
        return location, vertices

def main():

    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]   
    
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        
        model_data = checkpoint['model_state_dict']
        # print(model_data.keys())
        # with open ("model_state_dic.txt",'a') as f:
        # with open ("model_state_dic.txt",'w') as f:
        #     f.write(str(model_data.keys()))



        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        # print(model)
    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)
    print(angle_bins)
    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir


    if FLAGS.video:
        if FLAGS.image_dir == "Kitti/testing/image_2/" and FLAGS.cal_dir == "camera_cal/":
            image_dir = "Kitti/testing/image_2/"
            cal_dir = "camera_cal/"


    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir

    calib_file = calib_path + "calib_cam_to_cam.txt"



    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for img_id in ids:

        start_time = time.time()

        img_file = img_path + img_id + ".png"


        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)

        detections = yolo.detect(yolo_img)

        for detection in detections:
            # print(detection.detected_class)

            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
                # detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix

            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)

            # print('orient, conf, dim', orient, conf, dim)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            # print(conf, argmax)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]

            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            if math.isnan(dim[0]) is False:
                if FLAGS.show_yolo:
                    location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)[0]
                else:
                    location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)[0]

            # if not FLAGS.hide_debug:
            #     if not FLAGS.hide_debug:
            #         print('class' + str(detection.detected_class), 'Estimated pose: %s' % location,
            #               '2D' + (str(box_2d[0][0]) + ' ' + str(box_2d[0][1]) + ' ' + str(box_2d[1][0]) + ' ' + str(
            #                   box_2d[1][1])),
            #               'alpha:' + str(alpha),
            #
            #               'dim' + str(dim),
            #               'ray' + str(theta_ray),
            #
            #               'proj_matrix' + str(proj_matrix),
            #               'img' + str(img.shape),
            #               'input_tensor' +'input_img'+ str(input_tensor.shape)+str(input_tensor)
            #               )
                # with open('./camera_para/image_label/'+str(file_name)+'/'+str(img_id)+'.txt','w') as file:

                # with open('./Kitti/testing/testing_result/'  + str(img_id) + '.txt', 'a') as file:
                #     file.write(str(detection.detected_class) + ' ')
                #     file.write(str(0.00)+' ')
                #     file.write(str(0)+' ')
                #     file.write(str(alpha)+ ' ')
                #     file.write(str(box_2d[0][0]) + ' ' + str(box_2d[0][1]) + ' ' + str(box_2d[1][0]) + ' ' + str(box_2d[1][1])+' ')
                #     file.write(str(dim[0])+' '+str(dim[1])+ ' '+ str(dim[2])+ ' ')
                #     file.write(str(location[0])+' '+str(location[1])+' '+str(location[2])+' ')
                #     file.write(str(theta_ray))
                #     file.write('\n')
        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        else:
            cv2.imshow('3D detections', img)

        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')

        if FLAGS.video:
            cv2.waitKey(1)
        else:
            if cv2.waitKey(0) != 32: # space bar
                exit()

if __name__ == '__main__':
    main()

