import cv2

import numpy as np
import os
import random
import torch
from torchvision import transforms
from torch.utils import data
import sys
sys.path.insert(1,os.path.dirname(__file__)+'library/')
sys.path.insert(1,os.path.dirname(__file__)+'torch_lib/')
from library.File import *
from torch_lib.ClassAverages import ClassAverages

# TODO: clean up where this is
def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins

class Dataset(data.Dataset):
    def __init__(self, path, bins=2, overlap=0.1):

        self.top_label_path = path + "/label_2/"
        self.top_img_path = path + "/image_2/"
        self.top_calib_path = path + "/calib/"
        # use a relative path instead?

        # TODO: which camera cal to use, per frame or global one?
        self.proj_matrix = get_calibration_cam_to_image(os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + '/camera_cal/calib_cam_to_cam.txt'))

        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # name of file
        self.num_images = len(self.ids)

        # create angle bins
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins
        for i in range(1,bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2 # center of the bin

        self.overlap = overlap
        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0,bins):
            self.bin_ranges.append(( (i*self.interval - overlap) % (2*np.pi), \
                                (i*self.interval + self.interval + overlap) % (2*np.pi)) )

        # hold average dimensions
        class_list = ['car', 'Van', 'Truck', 'Pedestrian','Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)

        self.object_list = self.get_objects(self.ids)

        # pre-fetch all labels
        self.labels = {}
        last_id = ""
        i = 0
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            i = i+1
            # print(i, label)
            if id != last_id:
                self.labels[id] = {}
                last_id = id

            self.labels[id][str(line_num)] = label

        # hold one image at a time
        self.curr_id = ""
        self.curr_img = None


    # should return (Input, Label)
    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = cv2.imread(self.top_img_path + '%s.png'%id)

        label = self.labels[id][str(line_num)]
        # P doesn't matter here
        obj = DetectedObject(self.curr_img, label['Class'], label['Box_2D'], self.proj_matrix, label=label)

        return obj.img, label

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        objects = []
        for id in ids:
            with open(self.top_label_path + '%s.txt'%id) as file:
                # print(self.top_label_path + '%s.txt'%id)
                for line_num,line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "DontCare":
                        continue

                    dimension = np.array([float(line[8]), float(line[9]), float(line[10])], dtype=np.double)
                    self.averages.add_item(obj_class, dimension)

                    objects.append((id, line_num))


        self.averages.dump_to_file()
        return objects


    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt'%id).read().splitlines()
        label = self.format_label(lines[line_num])
        # print(self.top_label_path + '%s.txt'%id)
        return label

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(' ')

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])
            # print(line[i])
        Alpha = line[3] # what we will be regressing

        Ry = line[14]
        top_left = (int(round(float(line[4]))), int(round(float(line[5]))))
        bottom_right = (int(round(float(line[6]))), int(round(float(line[7]))))
        Box_2D = [top_left, bottom_right]

        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double) # height, width, length
        # modify for the average
        Dimension -= self.averages.get_item(Class)

        Location = [line[11], line[12], line[13]] # x, y, z
        Location[1] = float(Location[1])
        Location[1] -= int(Dimension[0]) / 2 # bring the KITTI center up to the middle of the object

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)

        # alpha is [-pi..pi], shift it to be [0..2pi]
        angle = float(Alpha) + np.pi

        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            Orientation[bin_idx,:] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1

        label = {
                'Class': Class,
                'Box_2D': Box_2D,
                'Dimensions': Dimension,
                'Alpha': Alpha,
                'Orientation': Orientation,
                'Confidence': Confidence
                }

        return label

    # will be deprc soon
    def parse_label(self, label_path):
        buf = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class == "DontCare":
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Alpha = line[3] # what we will be regressing
                Ry = line[14]
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]

                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

                buf.append({
                        'Class': Class,
                        'Box_2D': Box_2D,
                        'Dimensions': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry
                    })
        return buf

    # will be deprc soon
    def all_objects(self):
        data = {}
        for id in self.ids:
            data[id] = {}
            img_path = self.top_img_path + '%s.png'%id
            img = cv2.imread(img_path)
            data[id]['Image'] = img

            # using p per frame
            calib_path = self.top_calib_path + '%s.txt'%id

            proj_matrix = get_calibration_cam_to_image(calib_path)

            # using P_rect from global calib file
            proj_matrix = self.proj_matrix

            data[id]['Calib'] = proj_matrix

            label_path = self.top_label_path + '%s.txt'%id

            labels = self.parse_label(label_path)
            objects = []
            for label in labels:
                box_2d = label['Box_2D']
                detection_class = label['Class']
                objects.append(DetectedObject(img, detection_class, box_2d, proj_matrix, label=label))

            data[id]['Objects'] = objects

        return data


"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""
class DetectedObject:
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):

        if isinstance(proj_matrix, str): # filename
            proj_matrix = get_P(proj_matrix)
            # proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d,proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d,proj_matrix):
        width = img.shape[1]
        # fovx = 2 * np.arctan(width / (2 * 1011.2))
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):

        # Should this happen? or does normalize take care of it. YOLO doesnt like
        # img=img.astype(np.float) / 255

        # torch transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        process = transforms.Compose ([
            transforms.ToTensor(),
            normalize
        ])

        # crop image
        pt1 = box_2d[0]
        pt2 = box_2d[1]


        crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("image", crop)
        # recolor, reformat
        batch = process(crop)

        return batch

class ImageDataset:
    def __init__(self, path):
        self.img_path = path + '/image_2'
        self.label_path = path + '/label_2'

        self.IDLst = [x.split('.')[0] for x in sorted(os.listdir(self.img_path))] 

    def __getitem__(self, index):
        tmp = {}
        #img = cv2.imread(self.img_path + '/%s.png'%self.IDLst[index], cv2.IMREAD_COLOR)
        with open(self.label_path + '/%s.txt'%self.IDLst[index], 'r') as f:
            buf = []
            for line in f:
                line = line[:-1].split(' ')
                for i in range(1, len(line)):
                    line[i] = float(line[i])
                Class = line[0]
                Alpha = line[3] / np.pi * 180
                Ry = line[14] / np.pi * 180
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]
                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                ThetaRay = (np.arctan2(Location[2], Location[0])) / np.pi * 180
                #if Ry > 0:
                #    LocalAngle = (180 - Ry) + (180 - ThetaRay)
                #else:
                #    LocalAngle = 360 - (ThetaRay + Ry)
                LocalAngle = 360 - (ThetaRay + Ry)
                if LocalAngle > 360:
                    LocalAngle -= 360
                #LocalAngle = Ry - ThetaRay
                LocalAngle = LocalAngle / 180 * np.pi

                if LocalAngle < 0:
                    LocalAngle += 2 * np.pi
                buf.append({
                        'Class': Class,
                        'Box_2D': Box_2D,
                        'Dimension': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry,
                        'ThetaRay': ThetaRay,
                        'LocalAngle': LocalAngle
                    })

        tmp['ID'] = self.IDLst[index]
        #tmp['Image'] = img
        tmp['Label'] = buf
        return tmp
    def GetImage(self, idx):
        name = '%s/%s.png'%(self.img_path, self.IDLst[idx])
        img = cv2.imread(name, cv2.IMREAD_COLOR).astype(np.float) / 255
        img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229
        return img

    def __len__(self):
        return len(self.IDLst)


class BatchDataset:
    def __init__(self, imgDataset, batchSize = 1, bins = 3, overlap = 25/180.0 * np.pi, mode='train'):
        self.imgDataset = imgDataset
        self.batchSize = batchSize
        self.bins = bins
        self.overlap = overlap
        self.mode = mode
        self.imgID = None
        centerAngle = np.zeros(bins)
        interval = 2 * np.pi / bins
        for i in range(1, bins):
            centerAngle[i] = i * interval
        self.centerAngle = centerAngle
        #print centerAngle / np.pi * 180
        self.intervalAngle = interval
        self.info = self.getBatchInfo()
        self.Total = len(self.info)
        if mode == 'train':
            self.idx = 0
            self.num_of_patch = 13
        else:
            self.idx = 13
            self.num_of_patch = 2
        #print len(self.info)
        #print self.info
    def getBatchInfo(self):
        #
        # get info of all crop image
        #   
        data = []
        total = len(self.imgDataset)
        centerAngle = self.centerAngle
        intervalAngle = self.intervalAngle
        for idx, one in enumerate(self.imgDataset):
            ID = one['ID']
            #img = one['Image']
            allLabel = one['Label']
            for label in allLabel:
                if label['Class'] != 'DontCare':
                    #crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
                    LocalAngle = label['LocalAngle']
                    confidence = np.zeros(self.bins)
                    confidence_multi = np.zeros(self.bins)
                    for i in range(self.bins):
                        diff = abs(centerAngle[i] - LocalAngle)
                        if diff > np.pi:
                            diff = 2 * np.pi - diff
                        if diff <= intervalAngle / 2 + self.overlap:
                            confidence_multi[i] = 1
                        if diff < intervalAngle / 2:
                            confidence[i] = 1
                    n = np.sum(confidence)
                    data.append({
                                'ID': ID, # img ID
                                'Index': idx, # id in Imagedataset
                                'Box_2D': label['Box_2D'],
                                'Dimension': label['Dimension'],
                                'Location': label['Location'],
                                'LocalAngle': LocalAngle,
                                'Confidence': confidence,
                                'ConfidenceMulti': confidence_multi,
                                'Ntheta':n,
                                'Ry': label['Ry'],
                                'ThetaRay': label['ThetaRay']
                            })
        return data

    def Next(self):
        batch = np.zeros([self.batchSize, 3, 224, 224], np.float) 
        confidence = np.zeros([self.batchSize, self.bins], np.float)
        confidence_multi = np.zeros([self.batchSize, self.bins], np.float)
        ntheta = np.zeros(self.batchSize, np.float)
        angleDiff = np.zeros([self.batchSize, self.bins], np.float)
        dim = np.zeros([self.batchSize, 3], np.float)
        record = None
        for one in range(self.batchSize):
            data = self.info[self.idx]
            imgID = data['Index']
            if imgID != record:
                img = self.imgDataset.GetImage(imgID)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #cv2.namedWindow('GG')
                #cv2.imshow('GG', img)
                #cv2.waitKey(0)
            pt1 = data['Box_2D'][0]
            pt2 = data['Box_2D'][1]
            crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
            crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            batch[one, 0, :, :] = crop[:, :, 2]
            batch[one, 1, :, :] = crop[:, :, 1]
            batch[one, 2, :, :] = crop[:, :, 0]
            confidence[one, :] = data['Confidence'][:]
            confidence_multi[one, :] = data['ConfidenceMulti'][:]
            #confidence[one, :] /= np.sum(confidence[one, :])
            ntheta[one] = data['Ntheta']
            angleDiff[one, :] = data['LocalAngle'] - self.centerAngle
            dim[one, :] = data['Dimension']
            if self.mode == 'train':
                if self.idx + 1 < self.num_of_patch:
                    self.idx += 1
                else:
                    self.idx = 0
            else:
                if self.idx + 1 < self.Total:
                    self.idx += 1
                else:
                    self.idx = 35570
        return batch, confidence, confidence_multi, angleDiff, dim

    def EvalBatch(self):
        batch = np.zeros([1, 3, 224, 224], np.float) 
        info = self.info[self.idx]
        imgID = info['Index']
        if imgID != self.imgID:
            self.img = self.imgDataset.GetImage(imgID)
            self.imgID = imgID
        pt1 = info['Box_2D'][0]
        pt2 = info['Box_2D'][1]
        crop = self.img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        batch[0, 0, :, :] = crop[:, :, 2]
        batch[0, 1, :, :] = crop[:, :, 1]
        batch[0, 2, :, :] = crop[:, :, 0]

        if self.mode == 'train':
            if self.idx + 1 < self.num_of_patch:
                self.idx += 1
            else:
                self.idx = 0
        else:
            if self.idx + 1 < self.Total:
                self.idx += 1
            else:
                self.idx = 10
        return batch, self.centerAngle,