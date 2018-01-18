import os
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
from glob import glob
import os.path
import re
import random
import cv2
from torch.utils import data

KITTI_CLASSES= [
    'BG','Car','Van','Truck',
    'Pedestrian','Person_sitting',
    'Cyclist','Tram','Misc','DontCare'
    ]


class Class_to_ind(object):
    def __init__(self,binary,binary_item):
        self.binary=binary
        self.binary_item=binary_item
        self.classes=KITTI_CLASSES

    def __call__(self, name):
        if not name in self.classes:
            raise ValueError('No such class name : {}'.format(name))
        else:
            if self.binary:
                if name==self.binary_item:
                    return True
                else:
                    return False
            else:
                return self.classes.index(name)
# def get_data_path(name):
#     js = open('config.json').read()
#     data = json.loads(js)
#     return data[name]['data_path']

class AnnotationTransform_kitti(object):
    '''
    Transform Kitti detection labeling type to norm type:
    source: Car 0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 1.62
    target: [xmin,ymin,xmax,ymax,label_ind]

    levels=['easy','medium']
    '''
    def __init__(self,class_to_ind=Class_to_ind(True,'Car'),levels=['easy','medium','hard']):
        self.class_to_ind=class_to_ind
        self.levels=levels if isinstance(levels,list) else [levels]

    def __call__(self,target_lines,width,height):

        res=list()
        for line in target_lines:
            xmin,ymin,xmax,ymax=tuple(line.strip().split(' ')[4:8])
            bnd_box=[xmin,ymin,xmax,ymax]
            new_bnd_box=list()
            for i,pt in enumerate(range(4)):
                cur_pt=float(bnd_box[i])
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                new_bnd_box.append(cur_pt)
            label_idx=self.class_to_ind(line.split(' ')[0])
            new_bnd_box.append(label_idx)
            res.append(new_bnd_box)
        return res

class KittiLoader(data.Dataset):
    def __init__(self, root, split="training",
                 img_size=512, transforms=None,target_transform=None):
        self.root = root
        self.split = split
        self.target_transform = target_transform
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.transforms = transforms
        self.name='kitti'

        for split in ["training", "testing"]:
            file_list = glob(os.path.join(root, split, 'image_2', '*.png'))
            self.files[split] = file_list

            if not split=='testing':
                label_list=glob(os.path.join(root, split, 'label_2', '*.txt'))
                self.labels[split] = label_list


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = img_name

        #img = m.imread(img_path)
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        #img = np.array(img, dtype=np.uint8)

        if self.split != "testing":
            lbl_path = self.labels[self.split][index]
            lbl_lines=open(lbl_path,'r').readlines()
            if self.target_transform is not None:
                target = self.target_transform(lbl_lines, width, height)
        else:
            lbl = None

        # if self.is_transform:
        #     img, lbl = self.transform(img, lbl)

        if self.transforms is not None:
            target = np.array(target)
            img, boxes, labels = self.transforms(img, target[:, :4], target[:, 4])
            #img, lbl = self.transforms(img, lbl)
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))            

        if self.split != "testing":
            #return img, lbl
            return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        else:
            return img

