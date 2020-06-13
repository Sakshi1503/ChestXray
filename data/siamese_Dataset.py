import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import os
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform
import tensorflow as tf
import random

class SiameseNetworkDataset():
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0', '1':'1', '0':'0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1', '1':'1', '0':'0'}, ]
        
        i = 0
        image_two = []
        labels_two = []

        lines = open(label_path).read().split('\n')
        header = lines.pop(0)
        self._label_header = [
            header[7],
            header[10],
            header[11],
            header[13],
            header[15]]

        for _ in range(2000):
            line0 = random.choice(lines)
            fields0 = line0.strip('\n').split(',')
            should_get_same_class = random.randint(0,1)
            if should_get_same_class:
                while True:
                    line1 = random.choice(lines) 
                    fields1 = line1.strip('\n').split(',')
                    print(fields0)
                    print(fields1)
                    print("************************************")
                    if self.dict[0].get(fields0[7]) == self.dict[0].get(fields1[7]):
                        break
            else:
                line1 = random.choice(lines) 
                fields1 = line1.strip('\n').split(',')

            image_path = fields0[0]
            image_path = "/kaggle/input/chexpert/" + image_path[21:]
            image_two.append(image_path)
            image_path = fields1[0]
            image_path = "/kaggle/input/chexpert/" + image_path[21:]
            image_two.append(image_path)
            print(fields0)
            labels_two.append(self.dict[0].get(fields0[7]))
            print(fields1)
            labels_two.append(self.dict[0].get(fields1[7]))
            
            i+=1
            if i==2:
                i=0
                self._image_paths.append(image_two)
                if(labels_two[0] == labels_two[1]):
                    self._labels.append(0)
                else:
                    self._labels.append(1)
                image_two = []
                labels_two = []
        self._num_image = len(self._image_paths)

    def __getitem__(self,index):
        #if index % 2 == 0:  
        
        
        img0 = cv2.imread(self._image_paths[index][0], 0)        
        img1 = cv2.imread(self._image_paths[index][1], 0)

        img0 = Image.fromarray(img0)
        img1 = Image.fromarray(img1)
        
        if self._mode == 'train':
            img0 = GetTransforms(img0, type=self.cfg.use_transforms_type)
            img1 = GetTransforms(img1, type=self.cfg.use_transforms_type)
        img0 = np.array(img0)
        img1 = np.array(img1)    
        
        img0 = transform(img0, self.cfg)
        img1 = transform(img1, self.cfg)
        
        labels = np.array(self._labels[index]).astype(np.float32) 
        print(self._image_paths[index][0],self._image_paths[index][1],labels)

        img0 = torch.from_numpy(img0).float()
        img1 = torch.from_numpy(img1).float()
        labels = torch.from_numpy(labels).float()

        if self._mode == 'train' or self._mode == 'dev':
            return (img0, img1, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
        
                #return img0, img1 , torch.from_numpy(np.array([int(self.training_df.iat[index,2])],dtype=np.float32))
        return img0, img1 , labels
    
    def __len__(self):
        return self._num_image