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
        
        self.lines = open(label_path).read().split('\n')
        header = self.lines.pop(0)
        self._label_header = [
            header[7],
            header[10],
            header[11],
            header[13],
            header[15]]
        self.lines.remove('')
        
        self._num_image = 200

    def __getitem__(self,index):
        #if index % 2 == 0:  
        #print(index)
        if(index % 4 == 0):
            while True:
                line0 = random.choice(self.lines) 
                fields0 = line0.strip('\n').split(',')
                if self.dict[0].get(fields0[15]) == '0':
                    break
            while True:
                line1 = random.choice(self.lines) 
                fields1 = line1.strip('\n').split(',')
                if self.dict[0].get(fields1[15]) == '1':
                    break
            self._labels.append(1)

        elif(index % 4 == 1):
            while True:
                line0 = random.choice(self.lines) 
                fields0 = line0.strip('\n').split(',')
                if self.dict[0].get(fields0[15]) == '1':
                    break   
            while True:
                line1 = random.choice(self.lines) 
                fields1 = line1.strip('\n').split(',')
                if self.dict[0].get(fields1[15]) == '0':
                    break
            self._labels.append(1)

        elif(index % 4 == 2 or index % 4 == 3):
            while True:
                line0 = random.choice(self.lines) 
                fields0 = line0.strip('\n').split(',')
                if self.dict[0].get(fields0[15]) == '1':
                    break
            while True:
                line1 = random.choice(self.lines) 
                fields1 = line1.strip('\n').split(',')
                if self.dict[0].get(fields1[15]) == '1':
                    break
            self._labels.append(0)
        '''

        if should_get_same_class:
            while True:
                line1 = random.choice(self.lines) 
                fields1 = line1.strip('\n').split(',')
                if self.dict[0].get(fields0[7]) == self.dict[0].get(fields1[7]):
                    break
        else:
            line1 = random.choice(self.lines) 
            fields1 = line1.strip('\n').split(',')
        '''
        
        image_path = fields0[0]
        image_path = "/kaggle/input/chexpert/" + image_path[21:]
        img0 = cv2.imread(image_path, 0)
        image_path = fields1[0]
        image_path = "/kaggle/input/chexpert/" + image_path[21:]
        img1 = cv2.imread(image_path, 0)

        img0 = Image.fromarray(img0)
        img1 = Image.fromarray(img1)
        
        if self._mode == 'train':
            img0 = GetTransforms(img0, type=self.cfg.use_transforms_type)
            img1 = GetTransforms(img1, type=self.cfg.use_transforms_type)
        img0 = np.array(img0)
        img1 = np.array(img1)    
        
        img0 = transform(img0, self.cfg)
        img1 = transform(img1, self.cfg)
        
        labels = np.array(self._labels[0]).astype(np.float32) 
        #print(self._image_paths[index][0],self._image_paths[index][1],labels)

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