import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import os
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform
import tensorflow as tf

class SiameseNetworkDataset():
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0', '1':'1', '0':'0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1', '1':'1', '0':'0'}, ]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            i = 0
            image_two = []
            labels_two = []
            for line in f:
                
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    '''
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                            '''
                    
                    if index == 2:# or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                    '''
                        if self.dict[0].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                            '''
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
               # path = "/kaggle/input/chexpert/"os.path.relpath(path) + 
                image_path = "/kaggle/input/chexpert/" + image_path[21:]
                image_two.append(image_path)
                labels_two.append(labels)
                '''
                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
                '''
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

        cv2.imshow('IMAGE0',img0)
        cv2.imshow('IMAGE1',img1)

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