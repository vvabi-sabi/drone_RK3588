import numpy as np
import cv2
import os 

class ResNetDataset():
    
    def __init__(self, path_to_file, input_size=64, segment_number=25):
        self.cr_w, self.cr_h = input_size, input_size
        img = cv2.imread(path_to_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        new_size = segment_number*self.cr_w, segment_number*self.cr_h
        self.im =  cv2.resize(img, new_size) #self.im.resize(new_size, Image.ANTIALIAS)
        self.nbx = segment_number
        self.nby = segment_number
        self.len = segment_number * segment_number
        self.cls_crops = []
        self.__init_classes()
        
    def cls2ltrb(self, cls_id):
        assert cls_id < self.len, f'class id must be less than num of bids {self.len}'
        row = cls_id//self.nbx
        col = cls_id%self.nbx
        left = col * self.cr_w
        top = row * self.cr_h
        right, bottom = left + self.cr_w, top + self.cr_h
        return left, top, right, bottom

    def get_crop_by_id(self, cls_id):
        ltrb = self.cls2ltrb(cls_id)
        crop = self.im[ltrb[0]:ltrb[2], ltrb[1]:ltrb[3]]
        crop = cv2.resize(crop, (self.cr_w, self.cr_h))
        return crop
    
    def __init_classes(self):
        for cls_id in range(self.len):
            row = cls_id//self.nbx
            col = cls_id%self.nbx
            if row == 0 or col == 0:
                continue
            if row == (self.nbx-1) or col == (self.nbx-1):
                continue
            self.source_crop(cls_id) # add img to list
                
    def source_crop(self, cls_id):
        ltrb = self.cls2ltrb(cls_id)
        crop = self.im[ltrb[0]:ltrb[2], ltrb[1]:ltrb[3]]
        crop = cv2.resize(crop, (self.cr_w, self.cr_h))
        self.cls_crops.append([crop, cls_id])
        
    def __len__(self):
        return len(self.cls_crops)


input_size = 64
segment_number = 25
path = '/home/firefly/Firefly_ROC-RK3588S-PC/rknpu2_python3/modules/shlyuz.jpg'

resnet_dataset = ResNetDataset(path, input_size, segment_number)

