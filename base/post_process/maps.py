import cv2
import numpy as np
from numpy.linalg import norm


class ResNetMap():
    
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

#______________________________
class AutoEncoderMap():
    
    def __init__(self, coords_path='coords.npy' , encoded_path='map.npy'):
        self.cur_angle = 0
        self.cur_scale = 0.8
        self.last_vector = None
        self.y_step_number = 316
        self.current_position = None
        self.square = 9*9 #the area around the position
        with open(encoded_path, 'rb') as f:
            self.vectors = np.load(f)
        with open(coords_path, 'rb') as f:
            self.coords = np.load(f)
        self.reference_indexes = np.arange(0, len(self.vectors), 1, dtype=int) 

    def get_position(self, vector):
        cos_max = 0.1
        found_indx = 0
        if len(self.reference_indexes) > 324: # 18*18
            db_vectors = self.vectors[45820:45836] # we know start position
        else:
            db_vectors = self.vectors[self.reference_indexes]
        # TODO
        # C-Python insert
        for n, vec in enumerate(db_vectors):
            res = np.dot(vec, vector)/(norm(vec)*norm(vector))
            if res > cos_max:
                cos_max = res
                found_indx = n
        return found_indx

    def get_reference_indexes(self, found_indx):
        indexes = []
        for step in range(-4, 5):
            x_ind = found_indx - step*self.y_step_number
            y_ind = [i for i in range((x_ind - 4), (x_ind + 5)) if i>0]
            if y_ind == []:
                continue
            indexes.append(y_ind)
        indexes = np.array(indexes)
        self.square = 9*len(indexes)
        self.reference_indexes = np.reshape(indexes, (self.square, 1))
        return self.reference_indexes

    def get_geo_data(self, reference_indexes):
        reference_imgs = self.vectors[reference_indexes] # Yge
        reference_coords = self.coords[reference_indexes] # Xge
        reference_imgs = np.reshape(reference_imgs, (self.square, 2000))
        reference_coords = np.reshape(reference_coords, (self.square, 2))
        return reference_imgs, reference_coords

input_size = 64
segment_number = 25
img_path = '/path/to/img/map_file.jpg'
resnet_map = 0 #ResNetMap(img_path, input_size, segment_number)

encoded_map = '/path/to/vectors/map.npy'
coords = '/path/to/coords/xy.npy'
autoen_map = AutoEncoderMap(coords, encoded_map)
