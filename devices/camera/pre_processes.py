import json
from pathlib import Path
import cv2
import numpy as np

Mat = np.ndarray[int, np.dtype[np.generic]]


def pre_yolov5(frame: Mat, net_size:int):
    """
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    net_size = (net_size, net_size)
    frame = cv2.resize(frame,net_size)
    return frame

def pre_yolact(frame: Mat, net_size:int):
    """
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    net_size = (net_size, net_size)
    frame = cv2.resize(frame,net_size)
    return frame

def pre_unet(frame: Mat, net_size:int):
    """
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    net_size = (net_size, net_size)
    height = frame.shape[0]
    x_0 = int((frame.shape[1]- height)/2)
    frame = frame[::, x_0:x_0+height]
    return cv2.resize(frame, net_size)

def pre_resnet(frame: Mat, net_size:int):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    net_size = (net_size, net_size)
    height = frame.shape[0]
    x_0 = int((frame.shape[1]- height)/2)
    frame = frame[::, x_0:x_0+height]
    return cv2.resize(frame, net_size)


def pre_autoencoder(frame: Mat, net_size:int):
    raw_source = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw_source = cv2.merge((raw_source,raw_source,raw_source))
    angles = rotate_imgs_list(raw_source) # Model.inference
    scales = scale_imgs_list(raw_source)
    transforms = angles + scales
    return np.array(transforms)


def rotate_imgs_list(raw):
    img_list = []
    for angle in [0,-5,-4,-3,-2,-1,1,2,3,4,5]:
        rot_img = rotate_image(raw, angle)
        #rot_crop = crop_image(rot_img)
        img_list.append(rot_img) #rot_crop)
    return img_list

def rotate_image(image, angle=None):
    if angle is None:
        angle = np.random.randint(5, 180)
    height, width = image.shape[:2]
    center = (width/2, height/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    result = cv2.warpAffine(image, rot_mat, dsize=(width, height), flags=cv2.INTER_LINEAR)
    return result

def scale_imgs_list(raw):
    img_list = []
    scales = [0.85, 0.9, 0.95, 1., 1.05, 1.1, 1.15]
    for scale in scales:
        scl_img = scale_image(raw, scale)
        #scl_crop = crop_image(scl_img)
        img_list.append(scl_img) #scl_crop)
    return img_list

def scale_image(image, scale=None):
    if scale is None:
        scale = np.random.uniform(0.7, 1.5)
    scale = np.round(scale, 4)
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    result = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #(photo, (320, 160), fx=scale, fy=scale)
    return result

# def crop_image(image):
#     NET_SIZE = cfg["inference"]["net_size"] # (320, 160)
#     cx, cy = int(image.shape[0]/2), int(image.shape[1]/2)
#     NET_SIZE_0 = int(NET_SIZE[0]/2)
#     NET_SIZE_1 = int(NET_SIZE[1]/2)
#     ltrb = cx - NET_SIZE_1, cy - NET_SIZE_0, cx + NET_SIZE_1, cy + NET_SIZE_0

#     ltrb = [max(0,i) for i in ltrb]
#     return image[ltrb[0]:ltrb[2], ltrb[1]:ltrb[3]] # crop
