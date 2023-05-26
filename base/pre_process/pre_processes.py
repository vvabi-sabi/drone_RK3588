import json
from pathlib import Path

import cv2
import numpy as np


CONFIG_FILE = str(Path(__file__).parent.parent.parent.absolute()) + "/config.json"
with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)
Mat = np.ndarray[int, np.dtype[np.generic]]


def pre_yolov5(frame: Mat):
    """Resizing raw frames from original size to net size
    Args
    -----------------------------------
    frame : np.ndarray
        Raw frame
    -----------------------------------
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(
        frame,
        (cfg["inference"]["net_size"], cfg["inference"]["net_size"])
    )
    return frame

def pre_yolact(frame: Mat):
    """
    """
    net_size = cfg["inference"]["net_size"]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    net_size = (net_size, net_size)
    frame = cv2.resize(frame,net_size)
    return frame

def pre_unet(frame):
    NET_SIZE = cfg["inference"]["net_size"]
    #cx, cy = int(frame.shape[0]/2), int(frame.shape[1]/2)
    #ltrb = cx - NET_SIZE, cy - NET_SIZE, cx + NET_SIZE, cy + NET_SIZE
    #return frame[ltrb[0]:ltrb[2], ltrb[1]:ltrb[3]]
    return cv2.resize(frame, (NET_SIZE, NET_SIZE))

def pre_resnet(frame):
    NET_SIZE = cfg["inference"]["net_size"]
    return cv2.resize(frame, (NET_SIZE, NET_SIZE))



def pre_autoencoder(frame, cur_angl=0, cur_scl=1.):
    raw_source = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Camera() -> RGB
    raw_source = cv2.merge((raw_source,raw_source,raw_source))
    angles = rotate_imgs_list(raw_source) # Model.inference
    scales = scale_imgs_list(raw_source)
    transforms = angles + scales
    return np.array(transforms)


def rotate_imgs_list(raw):
    img_list = []
    for angle in [0,-5,-4,-3,-2,-1,1,2,3,4,5]:
        rot_img = rotate_image(raw, angle)
        rot_crop = crop_image(rot_img)
        img_list.append(rot_crop)
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
        scl_crop = crop_image(scl_img)
        img_list.append(scl_crop)
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

def crop_image(image):
    NET_SIZE = cfg["inference"]["net_size"] # (320, 160)
    cx, cy = int(image.shape[0]/2), int(image.shape[1]/2)
    NET_SIZE_0 = int(NET_SIZE[0]/2)
    NET_SIZE_1 = int(NET_SIZE[1]/2)
    ltrb = cx - NET_SIZE_1, cy - NET_SIZE_0, cx + NET_SIZE_1, cy + NET_SIZE_0

    ltrb = [max(0,i) for i in ltrb]
    return image[ltrb[0]:ltrb[2], ltrb[1]:ltrb[3]] # crop
