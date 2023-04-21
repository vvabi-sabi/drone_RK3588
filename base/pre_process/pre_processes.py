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

def pre_unet(frame):
    NET_SIZE = cfg["inference"]["net_size"]
    #cx, cy = int(frame.shape[0]/2), int(frame.shape[1]/2)
    #ltrb = cx - NET_SIZE, cy - NET_SIZE, cx + NET_SIZE, cy + NET_SIZE
    #return frame[ltrb[0]:ltrb[2], ltrb[1]:ltrb[3]]
    return cv2.resize(frame, (NET_SIZE, NET_SIZE))

def pre_resnet(frame):
    NET_SIZE = cfg["inference"]["net_size"]
    return cv2.resize(frame, (NET_SIZE, NET_SIZE))

def pre_autoencoder(frame):
	NET_SIZE = cfg["inference"]["net_size"] # (320, 160)
	cx, cy = int(frame.shape[0]/2), int(frame.shape[1]/2)
	ltrb = cx - NET_SIZE[0]/2, cy - NET_SIZE[1]/2, cx + NET_SIZE[0]/2, cy + NET_SIZE[1]/2
	frame = frame[ltrb[0]:ltrb[2], ltrb[1]:ltrb[3]]
	return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
