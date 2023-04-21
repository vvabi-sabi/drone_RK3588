from multiprocessing import Queue
import json
from pathlib import Path
import cv2
import numpy as np

from base.utils import format_dets
from .maps import resnet_map, autoen_map

CONFIG_FILE = str(Path(__file__).parent.parent.parent.absolute()) + "/config.json"
with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)


def post_yolov5(outputs, frame):
    """Overlays bboxes on frames
    Args
    -----------------------------------
    q_in : multiprocessing.Queue
        Queue that data reads from
    q_out : multiprocessing.Queue
        Queue that data sends to
    -----------------------------------
    """
    dets = None
    data = list()
    for out in outputs:
        out = out.reshape([3, -1]+list(out.shape[-2:]))
        data.append(np.transpose(out, (2, 3, 0, 1)))
    boxes, classes, scores = yolov5_post_process(data)
    if boxes is not None:
        draw(frame, boxes, scores, classes)
        dets = format_dets(
            boxes = boxes,
            classes = classes, # type: ignore
            scores = scores # type: ignore
        )
    return frame, dets

def post_unet(outputs, frame):
    alpha = 0.7
    beta = (1.0 - alpha)
    frame_shape = (frame.shape[1], frame.shape[0])
    raw_mask = np.array(outputs[0][0])
    pred_mask = get_mask(raw_mask)
    pred_mask = cv2.resize(pred_mask, frame_shape, interpolation=cv2.INTER_CUBIC)
    #frame = np.hstack([frame, pred_mask])
    frame = cv2.addWeighted(frame, alpha, pred_mask, beta, 0.0)
    return (frame, )

def post_resnet(outputs, frame):
    y_step_number = 25 # crop number
    z_dataset_photo = 200
    images_list = resnet_post_process(outputs)
    hm.found_img = frame
    hm.imgs_list = images_list
    index, scale, angle = hm()
    if type(scale) == type(': nan'):
        index = 0.
        x, y = 0., 0.
        scale = 1.
        angle = 0.
    draw_position(frame, index, angle)
    x = index//y_step_number
    y = index%y_step_number
    z = scale*z_dataset_photo
    coords = np.array([x, y, z, angle])
    return frame, coords

def post_autoencoder(outputs, frame):
    z_dataset_photo = 200
    vector = outputs[0]
    index = autoen_map.get_position(vector)
    indexes = autoen_map.get_reference_indexes(index) # [96200 x 1000] or [324 x 1000]
    xy = xy_compute(vector, indexes)       
    # determine the angle of rotation and rotate the source photo
    rotation_list = outputs[:72]
    scale_list = outputs[72:]
    vec = autoen_map.vectors[index] # True Map Crop
    angle = find_angle(rotation_list, vec)
    scale = find_scale(scale_list, vec)
    
    z = scale*z_dataset_photo
    coords = np.array([xy[0], xy[1], z, angle])
    return frame, coords

#__YOLOv5__
def sigmoid(x):
    return x# 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def detect(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(cfg["inference"]["net_size"]/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin
    yolov5 post process!
    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.
    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= cfg["inference"]["obj_thresh"])
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= cfg["inference"]["obj_thresh"])

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= cfg["inference"]["nms_thresh"])[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = detect(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.
    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        width = cfg["camera"]["width"]
        height = cfg["camera"]["height"]
        net_size = cfg["inference"]["net_size"]
        top, left, right, bottom = box
        top = int(top*(width / net_size))
        left = int(left*(height / net_size))
        right = int(right*(width / net_size))
        bottom = int(bottom*(height / net_size))

        cv2.rectangle(
            img=image,
            pt1=(top, left),
            pt2=(right, bottom),
            color=(255, 0, 0),
            thickness=2
        )
        cv2.putText(
            img=image,
            text='{0} {1:.2f}'.format(cfg["inference"]["classes"][cl], score),
            org=(top, left - 6),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 0, 255),
            thickness=2
        )
#__________________________

#__UNet__
def get_mask(raw_mask):
    select_class_rgb_values = np.array( [[0, 0, 0], [255, 255, 255]])
    pred_mask = np.transpose(raw_mask,(1,2,0))
    pred_mask = reverse_one_hot(pred_mask)
    pred_mask = colour_code_segmentation(pred_mask, select_class_rgb_values)
    return pred_mask.astype(np.uint8)

# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x
#__________________________

#__ResNet__
def resnet_post_process(result):
    output = result[0].reshape(-1)
    # softmax
    output = np.exp(output)/sum(np.exp(output))
    output_sorted = sorted(output, reverse=True)
    top = output_sorted[:10]
    images_list = [resnet_map.get_crop_by_id(int(cls_id)) for cls_id in top]
    return images_list

def draw_position(image, scale, angle):
    width = image.shape[1]
    height = image.shape[0]
    top = int((height/2))
    left = int(50)
    cv2.putText(img=image,
                text=f'index-{scale}, angle-{angle}',
                org=(top, left),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 0, 255),
                thickness=2)

class SiftData:
    
    def __init__(self, kpts, desc):
        self.kp = kpts
        self.desc = desc

class Homograpy_Match:
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.index_params = dict(algorithm = 0, trees = 5)
        self.search_params = dict()
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.dd = 0.6 #descriptors distance coeff
        self.img = None
        self.source_imgs = []
        
        self.img_data = None
        self.sorce_data_list = []
    
    @property
    def found_img(self):
        return self.img
    
    @found_img.setter
    def found_img(self, np_img):
        self.img = np_img
    
    @property
    def imgs_list(self):
        return self.source_imgs
    
    @imgs_list.setter
    def imgs_list(self, imgs_list):
        self.source_imgs = imgs_list
    
    def find_most_simular(self):
        kp_img, desc_img = self.sift.detectAndCompute(self.img , None)
        self.img_data = SiftData(kp_img, desc_img)
        index = 0
        matrix = 0
        for i, img in enumerate(self.source_imgs):
            kp_img, desc_img = self.sift.detectAndCompute(img , None)
            self.sorce_data_list.append(SiftData(kp_img, desc_img))
            try:
                matrix = self.get_matrix(i)
                index = i
            except:
                continue

        return index, matrix
    
    def get_matrix(self, index):
        matches = self.flann.knnMatch(self.img_data.desc, 
                                 self.sorce_data_list[index].desc, k=2)
        good_points = []
        for m, n in matches:
            if(m.distance < self.dd*n.distance):
                good_points.append(m)
        query_pts = np.float32([self.img_data.kp[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([self.sorce_data_list[index].kp[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, _ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0) #ошибка
        return matrix
    
    def get_angle(self, matrix):
        a = 0 
        l = 1 # - zoom
        l =  np.nanmax([np.sqrt(np.absolute(matrix[0][0])*(matrix[1][1]) -(matrix[0][1])*(matrix[1][0]) ), 
                        np.sqrt((matrix[0][0])*(matrix[0][0]) +(matrix[0][1])*(matrix[0][1]) ),
                        np.sqrt((matrix[1][0])*(matrix[1][0]) +(matrix[1][1])*(matrix[1][1]) ) ])
        l = np.round(l,2)
        matrix /= l
        for j in range(2):
            if matrix[j][j] > 1:
                matrix[j][j] = 1
            a += -1* np.arccos(np.round(matrix[j][j],3))
        a *= 180/np.pi
        a /= 2 
        return l, np.round(a,2)
    
    def __call__(self):
        index, matrix = self.find_most_simular()
        if type(matrix) != type(np.array(0)):
            return ': nan', ': nan', ': nan'
        scale, angle = self.get_angle(matrix)
        return index, scale, angle

hm = Homograpy_Match()

#____________________________

#_____AutoEncoder____
def get_wth(w):
	std_w = np.std(w)
	max_w = w.max()
	# discard outliers
	limit = max_w - std_w
	w[w<limit] = 0
	summ = w.sum()
	return w/summ

def xy_compute(vector, indexes):
    reference_img, reference_coord = autoen_map.get_geo_data(indexes)
    w = reference_img @ vector # Yge * y
    Wth = get_wth(w)
    xy = reference_coord.T @ Wth # Xge * Wth
    return xy

def find_angle(vectors, vec):
    angles = [angle for angle in range(0, 360, 5)]
    angles = np.array(angles)
    w = vectors @ vec # Scale_ge * vec
    Wth = get_wth(w)
    found_angle = angles.T @ Wth
    return found_angle

def find_scale(vectors, vec):
    scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]#, 1.7]
    scales = np.array(scales)
    w = vectors @ vec # Scale_ge * vec
    Wth = get_wth(w)
    found_scale = scales.T @ Wth
    return found_scale
