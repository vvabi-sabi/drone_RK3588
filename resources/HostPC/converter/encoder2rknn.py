import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = 'encoder/encoder.onnx'
RKNN_MODEL = 'encoder/encoder.rknn'


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    if not os.path.exists(ONNX_MODEL):
        print(f' Not exists {ONNX_MODEL}')
        exit(-1)

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[127.5], std_values=[127.5], target_platform="rk3588")
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset/dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./dataset/14.png')
    print('img', img.shape)
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img = np.expand_dims(input_img, 0)
    input_img = np.expand_dims(input_img, axis=-1) # !!
    print('input_img', input_img.shape)
    
    # Init runtime environment img = cv2.resize(frame, (config.NET_SIZE, config.NET_SIZE))
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    outputs = rknn.inference(inputs=input_img)
    print('\nLen', len(outputs))
    print(outputs)
    print('done')

    rknn.release()
