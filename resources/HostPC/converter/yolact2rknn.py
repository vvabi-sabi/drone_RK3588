import argparse
import os
import cv2
import sys
from rknn.api import RKNN

QUANTIZE_ON = False

IMG_SIZE = 544

def parse_opt():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--input', '-i',
        required=True,
        default=os.path.abspath(os.getcwd())+"/yolact/yolact.onnx",
        type=str,
        help='path to onnx model'
    )
    parser.add_argument(
        '--ouput', '-o',
        default=os.path.abspath(os.getcwd())+"/yolact/yolact.rknn",
        type=str,
        help='path to RKNN model'
    )
    parser.add_argument(
        '--dataset', '-d',
        default=os.path.abspath(os.getcwd())+"/dataset/dataset.txt",
        type=str,
        help='path to calibration dataset (txt file with names of images)'
    )
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()

def main(opt):
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=opt.input)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=opt.dataset)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(opt.ouput)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # __TEST____
    # Set inputs
    img = cv2.imread('./dataset/test_544.jpg')
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('input_img', input_img.shape)
    assert(input_img.ndim==3 and input_img.shape[0]==IMG_SIZE), "Frame: provided image has not the same size as the model"
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    outputs = rknn.inference(inputs=input_img)
    print(outputs)
    print('done')
    rknn.release()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
