import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
import os

MODEL = './resnet18.pt'

def export_pytorch_model():
    net = models.resnet18(pretrained=True)
    net.eval()
    trace_model = torch.jit.trace(net, torch.Tensor(1, 3, 64, 64))
    trace_model.save('./resnet18.pt')


def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    print("\noutput", output)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':

    model = MODEL
    if not os.path.exists(model):
        export_pytorch_model()

    input_size_list = [[1, 3, 64, 64]]

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[123.675, 116.28, 103.53], std_values=[58.395, 58.395, 58.395], target_platform="rk3588")
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset/resnet_dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('resnet/resnet_18_quant.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./dataset/414.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    input_img = np.expand_dims(img, axis=0) 
    outputs = rknn.inference(inputs=[input_img])
    np.save('./pytorch_resnet18_0.npy', outputs[0])
    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn.release()
