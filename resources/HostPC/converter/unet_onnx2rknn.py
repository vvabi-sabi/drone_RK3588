import os
from rknn.api import RKNN

OUTPUTS_NODES = None
# ./Portrait_PP_HumanSegV2_Lite_256x144_infer/Portrait_PP_HumanSegV2_Lite_256x144_infer.onnx
model_base_name = 'unet/unet_256'
ONNX_MODEL = model_base_name + '.onnx' # model_path
RKNN_MODEL = model_base_name + '.rknn'


if __name__ == "__main__":
    rknn = RKNN(verbose=True)

    # Config
    print('--> config model')
    mean_values = [127.5, 127.5, 127.5] #yaml_config["mean"]
    std_values = [127.5, 127.5, 127.5] #yaml_config["std"]
    target_platform = "rk3588"
    rknn.config(
        mean_values=mean_values,
        std_values=std_values,
<<<<<<< HEAD
        target_platform=target_platform) #quant_img_RGB2BGR=True,  quantized_method='layer', quantized_algorithm='mmse') 
=======
        target_platform=target_platform)
>>>>>>> a41b235c55dcc21b32472f18c6f35413e007bbc1
    print('done')

    # Load ONNX model
    if OUTPUTS_NODES is None:
        ret = rknn.load_onnx(model=ONNX_MODEL)
    else:
        ret = rknn.load_onnx(
                model=ONNX_MODEL,
                outputs=OUTPUTS_NODES)
    assert ret == 0, "Load model failed!"

    # Build model
    do_quantization=True
    ret = rknn.build(do_quantization=do_quantization, dataset='./dataset/dataset.txt')
    assert ret == 0, "Build model failed!"

    # Init Runtime
    ret = rknn.init_runtime()
    assert ret == 0, "Init runtime environment failed!"

    # Export
    model_device_name = target_platform.lower()
    if do_quantization is True:
        model_save_name = model_base_name + "_" + model_device_name + "_quantized" + ".rknn"
    else:
        model_save_name = model_base_name + "_" + model_device_name + "_unquantized" + ".rknn"
    
    ret = rknn.export_rknn(os.path.join(model_save_name))
    assert ret == 0, "Export rknn model failed!"
    print("Export OK!")
