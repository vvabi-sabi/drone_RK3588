# RKNPU2
...

# ONNX2RKNN convert
Unet
```
python3 unet_onnx2rknn.py \
        --input path-to-your-onnx-model \
        --output path-where-save-rknn-model \
        --dataset path-to-txt-file-with-calibration-images-names
```
ResNet
```
python3 resnet18_qat.py \
        --input path-to-your-pt-model \
        --output path-where-save-rknn-model \
        --dataset path-to-txt-file-with-calibration-images-names
```
