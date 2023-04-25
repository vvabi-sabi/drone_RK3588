# RKNPU2
## YOLOv5
![yolo_result](images/YOLOv5.png)

```
photo --> YOLOv5 --> bbox centers
```
## ResNet
![resnet_result](images/ResNet.png)

```
photo --> ResNet --> index, crop (localization)
crop --> SIFT --> scale, angle (and orientation)
```

## UNet
![unet_result](images/UNet.png)

```
photo --> UNet --> mask
```

## AutoEncoder
![ae_result](images/AutoEncoder.png)

```
photo --> AE --> vector
vector --> db --> index, scale, angle (localization and orientation)
```

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
python3 resnet2rknn.py \
        --input path-to-your-pt-model \
        --output path-where-save-rknn-model \
        --dataset path-to-txt-file-with-calibration-images-names
```
