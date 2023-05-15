# UAV navigation based on Firefly ROC-RK3588S
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
