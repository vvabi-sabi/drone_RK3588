# RKNPU2

## Install
Install miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh
```
Create conda python3.9 - env
```
conda create -n <env-name> python=3.9
```
Activate virtualenv
```
conda activate <env-name>
```
Install RKNN-Toolkit2-Lite
```
cd drone_RK3588/install/
pip install rknn_toolkit_lite2-1.4.0-cp39-cp39-linux_aarch64.whl
pip install -r requirements.txt
```
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

# onnx2rknn convert
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
