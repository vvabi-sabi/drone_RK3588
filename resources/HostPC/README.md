# RKNPU2

## Install dependencies
Conda:

Create conda python3.8 - env
```
conda create -n <env-name> python=3.8
```
Activate virtualenv
```
conda activate <env-name>
```
Install Python3 and pip3
```
sudo apt-get install python3 python3-dev python3-pip
```
Install dependent libraries
```
sudo apt-get update
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 \
libgl1-mesa-glx libprotobuf-dev gcc
```
Install Python dependency
```
#cd drone_RK3588/resources/HostPC/
pip install -r requirements_cp38-1.4.0.txt
#if doesn't installing then install numpy before that
#pip install numpy
```

Install RKNN-Toolkit2
```
sudo pip3 install ./rknn_toolkit2*cp38*.whl
```