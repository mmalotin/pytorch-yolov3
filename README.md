# YOLO v3 Object Detector implemented in PyTorch
This repository contains PyTorch implementation of [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf) paper.
__Why this repo?__ The code implements the paper in pythonic/"pytorch*ish*" style and will be familiar and (I hope) helpful for PyTorch users for understanding the paper/using YOLOv3 object detector. Original pretrained weights (on COCO dataset) were converted into PyTorch's state_dict and can be downloaded [here](https://www.dropbox.com/s/6rd0392l3psmm10/yolo3_weights.pth.zip?dl=0) (no need for complicated network builders and weights converters just common `torch.load()`).

## Requirements
- Python 3
- PyTorch (0.4)
- torchvision
- OpenCV



## Usage
The repo contains 2 runable scripts.
- Detection:
```
$ python detect.py images/dog.jpg
```
If you want to specify your own weights file use `-weights path/to/weights` or if you want to save the detection result use `-save` flag (__Note__: by default weights should be placed in `data` folder).

- Webcam Detection:
```
$ python webcam.py
```
Arguments are the same, except `webcam.py` doesn't have `-save` flag.

## Examples
- <img src="https://github.com/mmalotin/pytorch-yolov3/blob/master/predictions/city_prediction.jpg?raw=true" width="800" height="512">

- <img src="https://github.com/mmalotin/pytorch-yolov3/blob/master/predictions/traffic_prediction.jpg?raw=true" width="512" height="512">

- <img src="https://github.com/mmalotin/pytorch-yolov3/blob/master/predictions/dog_prediction.jpg?raw=true" width="612" height="512">
