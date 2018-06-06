# YOLO v3 Object Detector implemented in PyTorch
This repository contains PyTorch implementation of [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf) paper.
*Why this repo?* The code implements the paper in clean pythonic/"pytorch*ish*" style and will be familiar to PyTorch users and (I hope) will be helpful for PyTorch users for understanding the paper. Original pretrained weights (on COCO dataset) were converted into PyTorch's state_dict and can be downloaded [here]() (no need for complicated network builders and weights converters just common `torch.load()`).

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
If you want to specify your own weights file use `-weights path/to/weights` or if you want to save the detection result use `-save` flag.

- Webcam Detection:
```
$ python webcam.py
```
Arguments are the same, except `webcam.py` doesn't have `-save` flag.

## Examples
- <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/mosaic.jpg" width="512" height="512">
- <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/mosaic.jpg" width="512" height="512">
- <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/mosaic.jpg" width="512" height="512">
