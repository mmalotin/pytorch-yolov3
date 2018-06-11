#!/bin/bash
mkdir -p data
cd data
wget https://www.dropbox.com/s/6rd0392l3psmm10/yolo3_weights.pth.zip?dl=0
unzip yolo3_weights.pth.zip\?dl\=0
rm yolo3_weights.pth.zip\?dl\=0
cd ..
