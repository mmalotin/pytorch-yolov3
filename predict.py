import torch
from utils import detect, save_im
from darknet import Darknet
from head import Yolo3
import cv2
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cline = argparse.ArgumentParser(description='YOLO v3 objext detection')
cline.add_argument('image', type=str, help='path to the input image')
cline.add_argument('-weights', default='data/yolo3_weights.pth',
                   help='path to pretrained weights')
cline.add_argument('-obj_thold', type=float, default=0.65,
                   help='threshold for objectness value')
cline.add_argument('-nms_thold', type=float, default=0.4,
                   help='threshold for non max supression')
cline.add_argument('-model_res', type=int, default=416,
                   help='resolution of the model\'s input')
cline.add_argument('-save', action='store_true',
                   help='whether to save result or not')


if __name__ == '__main__':
    args = cline.parse_args()
    with torch.no_grad():
        bbone = Darknet()
        bbone = bbone.extractor
        model = Yolo3(bbone)

        print(f'Loading weights from {args.weights}')
        model.load_state_dict(torch.load(args.weights))
        model.to(device)
        image = cv2.imread(args.image)
        res = detect(model, image, device, args.obj_thold,
                   args.nms_thold, args.model_res)
    cv2.imshow('prediction', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if args.save:
        save_im(res, args.image)
