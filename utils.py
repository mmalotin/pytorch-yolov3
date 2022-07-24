import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from pathlib import Path


def create_class_list(file):
    with open(file, 'r') as f:
        classes = f.read().split('\n')[:-1]
    return classes


CLASS_LIST = create_class_list('data/coco.names')

COLORS = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (0, 255, 255), (255, 0, 255), (192, 192, 192),
          (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0),
          (128, 0, 128), (0, 128, 128), (0, 0, 128)]


def create_x_grid(w, h, num_anchors, bs):
    grid_x = torch.linspace(0, w-1, w)
    grid_x = grid_x.repeat(h).unsqueeze(0).repeat(bs, num_anchors, 1, 1)
    return grid_x


def create_y_grid(w, h, num_anchors, bs):
    grid_y = torch.linspace(0, h-1, h)
    grid_y = (grid_y.view(-1, 1).repeat(1, w).view(-1)      # ~np.tile
              .unsqueeze(0).repeat(bs, num_anchors, 1, 1))
    return grid_y


def create_wh(x, w, h, num_anchors, bs):
    p = (torch.Tensor(x).view(-1, 1).repeat(1, h*w)
         .view(num_anchors, -1).unsqueeze(1).repeat(bs, 1, 1, 1))
    return p


def convert_to_boxes(out, anchors, model_res):
    out = out.cpu()
    bs, ch, h, w = out.size()
    out = out.view(bs, len(anchors), ch//len(anchors), h*w)
    grid_x = create_x_grid(w, h, len(anchors), bs)
    grid_y = create_y_grid(w, h, len(anchors), bs)
    widths = [x[0] for x in anchors]
    heights = [x[1] for x in anchors]
    p_w = create_wh(widths, w, h, len(anchors), bs)
    p_h = create_wh(heights, w, h, len(anchors), bs)
    c_x = (F.sigmoid(out[:, :, 0:1, :]) + grid_x) * (model_res/w)
    c_y = (F.sigmoid(out[:, :, 1:2, :]) + grid_y) * (model_res/h)
    ws = torch.exp(out[:, :, 2:3, :]) * p_w
    hs = torch.exp(out[:, :, 3:4, :]) * p_h
    obj = F.sigmoid(out[:, :, 4:5, :])
    class_probs = F.softmax(out[:, :, 5:, :], 2)
    probs, idxs = class_probs.max(2)
    probs = probs.unsqueeze(2)
    idxs = idxs.unsqueeze(2).type_as(out)
    result = torch.cat([c_x, c_y, ws, hs, obj, probs, idxs], 2)
    return result.transpose(1, 2).contiguous().view(1, 7, -1)


def hw_to_corners(x):
    x_start = x[:, :, 0:1] - x[:, :, 2:3]/2
    x_end = x[:, :, 0:1] + x[:, :, 2:3]/2
    y_start = x[:, :, 1:2] - x[:, :, 3:4]/2
    y_end = x[:, :, 1:2] + x[:, :, 3:4]/2
    return torch.cat([x_start, y_start, x_end, y_end], 2)


def iou(bbox1, bbox2):
    x_start = max(bbox1[0], bbox2[0])
    y_start = max(bbox1[1], bbox2[1])
    x_end = min(bbox1[2], bbox2[2])
    y_end = min(bbox1[3], bbox2[3])
    if (x_start >= x_end) or (y_start >= y_end):
        return 0.0
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]
    w_int = x_end - x_start
    h_int = y_end - y_start
    intersection = w_int*h_int
    union = (w1*h1) + (w2*h2) - intersection
    return intersection/union


def nms(boxes, thresh=0.4):
    if boxes.size(0) == 0:
        return boxes
    conf = boxes[:, 4]
    _, idxs = torch.sort(conf, descending=True)
    for tail, i in enumerate(idxs):
        if boxes[i][4] > 0:
            for j in idxs[(tail+1):]:
                if ((iou(boxes[i], boxes[j])) > thresh and
                   (boxes[i][-1] == boxes[j][-1])):
                    boxes[j][4] = 0
    return boxes[boxes[:, 4] > 0]


def add_boxes_to_image(image, boxes, class_list, model_res):
    h = image.shape[0]/model_res
    w = image.shape[1]/model_res
    c = len(COLORS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for b in boxes:
        cls = int(b[-1])
        text_color = tuple(255 - x for x in COLORS[cls % c])
        t_w, t_h = cv2.getTextSize(class_list[cls], font, 0.5, 1)[0]
        start = (int(b[0]*w), int(b[1]*h))
        end = (int(b[2]*w), int(b[3]*h))
        cv2.rectangle(image, start, end, color=(255, 0, 0), thickness=2)
        cv2.rectangle(image, start, (start[0] + t_w, start[1] + t_h),
                      color=COLORS[cls % c], thickness=-1)
        cv2.putText(image, class_list[cls], (start[0], start[1]+t_h-1),
                    font, 0.5, color=text_color, thickness=1)
    return image


def detect(model, image, device,
           obj_threshold=0.7, nms_thresh=0.4, model_res=416):
    model.eval()
    resized = cv2.resize(image, (model_res, model_res))
    tfms = transforms.ToTensor()
    x = tfms(resized).unsqueeze(0).to(device)
    res = model(x)
    anchors = ([(116, 90), (156, 198), (373, 326)],
               [(30, 61),  (62, 45),  (59, 119)],
               [(10, 13), (16, 30), (33, 23)])
    res = [convert_to_boxes(r, ancs, model_res)
           for r, ancs in zip(res, anchors)]
    res = torch.cat(res, 2).transpose(1, 2).contiguous()
    res[:, :, :4] = hw_to_corners(res[:, :, :4])
    boxes = res[res[:, :, 4] > obj_threshold]
    boxes = nms(boxes, nms_thresh).cpu()
    return add_boxes_to_image(image, boxes, CLASS_LIST, model_res)


def save_im(image, image_path):
    p = Path(image_path)
    new_name = p.stem + '_prediction' + '.jpg'
    new_p = Path('predictions') / new_name
    cv2.imwrite(str(new_p), image)
    print(f'saved under {new_p}')
