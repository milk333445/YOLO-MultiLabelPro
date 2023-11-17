import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import os
import pandas as pd
from pathlib import Path
import torch
import copy
import sys
from autolabel_config import ConfigManager
import re
import time
import torchvision

config_manager = ConfigManager()
conf = config_manager.get_config()
obj = conf['obj']
clr = conf['clr']
key_actions = conf['key_actions_normal']

def DrawText(img, 
             text,
             font=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.5,
             font_thickness=1,
             text_color=(0, 0, 255),
             text_color_bg=(255, 255, 255),
             pos='tl',
             axis=(0, 0)
             ):
    axis = (int(axis[0]), int(axis[1]))
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    try:
        if pos == 'bl':
            cv2.rectangle(img, axis, (axis[0]+text_w, axis[1]-text_h*2), text_color_bg, -1)
            cv2.putText(img, text, (axis[0], int(axis[1] - text_h/2)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        elif pos == 'tl':
            cv2.rectangle(img, axis, (axis[0]+text_w, axis[1]+text_h*2), text_color_bg, thickness=-1)
            cv2.putText(img, text, (axis[0], int(axis[1] + text_h*3/2)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        elif pos == 'tr':
            cv2.rectangle(img, axis, (axis[0]-text_w, axis[1]+text_h*2), text_color_bg, thickness=-1)
            cv2.putText(img, text, (axis[0]-text_w, int(axis[1] + text_h*3/2)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        elif pos == 'br':
            cv2.rectangle(img, axis, (axis[0]-text_w, axis[1]-text_h*2), text_color_bg, thickness=-1)
            cv2.putText(img, text, (axis[0]-text_w, int(axis[1] - text_h/2)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    except:
        print('position and axis are wrong setting')
        
def location(x, y, w, h, im_w, im_h, classes):
    # x, y, w, h: Detect the coordinates and width of the upper left corner of the frame.
    # im_w, im_h: Image width, height
    tem_lst = []
    ln_x = round((x+w/2)/im_w, 6) # The ratio of the center point x to the whole width of the graph.
    ln_y = round((y+h/2)/im_h, 6) # The ratio of the center point y to the height of the whole picture.
    ob_w = round(w/im_w, 6) # The ratio of the object width to the whole picture width.
    ob_h = round(h/im_h, 6) # Object Height as a Percentage of the Height of the Picture
    tem_lst.append(classes)
    tem_lst.append(ln_x)
    tem_lst.append(ln_y)
    tem_lst.append(ob_w)
    tem_lst.append(ob_h)
    return tem_lst

def handle_mouse_move_normal(x, y, drawing, param):
    tmp_im = copy.deepcopy(param[1])
    current_category = obj[param[3]]
    info_text = f'current category: {current_category}'
    if drawing:
        if len(param[2]) > 0:
            for i in range(len(param[2])):
                cv2.rectangle(tmp_im, (param[0][i][0], param[0][i][1]), 
                                (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
                DrawText(tmp_im, obj[param[2][i]], font_thickness=2, font_scale=1, pos='tl', axis= (param[0][i][0], param[0][i][1]))
                              
        cv2.rectangle(tmp_im, (param[0][-1][0], param[0][-1][1]), 
                        (x, y), (clr[param[3]]), 2)
        cv2.putText(tmp_im, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        DrawText(tmp_im, obj[param[3]], font_thickness=2, font_scale=1, pos='tl', axis= (param[0][-1][0], param[0][-1][1]))
        cv2.circle(tmp_im, (x, y), 10, (clr[param[3]]), 2)
    else:
        if len(param[2]) > 0:
            for i in range(len(param[2])):
                cv2.rectangle(tmp_im, (param[0][i][0], param[0][i][1]), 
                                (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
                DrawText(tmp_im, obj[param[2][i]], font_thickness=2, font_scale=1, pos='tl', axis= (param[0][i][0], param[0][i][1]))
                
        cv2.line(tmp_im, (x, 0), (x, tmp_im.shape[0]), (0, 0, 0), 1)
        cv2.line(tmp_im, (0, y), (tmp_im.shape[1], y), (0, 0, 0), 1)
        cv2.putText(tmp_im, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
    return tmp_im

def handle_mouse_move(x, y, drawing, param):
    tmp_im = copy.deepcopy(param[1])
    if drawing:
        if len(param[2]) > 0:
            for i in range(len(param[2])):
                cv2.rectangle(tmp_im, (param[0][i][0], param[0][i][1]), 
                              (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)                        
        cv2.rectangle(tmp_im, (param[0][-1][0], param[0][-1][1]), 
                      (x, y), (clr[param[3]]), 2)
        cv2.circle(tmp_im, (x, y), 10, (clr[param[3]]), 2)
    else:
        if len(param[2]) > 0:
            for i in range(len(param[2])):
                cv2.rectangle(tmp_im, (param[0][i][0], param[0][i][1]), 
                              (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
                
        cv2.line(tmp_im, (x, 0), (x, tmp_im.shape[0]), (0, 0, 0), 1)
        cv2.line(tmp_im, (0, y), (tmp_im.shape[1], y), (0, 0, 0), 1)
    return tmp_im

def handle_left_buttom_up(x, y, param, flag=1): # flag=1: normal, flag=2: LPR
    tmp_param = copy.deepcopy(param)
    tmp_param[0][-1].append(x)
    tmp_param[0][-1].append(y)
    
    new_objs = param[2] + [param[3]]
    print('Labeling Completed')
    print('Current Label Count: ', len(tmp_param[0]))
    
    if flag == 2:
    
        if len(tmp_param[0]) < 7:
            reminder_text = f'Reminder: {7-len(tmp_param[0])} more annotations needed.'
            print(f'Reminder: Label count not reached the maximum, you can still label {7-len(tmp_param[0])} more.')
        elif len(tmp_param[0]) == 7:
            reminder_text = 'Reminder: Annotations are full.'
            print('Reminder: Label limit reached.')
        else:
            reminder_text = f'Reminder: Over the limit by {len(tmp_param[0])-7} annotations.'
            print(f'Reminder: Exceeded label limit by {len(tmp_param[0])-7}')
    elif flag == 1:
        reminder_text = ' '
    else:
        print('Flag Error')
    
    return tmp_param[0], new_objs, reminder_text

def handle_delete_buttom_up(param): # flag=1: normal, flag=2: LPR
    if param[0] and param[2]:
        param[0].pop()
        param[2].pop()
    else:
        print("No more items to undo.")
    
    print('Previous Step')
    print('Current Label Count: ', len(param[0]))
    
    if len(param[0]) < 7:
        reminder_text = f'Reminder: {7-len(param[0])} more annotations needed.'
        print(f'Reminder: Label count not reached the maximum, you can still label {7-len(param[0])} more.')
    elif len(param[0]) == 7:
        reminder_text = 'Reminder: Annotations are full.'
        print('Reminder: Label limit reached.')
    else:
        reminder_text = f'Reminder: Over the limit by {len(param[0])-7} annotations.'
        print(f'Reminder: Exceeded label limit by {len(param[0])-7}')
            
    tmp_im = copy.deepcopy(param[1])
    for i in range(len(param[2])):
        cv2.rectangle(tmp_im, (param[0][i][0], param[0][i][1]), (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
    return tmp_im

def handle_delete_buttom_up_normal(param):
    if param[0] and param[2]:
        param[0].pop()
        param[2].pop()
    else:
        print("No more items to undo.")
    
    print('Previous Step')
    print('Current Label Count: ', len(param[0]))
    
    tmp_im = copy.deepcopy(param[1])
    
    for i in range(len(param[2])):
        cv2.rectangle(tmp_im, (param[0][i][0], param[0][i][1]), (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
        DrawText(tmp_im, obj[param[2][i]], font_thickness=2, font_scale=1, pos='tl', axis= (param[0][i][0], param[0][i][1]))
    return tmp_im
        
def initialize_parameters(last_time_num, source):
    source = str(source)
    if last_time_num is None:
        img_count = 0
    else:
        img_count = max(0, last_time_num)
    return source, img_count

def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')
    return device

def load_model(weights, device):
    name = Path(os.getcwd(), 'my_yolov5').as_posix()
    sys.path.insert(0, name)
    from models.experimental import attempt_load
    model = attempt_load(weights, device=device)
    sys.path.remove(name)
    model.eval()
    return model

def get_image_files(source_dir):
    files = sorted(os.listdir(Path(source_dir))) # Get the list of pictures
    img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(img_files)
    return img_files, total_images

def get_image_path_and_name(source, img_files, img_count, save_dir):
    img_path = os.path.join(source, img_files[img_count]) # Image Path
    window_name, _ = os.path.splitext(img_files[img_count])
    sav_img = os.path.join(save_dir, window_name + '.txt')
    return img_path, window_name, sav_img

def read_and_preprocess_image(img_path, imagesz, device):
    im0 = cv2.imread(img_path)
    im_tmp = copy.deepcopy(im0)
    im = letterbox(im0, imagesz)[0]
    im = im.transpose((2, 0, 1))[::-1].copy()
    im = torch.from_numpy(im).float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    im = im.to(device)
    return im, im_tmp

def predict_image(im, model, conf_thres, iou_thres, max_det, im0_shape):
    pred = model(im)
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
    pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0_shape).round()
    return pred

def draw_prediction_on_image(pred, im0, obj, clr):
    img_tmp = copy.deepcopy(im0)
    lst = []
    
    for i in range(len(pred[0])):
        x = int(pred[0][i][0])
        y = int(pred[0][i][1])
        w = int(pred[0][i][2]) - int(pred[0][i][0])
        h = int(pred[0][i][3]) - int(pred[0][i][1])
        classes = int(pred[0][i][5])
        if classes != 0:
            continue
        cv2.rectangle(img_tmp, (int(pred[0][i][0]), int(pred[0][i][1])), (int(pred[0][i][2]), int(pred[0][i][3])), clr[classes], 3, cv2.LINE_AA)
        DrawText(img_tmp, obj[classes], font_thickness=2, font_scale=1, pos='tl', axis=(int(pred[0][i][0]), int(pred[0][i][1])))
        lst_i = location(x, y, w, h, im0.shape[1], im0.shape[0], classes)
        lst.append(lst_i)
    return img_tmp, lst

def get_plate_string():
    while True:
        plate_string = input('Please enter the license plate (3 letters + 4 numbers): ').upper()

        if re.match(r'^[A-Z]{3}\d{4}$', plate_string):
            return [obj.index(char) for char in plate_string]
        else:
            print('Incorrect format. Please re-enter.')
                  
def process_image_annotations(lst_a):
    im0 = lst_a[1]
    lst = []
    for i in range(len(lst_a[0])):
        x0, x1 = sorted([int(lst_a[0][i][0]), int(lst_a[0][i][2])])
        y0, y1 = sorted([int(lst_a[0][i][1]), int(lst_a[0][i][3])])
        w, h = x1 - x0, y1 - y0
        classes = lst_a[2][i]
        lst_i = location(x0, y0, w, h, im0.shape[1], im0.shape[0], classes)
        lst.append(lst_i)
    return lst

def save_labels_to_file(lst, save_img):
    with open(save_img, 'w+') as f:
        for entry in lst:
            f.write(" ".join(map(str, entry)) + "\n")
    print('Label Saved Successfully')
           
def get_key_action(key):
    return key_actions.get(key, 'invalid key')

def update_label_and_display(action, count, obj, lst_a, window_name):
    try:
        if action =='switch_next':
            count += 1
        elif action == 'switch_prev':
            count -= 1        
        item = count % len(obj)
        lst_a[3] = item
        im_new = copy.deepcopy(lst_a[1])
        for i in range(len(lst_a[0])):
            cv2.rectangle(im_new, (lst_a[0][i][0], lst_a[0][i][1]), (lst_a[0][i][2], lst_a[0][i][3]), (clr[lst_a[2][i]]), 2)
            DrawText(im_new, obj[lst_a[2][i]], font_thickness=2, font_scale=1, pos='tl', axis=(lst_a[0][i][0], lst_a[0][i][1]))

        current_category = obj[item]
        info_text = f'current category: {current_category}'
        cv2.putText(im_new, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, im_new)
    except IndexError:
        print('Index Error')
        pass
    return count

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def resize_to_fit_screen(image, screen_width=1920, screen_height=1080):
    img_tmp = copy.deepcopy(image)
    h, w = img_tmp.shape[:2]
    scale_w = screen_width / w
    scale_h = screen_height / h
    scale = min(scale_w, scale_h)
    scale = scale * 0.9
    
    if scale < 1:
        # Resizing the image if it's larger than the screen size
        resized_image = cv2.resize(img_tmp, (int(w * scale), int(h * scale)))
        resize_scale = scale
    else:
        # If the image is smaller than the screen size, we don't resize
        resized_image = img_tmp
        resize_scale = 1
    h, w = resized_image.shape[:2]
    
    return resized_image, resize_scale