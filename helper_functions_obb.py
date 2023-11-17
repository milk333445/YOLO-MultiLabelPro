import cv2
import numpy as np
import copy
from pathlib import Path
import argparse
import os
import sys
from autolabel_config import ConfigManager
import torch
import math
from helper_functions_normal import letterbox
from detection import Detector

config_manager = ConfigManager()
conf = config_manager.get_config()
obj = conf['obj']
clr = conf['clr']
key_actions = conf['key_actions_obb']

def get_key_action_obb(key):
    return key_actions.get(key, 'invalid key')

def draw_info(image, info_text):
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return image

def draw_existing_annotations(image, annotations, categories):
    for i, pts in enumerate(annotations):
        cv2.polylines(image, [np.array(pts, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
        label = obj[categories[i]]
        label_position = (int(pts[0][0]), int(pts[0][1]) - 5)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image
        
def calculate_rotated_rectangle(center_point, distance, angle_rad):
    rect_w = int(distance * 2)
    rect_h = int(distance)
    pts = np.array([
        [-rect_w//2, -rect_h//2],
        [rect_w//2, -rect_h//2],
        [rect_w//2, rect_h//2],
        [-rect_w//2, rect_h//2]
    ])
    rotated_pts = [(pt[0] * np.cos(angle_rad) - pt[1] * np.sin(angle_rad) + center_point[0],
                    pt[0] * np.sin(angle_rad) + pt[1] * np.cos(angle_rad) + center_point[1]) for pt in pts]
    return np.array(rotated_pts, dtype=np.int32)

def calaulate_movement(prev_mouse_move, current_mouse_move):
    dx = current_mouse_move[0] - prev_mouse_move[0]
    dy = current_mouse_move[1] - prev_mouse_move[1]
    return dx, dy

def move_edge(points, edge_index1, edge_index2, offset):
    # Calculate the center of the polygon.
    centroid_x = sum([pt[0] for pt in points]) / len(points)
    centroid_y = sum([pt[1] for pt in points]) / len(points)
    # Calculate the midpoint of the selected side
    midpoint_x = (points[edge_index1][0] + points[edge_index2][0]) / 2
    midpoint_y = (points[edge_index1][1] + points[edge_index2][1]) / 2
    # Calculate the vector from the center to the midpoint of the selected edge.
    direction_dx = midpoint_x - centroid_x
    direction_dy = midpoint_y - centroid_y
    # Normalize the vector
    magnitude = np.sqrt(direction_dx**2 + direction_dy**2)
    direction_dx /= magnitude
    direction_dy /= magnitude
    # Use this vector and offset for movement.
    points[edge_index1][0] += offset * direction_dx
    points[edge_index1][1] += offset * direction_dy
    points[edge_index2][0] += offset * direction_dx
    points[edge_index2][1] += offset * direction_dy

def draw_selected_edge(lst_a_full, selected_edge):
    points = lst_a_full[0][-1]
    img = lst_a_full[1].copy()
    
    if len(lst_a_full[0]) > 0:
        for i, pt in enumerate(lst_a_full[0]):
            cv2.polylines(img, [np.array(lst_a_full[0][i], dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
            label = obj[lst_a_full[5][i]]
            cv2.putText(img, label, (int(pt[0][0]), int(pt[0][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array(points, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
    # Drawing Selected Edges
    if selected_edge == 'top':
        cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), color=(0, 0, 255), thickness=2)  
    elif selected_edge == 'bottom':
        cv2.line(img, (int(points[2][0]), int(points[2][1])), (int(points[3][0]), int(points[3][1])), color=(0, 0, 255), thickness=2)
    elif selected_edge == 'left':
        cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[3][0]), int(points[3][1])), color=(0, 0, 255), thickness=2)
    elif selected_edge == 'right':
        cv2.line(img, (int(points[1][0]), int(points[1][1])), (int(points[2][0]), int(points[2][1])), color=(0, 0, 255), thickness=2)    
    return img
    
def update_image_and_label(lst_a, obj, count2):
    item = count2 % len(obj)
    current_category = obj[item]
    lst_a[4] = item
    img_tmp = lst_a[1].copy()
    # Add a reminder of the current category
    info_text = f'current category: {current_category}'
    cv2.putText(img_tmp, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow(lst_a[2], img_tmp)

def initialize_parameters(last_time_num, source):
    source = str(source)
    if last_time_num is None:
        img_count = 0
    else:
        img_count = max(0, last_time_num)
    return source, img_count

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
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

def poly_label(poly, label, conf, color, img, lst, line_thickness=3, txt_color=(255, 255, 255)):
    
    lst.append([poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label])
    
    if isinstance(poly, torch.Tensor):
        poly = poly.cpu().numpy()
    if isinstance(poly[0], torch.Tensor):
        poly = [x.cpu().numpy() for x in poly]
        
    polygon_list = np.array([
        (poly[0], poly[1]),
        (poly[2], poly[3]),
        (poly[4], poly[5]),
        (poly[6], poly[7]),
    ], np.int32)
    cv2.drawContours(img, contours=[polygon_list], contourIdx=-1, color=color, thickness=line_thickness * 2)
    
    label_text = f"{label} {conf:.2f}"
    
    if label_text:
        tf = max(line_thickness - 1, 1)  # font thickness
        xmax, xmin, ymax, ymin = max(poly[0::2]), min(poly[0::2]), max(poly[1::2]), min(poly[1::2])
        x_label, y_label = int((xmax + xmin) / 2), int((ymax + ymin) / 2)
        w, h = cv2.getTextSize(label_text, 0, fontScale=line_thickness, thickness=tf)[0]
        cv2.rectangle(
            img,
            (x_label, y_label),
            (x_label + w + 1, y_label + int(1.5 * h)),
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(img, label_text, (x_label, y_label + h), 0, line_thickness, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    return img, lst
      
def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    if isinstance(obboxes, torch.Tensor):
        center, w, h, theta = obboxes[:, :2], obboxes[:, 2:3], obboxes[:, 3:4], obboxes[:, 4:5]
        Cos, Sin = torch.cos(theta), torch.sin(theta)

        vector1 = torch.cat(
            (w/2 * Cos, -w/2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h/2 * Sin, -h/2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)
        
def select_device(device='', batch_size=0, newline=True):
    cuda = False
    device = str(device).strip().lower().replace('cuda:', '')
    
    if device and device != 'cpu':
        cuda = torch.cuda.is_available()
        if cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = device
        else:
            raise AssertionError(f'CUDA unavailable, invalid device {device} requested')
    
    if cuda:
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
    
    return torch.device('cuda:0' if cuda else 'cpu')
        
def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor
     
def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def scale_polys(img1_shape, polys, img0_shape, ratio_pad=None):
    # ratio_pad: [(h_raw, w_raw), (hw_ratios, wh_paddings)]
    # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0] # h_ratios
        pad = ratio_pad[1] # wh_paddings

    polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    polys[:, :8] /= gain # Rescale poly shape to img0_shape
    #clip_polys(polys, img0_shape)
    return polys

def process_image_with_mdoel(model, device, img_path, imagesz, conf_thres, iou_thres, screen_width=1920, screen_height=1080):
    img0 = cv2.imread(img_path)
    assert img0 is not None, 'Image Not Found ' + img_path
    h, w = img0.shape[:2]
    scale_w = screen_width / w
    scale_h = screen_height / h
    scale = min(scale_w, scale_h)
    if scale < 1:
        resize_scale = scale
    else:
        resize_scale = 1
    img =  letterbox(img0, imagesz, stride=int(model.stride.max()), auto='auto')[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if len(img.shape) == 3:
        img = img[None]
    pred = model(img)[0]
    name = Path(os.getcwd(), 'my_yolov5_obb').as_posix()
    sys.path.insert(0, name)
    from utils.general import non_max_suppression_obb
    pred = non_max_suppression_obb(pred, conf_thres, iou_thres)
    sys.path.remove(name)
    return pred, img, img0, resize_scale, h, w

def visualize_results(model, pred, img, img0, window_name, resize_scale, line_thickness):
    tmp_img = copy.deepcopy(img0)
    colors = Colors()
    lst = []
    for i, det in enumerate(pred):
        if len(det):
            pred_poly = rbox2poly(det[:, :5])
            pred_poly = scale_polys(img.shape[2:], pred_poly, img0.shape)
            det = torch.cat((pred_poly, det[:, -2:]), dim=1)

            for *poly, conf, cls in det:
                label = f"{model.module.names[int(cls)] if hasattr(model, 'module') else model.names[int(cls)]}"
                tmp_img, lst = poly_label(
                    poly,
                    label,
                    conf=conf,
                    color=colors(int(cls)),
                    img=tmp_img,
                    lst=lst,
                    line_thickness=line_thickness
                )
    tmp_img_all = cv2.resize(tmp_img, (int(tmp_img.shape[1] * resize_scale), int(tmp_img.shape[0] * resize_scale)))
    cv2.imshow(window_name, tmp_img_all)
    return tmp_img_all, lst
