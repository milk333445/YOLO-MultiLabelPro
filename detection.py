import os
import cv2
import time
import numpy as np
import torch
import torchvision
import sys
from pathlib import Path
def DIoU(bboxes1, bboxes2, w, h):
    """
    Parameters
    ----------
    bboxes1 : numpy.ndarray (n, 4)
    bboxes2 : numpy.ndarray (m, 4)

    Returns
    -------
    dious : numpy.ndarray (n, m)
    """
    if bboxes1.ndim == 1:
        bboxes1 = np.expand_dims(bboxes1, 0)
    if bboxes2.ndim == 1:
        bboxes2 = np.expand_dims(bboxes2, 0)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = np.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    
    # xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2:2+1] - bboxes1[:, 0:0+1]  # r
    h1 = bboxes1[:, 3:3+1] - bboxes1[:, 1:1+1]  # r
    w2 = bboxes2[:, 2:2+1] - bboxes2[:, 0:0+1]  # c
    h2 = bboxes2[:, 3:3+1] - bboxes2[:, 1:1+1]  # c
    
    area1 = w1 * h1  # r
    area2 = w2 * h2  # c

    center_x1 = (bboxes1[:, 2:2+1] + bboxes1[:, 0:0+1]) / 2  # r
    center_y1 = (bboxes1[:, 3:3+1] + bboxes1[:, 1:1+1]) / 2  # r
    center_x2 = (bboxes2[:, 2:2+1] + bboxes2[:, 0:0+1]) / 2  # c
    center_y2 = (bboxes2[:, 3:3+1] + bboxes2[:, 1:1+1]) / 2  # c
    # (r, c)
    center_dx = center_x1 - center_x2.T
    center_dy = center_y1 - center_y2.T
    
    # (r, c)
    inter_min_x = np.maximum(bboxes1[:, 0:0+1], bboxes2[:, 0:0+1].T)  
    inter_min_y = np.maximum(bboxes1[:, 1:1+1], bboxes2[:, 1:1+1].T) 
    inter_max_x = np.minimum(bboxes1[:, 2:2+1], bboxes2[:, 2:2+1].T)
    inter_max_y = np.minimum(bboxes1[:, 3:3+1], bboxes2[:, 3:3+1].T)
    outer_min_x = np.minimum(bboxes1[:, 0:0+1], bboxes2[:, 0:0+1].T)
    outer_min_y = np.minimum(bboxes1[:, 1:1+1], bboxes2[:, 1:1+1].T)
    outer_max_x = np.maximum(bboxes1[:, 2:2+1], bboxes2[:, 2:2+1].T)
    outer_max_y = np.maximum(bboxes1[:, 3:3+1], bboxes2[:, 3:3+1].T)
    
    # (r, c)
    inter_x = np.clip((inter_max_x - inter_min_x), a_min=0, a_max=None)
    inter_y = np.clip((inter_max_y - inter_min_y), a_min=0, a_max=None)
    inter_area = inter_x * inter_y
    outer_x = np.clip((outer_max_x - outer_min_x), a_min=0, a_max=None)
    outer_y = np.clip((outer_max_y - outer_min_y), a_min=0, a_max=None)
    inter_diag = center_dx ** 2 + center_dy ** 2
    outer_diag = outer_x ** 2 + outer_y ** 2
    # outer_diag = (w)**2 + (h)**2
    union_area = area1 + area2.T - inter_area
    dious = inter_area / union_area - inter_diag / outer_diag
    dious = np.clip(dious, a_min=-1.0, a_max=1.0)
    # print(inter_area / union_area ,  inter_diag / outer_diag)
    return dious

# =============================================================================
# Non-maximum-suppression
# =============================================================================
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


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
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)

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

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

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
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
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
            # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


# =============================================================================
# Yolo detection tool
# =============================================================================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
    
    
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

"""
class Detector():
    def __init__(self, 
                 model, 
                 size,  # int or tuple(int, int)
                 device,  # 'cpu' or 'cuda'
                 conf_thres, 
                 classes=None):
        self.model = model.float().eval()
        self.size = size
        self.device = device
        self.conf_thres = conf_thres
        self.classes = classes
        
    def run(self, imgs):
        # preprocess
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
        
        pre_imgs = []
        for img in imgs:
            pre_img = letterbox(img, self.size)[0]
            pre_img = pre_img.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
            pre_img = np.ascontiguousarray(pre_img)
            pre_img = torch.from_numpy(pre_img).float().to(self.device)
            pre_img /= 255
            pre_imgs.append(pre_img)
        pre_imgs = torch.stack(pre_imgs)
        
        with torch.no_grad():
            # detect
            pred = self.model(pre_imgs)[0]
            
            # postprocess
            pred = non_max_suppression(pred, 
                                       conf_thres=self.conf_thres, 
                                       iou_thres=0.45, 
                                       classes=self.classes, 
                                       agnostic=False, 
                                       max_det=1000)
            dets = []
            for i, det in enumerate(pred):
                det[:, :4] = scale_coords(pre_imgs.shape[2:], det[:, :4], imgs[i].shape).round() 
                
                dets.append(det)
            return dets
"""

class Detector():
    def __init__(
            self,
            model,
            size,
            device,
            conf_thres,
            classes=None,
            mode='normal'
    ):
        self.mode = mode
        self.size = size
        self.device = device
        self.conf_thres = conf_thres
        self.classes = classes
        self.model = model.float().eval()

    def run(self, imgs):
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        pre_imgs = []

        for img in imgs:
            pre_img = letterbox(img, self.size, auto=False)[0]
            pre_img = pre_img.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
            pre_img = np.ascontiguousarray(pre_img)
            pre_img = torch.from_numpy(pre_img).float().to(self.device)
            pre_img /= 255
            pre_imgs.append(pre_img)
        pre_imgs = torch.stack(pre_imgs)
            
        with torch.no_grad():
            pred = self.model(pre_imgs)[0]

            if self.mode == 'normal':
                pred = non_max_suppression(pred, 
                                       conf_thres=self.conf_thres, 
                                       iou_thres=0.45, 
                                       classes=self.classes, 
                                       agnostic=False, 
                                       max_det=1000)
                dets = []
                for i, det in enumerate(pred):
                    det[:, :4] = scale_coords(pre_imgs.shape[2:], det[:, :4], imgs[i].shape).round() 
                    dets.append(det)
                return dets
            elif self.mode == 'obb':
                name = Path(os.getcwd(), 'my_yolov5_obb').as_posix()
                sys.path.insert(0, name)
                from utils.general import non_max_suppression_obb
                pred = non_max_suppression_obb(pred, self.conf_thres, 0.45)
                sys.path.remove(name)
                return pred, pre_imgs

if __name__ == '__main__':
    os.chdir('../')
    import time
    from utils_va.vision_tools import DrawBBOX
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modelpath = 'weights/clean_clothes.pt'
    
    os.chdir('./my_yolov5')
    from models.experimental import attempt_load
    os.chdir('../')
    model = attempt_load(modelpath, device=device)
    
    model.float()
    class_names = model.module.names if hasattr(model, 'module') else model.names
    detector = Detector(model, size=640, device=device, conf=0.25, classes=None)
    
    path = 'C:/Users/William/Desktop/images'
    listdir = os.listdir(path)
    for i in range(len(listdir)):
        img = cv2.imread(os.path.join(path, listdir[i]))
        t = time.time()
        results = detector.run(img)
        print(time.time() - t)
        
        DrawBBOX(img, results[0], class_names, [0], color=(255, 0, 0))
        
        cv2.imshow('test', img)
        key = cv2.waitKey(1)
        if key == 27: 
            break
    cv2.destroyAllWindows()