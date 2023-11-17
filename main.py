import os
import pandas as pd
from pathlib import Path
import torch
import copy
import cv2
import numpy as np
import sys
from helper_functions_normal import *
import argparse
import yaml
from autolabel_config import *
import re
from helper_functions_obb import *
from detection import Detector

def show_xy_rotated_rect(event, x, y, flags, param):
    global drawing, center_point, rotated_pts, prev_mouse_move
    
    temp_image = copy.deepcopy(param[1])
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            drawing = True
            center_point = (x, y)
            # Drawing centers
            cv2.circle(temp_image, center_point, 5, (0, 0, 255), -1)
            current_category = obj[param[4]]
            cv2.imshow(param[2], temp_image)
        else:
            drawing = False
            cv2.polylines(param[1], [rotated_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.circle(param[1], center_point, 5, (0, 0, 255), -1)
            current_category = obj[param[4]]
            left_top_x = min([pt[0] for pt in rotated_pts])
            left_top_y = min([pt[1] for pt in rotated_pts])
            cv2.putText(param[1], current_category, (left_top_x, left_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            new_obj = param[5] + [param[4]]
            param[0].append(rotated_pts.tolist())
            param[3].append(center_point)
            param[5] = new_obj
            print('labels', param[5])
            cv2.imshow(param[2], param[1])
            print('save successfully')
            print('Labeling Objects...')
    
    elif event == cv2.EVENT_RBUTTONUP:
        prev_mouse_move = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:       
            dx = x - center_point[0]
            dy = y - center_point[1]
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))
            rotated_pts = calculate_rotated_rectangle(center_point, distance, np.radians(angle))
            # Drawing rotated rectangles and centers
            cv2.polylines(temp_image, [rotated_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            left_top_x = min([pt[0] for pt in rotated_pts])
            left_top_y = min([pt[1] for pt in rotated_pts])
            category = obj[param[4]]   
            cv2.putText(temp_image, category, (left_top_x, left_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(temp_image, center_point, 5, (0, 0, 255), -1)
            current_category = obj[param[4]]
            info_text = f'current category: {current_category}'
            cv2.putText(temp_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow(param[2], temp_image)
            
            if flags & cv2.EVENT_FLAG_RBUTTON:
                if not prev_mouse_move:
                    prev_mouse_move = (x, y)
                dx = x - prev_mouse_move[0]
                dy = y - prev_mouse_move[1]
                center_point = (center_point[0] + dx, center_point[1] + dy)
                # Update all points
                for i, pt in enumerate(rotated_pts):
                    rotated_pts[i] = (pt[0] + dx, pt[1] + dy)   
                prev_mouse_move = (x, y)   
                cv2.polylines(temp_image, [rotated_pts], isClosed=True, color=(255, 0, 0), thickness=2)
                left_top_x = min([pt[0] for pt in rotated_pts])
                left_top_y = min([pt[1] for pt in rotated_pts])
                category = obj[param[4]]
                cv2.putText(temp_image, category, (left_top_x, left_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                current_category = obj[param[4]]
                info_text = f'current category: {current_category}'
                cv2.putText(temp_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(temp_image, center_point, 5, (0, 0, 255), -1)
                cv2.imshow(param[2], temp_image)          

def show_xy_combined(event, x, y, flags, param):
    tmp_im = copy.deepcopy(param[1])
    mode = param[5]  # normal or not normal

    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        param[0].append([x1, y1])
        print('Start Labeling')
        print('Labeling Objects...')

    elif event == cv2.EVENT_MOUSEMOVE:
        is_drawing = flags & cv2.EVENT_FLAG_LBUTTON
        if mode == 'normal':
            tmp_im = handle_mouse_move_normal(x, y, is_drawing, param)
        else:
            tmp_im = handle_mouse_move(x, y, is_drawing, param)

        if is_drawing:
            x1 = param[0][-1][0]
            y1 = param[0][-1][1]
            cv2.line(tmp_im, (x1, 0), (x1, tmp_im.shape[0]), (0, 0, 0), 1)
            cv2.line(tmp_im, (x, 0), (x, tmp_im.shape[0]), (0, 0, 0), 1)
            cv2.line(tmp_im, (0, y1), (tmp_im.shape[1], y1), (0, 0, 0), 1)
            cv2.line(tmp_im, (0, y), (tmp_im.shape[1], y), (0, 0, 0), 1)
        cv2.imshow(param[4], tmp_im)

    elif event == cv2.EVENT_LBUTTONUP:
        if mode == 'normal':
            new_coords, new_objs, _ = handle_left_buttom_up(x, y, param)
        else:
            new_coords, new_objs, _ = handle_left_buttom_up(x, y, param, flag=2)
        param[0] = new_coords
        param[2] = new_objs
        if mode == 'normal':
            if len(new_coords) > 0:
                
                cv2.rectangle(param[1], (new_coords[-1][0], new_coords[-1][1]),
                              (new_coords[-1][2], new_coords[-1][3]), (clr[new_objs[-1]]), 2)
                DrawText(param[1], obj[new_objs[-1]], font_thickness=2, font_scale=1, pos='tl',
                         axis=(new_coords[-1][0], new_coords[-1][1]))
        else:
            
            cv2.rectangle(param[1], (new_coords[-1][0], new_coords[-1][1]),
                            (new_coords[-1][2], new_coords[-1][3]), (clr[new_objs[-1]]), 2)
        cv2.imshow(param[4], tmp_im)
    
class Statemachine_ATL:
    def __init__(self, last_time_num=None, obj=None, clr=None, source='./data/images/',
                 weights='./runs/train/exp3/weights/best.pt', imagesz=(640, 640), conf_thres=0.25, iou_thres=0.45,
                 max_det=1000, store='./data/labels/'):
        self.state = 'initial'
        self.img_count = 0
        self.img_files, self.total_images, self.lst, self.img_tmp = [], 0, None, None
        self.last_time_num, self.source, self.weights = last_time_num, source, weights
        self.imagesz, self.conf_thres, self.iou_thres, self.max_det = imagesz, conf_thres, iou_thres, max_det
        self.store, self.model, self.obj, self.clr = store, None, obj, clr
        self.window_name, self.save_img, self.detector = None, None, None
        self.is_lpr, self.is_obbs = False, False
        self.device = '0'
        self.resize_scale = 1

    def load_parameters(self):
        self.source, self.img_count = initialize_parameters(self.last_time_num, self.source)
        self.device = get_device()
        self.model = load_model(self.weights, self.device)
        self.img_files, self.total_images = get_image_files(self.source)
        self.detector = Detector(self.model, self.imagesz[0], self.device, self.conf_thres)

    def load_parameters_obb(self):
        self.source, self.img_count = initialize_parameters(self.last_time_num, self.source)
        self.img_files, self.total_images = get_image_files(self.source)
        self.device = select_device(self.device)
        w = self.weights if isinstance(self.weights, list) else self.weights
        name = Path(os.getcwd(), 'my_yolov5_obb').as_posix()
        sys.path.insert(0, name)
        from models.experimental import attempt_load
        self.model = attempt_load(w, map_location=self.device)
        sys.path.remove(name)
        stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # module時是True，否則是False，hasattr意思是判斷model有沒有module這個屬性
        self.model.to(self.device).eval()
      
    def run_model_multilabel(self):
        self.load_parameters()
        while self.img_count < self.total_images:
            self.load_next_image()
        cv2.destroyAllWindows()  
    
    def run_without_model(self):
        self.source, self.img_count = initialize_parameters(self.last_time_num, self.source)
        self.img_files, self.total_images = get_image_files(self.source)
        while self.img_count < self.total_images:
            self.load_image_without_model()
        cv2.destroyAllWindows()
    
    def run_obb_model_label(self):
        self.is_obbs = True 
        self.load_parameters_obb()
        while self.img_count < self.total_images:
            self.load_image_obb()
        cv2.destroyAllWindows()
        
    def run_without_model_obb_label(self):
        self.source, self.img_count = initialize_parameters(self.last_time_num, self.source)
        self.img_files, self.total_images = get_image_files(self.source)
        while self.img_count < self.total_images:
            self.load_image_without_model_obb_label()
        cv2.destroyAllWindows()
    
    def run_model_LPR_label(self):
        self.is_lpr = True
        self.load_parameters()
        while self.img_count < self.total_images:
            self.load_image_LPR_label()
        cv2.destroyAllWindows()
    
    def run_without_model_LPR_label(self):
        self.source, self.img_count = initialize_parameters(self.last_time_num, self.source)
        self.img_files, self.total_images = get_image_files(self.source)
        while self.img_count < self.total_images:
            self.load_image_without_model_LPR_label()
        cv2.destroyAllWindows()
    
    def load_image_obb(self):
        image_path, self.window_name, self.save_img = get_image_path_and_name(self.source, self.img_files, self.img_count, self.store)
        print('Current Image Path: ', image_path)
        pred, img, img0, self.resize_scale, h, w = process_image_with_mdoel(self.model, self.device, image_path, self.imagesz, self.conf_thres, self.iou_thres)
        self.img_tmp, self.lst = visualize_results(self.model, pred, img, img0, self.window_name, self.resize_scale)
        action = get_key_action_obb(cv2.waitKey(0))
        self.process_key_input(action)
        
    def modify_images_obb(self):
        self.img_tmp, self.resize_scale = resize_to_fit_screen(self.img_tmp)
        lst_obj = []
        im0 = copy.deepcopy(self.img_tmp)
        lst_a = [[], self.img_tmp, self.window_name, [], 0, lst_obj]
        cv2.imshow(self.window_name, self.img_tmp)
        cv2.setMouseCallback(self.window_name, show_xy_rotated_rect, lst_a)
        count = 0
        count2 = 0
        selected_edge = None
        edges = ['top', 'bottom', 'left', 'right']
        offset = 5
        while True:
            action = get_key_action_obb(cv2.waitKey(0))
            if action == 'save':
                with open(self.save_img, 'w+') as file:
                    for i, coords in enumerate(lst_a[0]):
                        scaled_coords = [str(int(float(item) / self.resize_scale)) for sublist in coords for item in sublist]
                        category = self.obj[lst_a[5][i]]
                        line = ' '.join(scaled_coords) + ' ' + category + ' 0'
                        file.write(line + '\n')
                print('save successfully')
                self.img_count += 1
                cv2.destroyAllWindows()
                break
            elif action == 'pass':
                self.img_count += 1
                cv2.destroyAllWindows()
                break
            elif action == 'exit':
                cv2.destroyAllWindows()
                print('Program Terminated')
                quit()
            elif action == 'previous':
                self.img_count -= 1
                self.img_count = max(0, self.img_count)
                cv2.destroyAllWindows()
                break
            elif action == 'delete':
                if lst_a[0]:
                    lst_a[0].pop()
                    lst_a[3].pop()
                    lst_a[5].pop()
                    print('delete successfully')
                    self.img_tmp = copy.deepcopy(im0)
                    lst_a[1] = self.img_tmp
                    if len(lst_a[0]) > 0:
                        self.img_tmp = draw_existing_annotations(self.img_tmp, lst_a[0], lst_a[5])
                    cv2.imshow(lst_a[2], self.img_tmp)
                else:
                    print('delete failed')
            elif action == 'switch_next_side':
                current_index = count % len(edges)
                selected_edge = edges[current_index]
                count += 1
            elif action == 'switch_next':
                count2 += 1
                update_image_and_label(lst_a, self.obj, count2)
            elif action == 'switch_prev':
                count2 -= 1
                update_image_and_label(lst_a, self.obj, count2)
            elif action not in key_actions_obb.values():
                print('Input error. Please re-enter.')
                continue
                
            if not selected_edge or not lst_a[0]:
                continue
            if selected_edge == 'top':
                if action == 'plus':
                    move_edge(lst_a[0][-1], 0, 1, offset)
                elif action == 'minus':
                    move_edge(lst_a[0][-1], 0, 1, -offset)
            elif selected_edge == 'bottom':
                if action == 'plus':
                    move_edge(lst_a[0][-1], 2, 3, offset)
                elif action == 'minus':
                    move_edge(lst_a[0][-1], 2, 3, -offset)
            elif selected_edge == 'left':
                if action == 'plus':
                    move_edge(lst_a[0][-1], 0, 3, offset)
                elif action == 'minus':
                    move_edge(lst_a[0][-1], 0, 3, -offset)
            elif selected_edge == 'right':
                if action == 'plus':
                    move_edge(lst_a[0][-1], 1, 2, offset)
                elif action == 'minus':
                    move_edge(lst_a[0][-1], 1, 2, -offset)
            if selected_edge:
                # get the original image
                self.img_tmp = copy.deepcopy(im0)
                lst_a[1] = self.img_tmp
                self.img_tmp = draw_selected_edge(lst_a, selected_edge)
            cv2.imshow(lst_a[2], self.img_tmp)
                
    def process_key_input(self, action):
        if action == 'exit':
            self.exit_program()
        elif self.state == 'initial':
            if action == 'save':
                if self.is_obbs:
                    self.save_labels_obb(self.lst, self.save_img)
                else:
                    self.save_labels(self.lst, self.save_img)
                self.load_next_image()
            elif action == 'modify':
                if self.is_lpr:
                    self.process_LPR_label()
                elif self.is_obbs:
                    self.modify_images_obb()
                else:
                    self.modify_images() 
                self.load_next_image()
            elif action == 'pass':
                cv2.destroyAllWindows()
                self.load_next_image()
            elif action == 'previous':
                cv2.destroyAllWindows()
                self.load_previous_image()
    
    def save_labels(self, lst, save_img):
        with open(save_img, 'w+') as f:
            for entry in lst:
                f.write(" ".join(map(str, entry)) + "\n")
        cv2.destroyAllWindows()
        print('Label Saved Successfully')

    def save_labels_obb(self, lst, save_img):
        with open(save_img, 'w+') as f:
            for item in lst:
                poly_coords = item[:8]
                label = item[8]
                coords = [str(int(x)) for x in poly_coords]
                line = ' '.join(coords) + ' ' + label + ' 0'
                f.write(line + '\n')
        print(f'annotations saved to {save_img}')

    def modify_images(self):
        lst_obj = []
        im0 = copy.deepcopy(self.img_tmp)
        lst_a = [[], self.img_tmp, lst_obj, 0, self.window_name, 'normal']
        cv2.imshow(self.window_name, self.img_tmp)
        cv2.setMouseCallback(self.window_name, show_xy_combined, lst_a)
        count = 0
        while True:
            key = cv2.waitKey(0)
            action = get_key_action(key)
            if action == 'save':
                lst = process_image_annotations(lst_a)
                self.save_labels(lst, self.save_img)
                self.img_count += 1
                cv2.destroyAllWindows()
                break
            elif action in ['switch_next', 'switch_prev']:
                lst_a[1] = copy.deepcopy(im0)
                count = update_label_and_display(action, count, self.obj, lst_a, self.window_name)
            elif action == 'pass':
                self.img_count += 1
                cv2.destroyAllWindows()
                break
            elif action == 'exit':
                cv2.destroyAllWindows()
                print('Program Terminated')
                quit()
            elif action == 'previous':
                self.img_count = max(0, self.img_count - 1)
                cv2.destroyAllWindows()
                break
            elif action == 'delete':
                lst_a[1] = copy.deepcopy(im0)
                self.img_tmp = handle_delete_buttom_up_normal(lst_a)
                cv2.imshow(self.window_name, self.img_tmp)
            else:
                print('Input error. Please re-enter.')

    def handle_action_in_modify_images(self, action, lst_a, save_img):
        if action == 'save':
            lst = process_image_annotations(lst_a)
            save_labels_to_file(lst, save_img)
            self.img_count += 1
            cv2.destroyAllWindows()
        elif action == 'pass':
            self.img_count += 1
            cv2.destroyAllWindows()
        elif action == 'exit':
            self.exit_program()
        elif action == 'previous':
            self.img_count = max(0, self.img_count - 1)
            cv2.destroyAllWindows()

    def load_image_common(self, increment):
        self.img_count = max(0, self.img_count + increment)
        print('Current image count:', self.img_count + 1)
        img_path, self.window_name, self.save_img = get_image_path_and_name(
            self.source, self.img_files, self.img_count, self.store)
        print('Current Image Path: ', img_path)
        cv2.destroyAllWindows()
        im0 = cv2.imread(img_path)
        if self.is_obbs:
            pred, img, img0, self.resize_scale, h, w = process_image_with_mdoel(self.model, self.device, image_path, self.imagesz, self.conf_thres, self.iou_thres)
            self.img_tmp, self.lst = visualize_results(self.model, pred, img, img0, self.window_name, self.resize_scale)
            action = get_key_action_obb(cv2.waitKey(0))
        else:
            pred = self.detector.run(im0)
            self.img_tmp, self.lst = draw_prediction_on_image(pred, im0, self.obj, self.clr)
            self.img_tmp, self.resize_scale = resize_to_fit_screen(self.img_tmp)
            cv2.imshow(self.window_name, self.img_tmp)
            action = get_key_action(cv2.waitKey(0))
        self.process_key_input(action)

    def load_next_image(self):
        self.load_image_common(increment=1)

    def load_previous_image(self):
        self.load_image_common(-1)

    def exit_program(self):
        cv2.destroyAllWindows()
        print('Program Terminated')
        quit()

    def load_image_without_model(self):
        print('Current image count:', self.img_count + 1)
        img_path, self.window_name, self.save_img = get_image_path_and_name(self.source, self.img_files, self.img_count, self.store)
        print('Current Image Path: ', img_path)
        im0 = cv2.imread(img_path)
        self.img_tmp, resize_scale = resize_to_fit_screen(im0)
        self.modify_images()    
        
    def load_image_LPR_label(self):
        print('Current image count:', self.img_count + 1)
        img_path, self.window_name, self.save_img = get_image_path_and_name(self.source, self.img_files, self.img_count, self.store)
        print('Current Image Path: ', img_path)
        im0 = cv2.imread(img_path)
        pred = self.detector.run(im0)
        self.img_tmp, self.lst = draw_prediction_on_image(pred, im0, self.obj, self.clr)
        self.img_tmp, resize_scale = resize_to_fit_screen(self.img_tmp)
        cv2.imshow(self.window_name, self.img_tmp)
        action = get_key_action(cv2.waitKey(0))
        self.process_key_input(action)
              
    def process_LPR_label(self):
        im0 = copy.deepcopy(self.img_tmp)
        lst_obj = []
        lst_a = [[], self.img_tmp, lst_obj, 0, self.window_name, 'LPR']
        cv2.imshow(self.window_name, self.img_tmp)
        cv2.setMouseCallback(self.window_name, show_xy_combined, lst_a)
        while True:
            key = cv2.waitKey(0)
            action = get_key_action(key)
            if action == 'save':
                if len(lst_a[0]) != 7:
                    print("Label count hasn't reached the maximum (Please label 7).")
                else:
                    lst_a[2] = get_plate_string()
                    lst_a[0] = sorted(lst_a[0], key=lambda x: x[0])
                    lst = process_image_annotations(lst_a)
                    save_labels_to_file(lst, self.save_img)
                    self.img_count += 1
                    cv2.destroyAllWindows()
                    break
            elif action == 'pass':
                self.img_count += 1
                cv2.destroyAllWindows()
                break
            elif action == 'exit':
                cv2.destroyAllWindows()
                print('Program Terminated')
                quit()
            elif action == 'previous':
                self.img_count -= 1
                self.img_count = max(0, self.img_count)
                cv2.destroyAllWindows()
                break
            elif action == 'delete':
                lst_a[1] = copy.deepcopy(im0)
                self.img_tmp = handle_delete_buttom_up(lst_a)
                cv2.imshow(self.window_name, self.img_tmp)
            else:
                print('Input error. Please re-enter.')
    
    def load_image_without_model_LPR_label(self):
        print('Current image count:', self.img_count + 1)
        img_path, self.window_name, self.save_img = get_image_path_and_name(self.source, self.img_files, self.img_count, self.store)
        print('Current Image Path: ', img_path)
        im0 = cv2.imread(img_path)
        self.img_tmp, self.resize_scale = resize_to_fit_screen(im0)
        self.process_LPR_label()
    
    def load_image_without_model_obb_label(self):
        print('Current image count:', self.img_count)
        image_path, self.window_name, self.save_img = get_image_path_and_name(self.source, self.img_files, self.img_count, self.store)
        print('Current Image Path: ', image_path)
        im0 = cv2.imread(image_path)
        self.img_tmp = copy.deepcopy(im0)
        self.modify_images_obb()
        
                    
if __name__ == '__main__':
    
    # This handles general multilabel related variables.
    config_manager = ConfigManager()
    conf = config_manager.get_config()
    obj = conf['obj']
    clr = conf['clr']
    drawing = False
    key_actions_normal = conf['key_actions_normal']
    
    # This is the multilabel related variable that handles rotation.
    key_actions_obb = conf['key_actions_obb']
    rotated_pts = np.array([])
    rotated_pts = None
    drawing = False
    selected_edge = None
    offset = 5
    prev_mouse_move = None
    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--mode', type=str, choices=['normal', 'LPR', 'OBB'], required=True, help='Choose a mode to run the script(normal_multilabel, multilabel_for_ANPR, label_with_no_model_ANPR, label_with_no_model_normal_multilabel)')
    parser.add_argument('--last_time_num', type=int, default=None, help='Last time you stop at which image')
    parser.add_argument('--weights', type=str, default=None, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./dataset/images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imagesz', type=int, nargs=2, default=(640, 640), help='image size')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--store', type=str, default='./dataset/labels/', help='store')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--line_thickness', type=int, default=3, help='bounding box thickness (pixels)')
    
    args = parser.parse_args()
    
    # Determine if source is a folder
    if not os.path.isdir(args.source):
        print(f"Error: '{args.source}' is not a valid directory. Please provide a valid directory path.")
        exit(1)
    # Determine if weights is a file
    if args.weights and not os.path.isfile(args.weights):
        print(f"Error: '{args.weights}'  is not a valid file. Please provide a valid file path.")
        exit(1) 
    # Determine if store is a folder
    if not os.path.isdir(args.store):
        print(f"Error: '{args.store}' is not a valid directory. Please provide a valid directory path.")
        exit(1)
    # initialize the mode
    ATL = Statemachine_ATL(
                last_time_num=args.last_time_num,
                weights=args.weights,
                source=args.source,
                imagesz=tuple(args.imagesz),
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                max_det=args.max_det,
                store=args.store,
                obj=obj,
            )
    
    if args.mode == 'normal':
        if args.weights:
            ATL.run_model_multilabel()
        else:
            ATL.run_without_model()
    
    elif args.mode == 'LPR':
        if args.weights:
            ATL.run_model_LPR_label()
        else:
            ATL.run_without_model_LPR_label()

    elif args.mode == 'OBB':
        if args.weights:
            ATL.run_obb_model_label()
        else:
            ATL.run_without_model_obb_label()
            