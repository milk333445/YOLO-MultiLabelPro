# YOLO-MultiLabelPro

MultiLabelPro is a powerful tool designed specifically for machine learning and computer vision researchers, aimed at simplifying the process of creating datasets for various types of annotations. It comprises two main functionalities:

1. YOLOAutoLabeler:This feature supports the rapid annotation of YoloV5 datasets, allowing users not only to label general multi-class objects and draw bounding boxes but also providing advanced automatic annotation capabilities. Users can annotate the required objects more swiftly. Additionally, it includes specialized support for annotating characters on license plates.
2. YOLOOBBAutoLabeler:This feature enables you to effortlessly create rotational box datasets. By interacting with images, you can swiftly label objects within them and easily adjust the position and angle of the rotational boxes, ensuring the accuracy of your dataset. It specializes in handling rotational box annotations and offers advanced automatic annotation capabilities.

MultiLabelPro combines the functionalities of these two tools, offering comprehensive annotation support and incorporating advanced automatic annotation features to enhance annotation efficiency. This allows researchers to create high-quality datasets with greater ease.


### How to Use
1. First, you need to git clone https://github.com/nexuni/YOLO-tools.git
2. Download the necessary Python packages
```python=
git clone https://github.com/nexuni/YOLO-tools.git
cd YOLO-tools
```

### Quick Start: Prepare Your Data
1. Before you begin, please make sure you have prepared a folder containing the image dataset you want to label (it must contain .jpg, .jpeg, or .png image files). Note that any other file formats will be automatically ignored by the program.
2. Prepare an empty folder that will be used to store your label data files (.txt). The program will generate corresponding labels in this folder.
3. If you plan to use the YoloV5 model for automatic labeling, ensure that you have prepared the relevant YoloV5 weight files. You will need to pass them as parameters in the subsequent steps.


## YOLOAutoLabeler
### Key Features:
1. Swift Multi-Class Labeling: YOLOLabelMaster offers an intuitive and efficient interface, allowing you to easily label multiple object classes and draw bounding boxes to create your YoloV5 dataset.
2. Advanced Automatic Annotation: With the assistance of the YoloV5 model you upload, YOLOLabelMaster possesses robust automatic annotation capabilities. It can automatically detect objects in images and generate preliminary labels, saving you valuable time.
3. License Plate Character Annotation: In addition to general object labeling, this tool provides convenient functionality for annotating characters on license plates, ensuring more precise recognition.
4. Editing and Saving: YOLOLabelMaster enables you to effortlessly edit automatically generated labels to ensure accuracy. Once your labeling is complete, you can conveniently save your work.


### Configure autolabel_settings.yaml
1. Before using this tool, you need to configure autolabel_settings.yaml to set the object classes you want to label. You can define class names by editing the "classes" field in autolabel_settings.yaml. Please be aware that class names will be encoded in the order they appear in the file, such as 0, 1, 2, and so on, so pay special attention to this part.
```python=
classes:
  - A # 0
  - B # 1
  - C # 2
  - D # 3
  、
  、
  、
```

2. Additionally, you can customize "key_actions" based on your personal preferences to configure custom keyboard triggers for adjusting labeling methods to meet your needs.
```python=
key_actions_normal:
  13: 'save' # enter
  32: 'modify' # space
  27: 'exit' # esc
  100: 'pass' # d
  68: 'pass' # D
  65: 'previous' # a
  97: 'previous' # A
  119: 'switch_prev' # w
  115: 'switch_next' # s
  87: 'switch_prev' # W
  83: 'switch_next' # S
  9: 'switch_next_bbox' # tab
```
- Here are some explanations:
  - Mouse Actions:
    - Left-Click and Drag: You can left-click and drag on the image to draw and mark annotations directly on the picture.
    - Right-Click: Right-click to cancel annotations you have previously marked on the image.
  - Key Actions:
    - Enter : Press the Enter key to save the annotation for the current image.
    - Space : Press the Space key to select and modify the annotation for the current image (only required when a model is loaded).
    - Esc : Press the Esc key to exit the tool.
    - D or d : Press D or d to skip annotating the current image.
    - A or a : Press A or a to navigate to the previous image.
    - W or w : Press W or w to switch to the previous annotation class.
    - S or s : Press S or s to switch to the next annotation class.
    - tab : Press tab to choose the bbox that you want to delete(press b).
- Use Case: Annotating an Image of an Apple:
  1. Launch the YOLOAutoLabeler tool.
  2. Load your YoloV5 model (if available) by pressing the Space key to enable annotation.
  3. Open an image for annotation that contains an apple.
  4. if you want to delete the bbox, you can press tab to selecte the bbox and press b to delete it.
  5. Left-click and drag the mouse to draw a bounding box around the apple. You can adjust the size and position of the bounding box to accurately enclose the apple.
  6. Right-click to cancel a bounding box if you accidentally marked the wrong object or location.
  7. When you are satisfied with the annotation, press the Enter key to save it.
  8. If you have multiple object classes to annotate, press the W or S keys to switch to a different annotation class.
  9. Repeat steps 4 to 6 until you have completed annotating the image.
  10. If you need to skip annotating the current image, press the "D" key.
  11. Press the "A" key to navigate to the previous image and repeat steps 4 to 8 until all images are annotated.
  12. When you have finished annotating all images, press the Esc key to exit the tool. Your YoloV5 dataset now contains accurate annotation information and can be used for model training.
### Launching the Main Program and Setting Parameters
```python=
python main.py --mode [choose mode(normal、LPR)] --last_time_num [last image number] --weights [model weights file path] --source [image folder path] --imagesz [image width] [image height] --conf_thres [object confidence threshold] --iou_thres [NMS IOU threshold] --max_det [maximum detections per image] --store [label storage path] --images_store [images that have been labeled]
```
- You don't need to configure all parameters every time. Here are detailed explanations of some important parameters such as mode, last_time_num, weights, source,  store and images_store:
  - mode (Mode):
    - normal (Normal Mode): This is the general mode for multi-class labeling where you can quickly switch object categories using the keyboard.
    - LPR (License Plate Mode): This mode is designed for annotating license plate characters (default is 3 English characters + 4 numeric characters). In this mode, you will need to input all labels via the terminal.
  - last_time_num (Last Image Number):
    - This parameter allows you to quickly jump to a specific image number. If you have previously labeled the first five images, you can input last_time_num 6 to directly jump to labeling the sixth image.
  - weights (Model Weights File):
    - If you wish to perform pre-labeling using the YoloV5 model, provide the path to the weight file. The program will automatically switch to pre-labeling mode and give you the option to modify labels or save them directly. 
  - source (Image Folder Path):
    - This is a mandatory parameter that specifies the path to the folder containing the images you want to label. 
  - store (Label Storage Path):
    - This is also a mandatory parameter that specifies where you want to save the label information.
  - images_store
    - if you want to make store the images(because if you skip the images, the labels file won't match the original images file), use this parameter. 
   
Using these parameters, you can easily configure the main program to label your image dataset and perform labeling in different modes according to your needs.

## Examples (Four Modes)
### License Plate Character Mode (No Model Pre-labeling)
```python=
python .\autolabeling.py --mode LPR --source "./images" --store "./labels_test"
```
### License Plate Character Mode (Model Pre-labeling)
```python=
python .\autolabeling.py --mode LPR --source "./images" --store "./labels_test" --weights "C:\Users\User\autolabeling\yolov5\runs\train\exp3\weights\best.pt"
```
### General Multi-Class Assistance Mode (No Model Pre-labeling)
```python=
python .\autolabeling.py --mode normal --source "./images" --store "./labels_test"
```
### General Multi-Class Assistance Mode (Model Pre-labeling)
```python=
python .\autolabeling.py --mode normal --source "./images" --store "./labels_test" --weights "C:\Users\User\autolabeling\yolov5\runs\train\exp3\weights\best.pt"
```
 


## YOLOOBBAutoLabeler

### Key Features:
1. Rapid Creation of Rotational Box Datasets:The Rotational Box Labeling Assistant enables you to effortlessly create rotational box datasets. By interacting with images, you can swiftly label objects within them and easily adjust the position and angle of the rotational boxes, ensuring the accuracy of your dataset.
2. Advanced Automatic Annotation: With the assistance of the YoloV5obb model you upload, YOLOOBBAutoLabeler possesses robust automatic annotation capabilities. It can automatically detect objects in images and generate preliminary labels, saving you valuable time.

### Data format
1. Our annotation generation method is designed to provide a structured and efficient way to label objects within images using a specific format. The annotations consist of the following elements, ordered from left to right, representing a clockwise coordinate system:
```python=
  x1      y1       x2        y2       x3       y3       x4       y4       classname     diffcult

 465.0   287.0    418.0     123.0    490.0    102.0    537.0    267.0        tag           0
```

### Configure autolabel_settings.yaml
1. Before using this tool, you need to configure autolabel_settings.yaml to set the object classes you want to label. You can define class names by editing the "classes" field in autolabel_settings.yaml. Please be aware that class names will be encoded in the order they appear in the file, such as 0, 1, 2, and so on, so pay special attention to this part.
```python=
classes:
  - A # 0
  - B # 1
  - C # 2
  - D # 3
  、
  、
  、
```
2. Additionally, you can customize "key_actions" based on your personal preferences to configure custom keyboard triggers for adjusting labeling methods to meet your needs.
```python=
key_actions_obb:
  27: 'exit' # esc
  61: 'plus' # =
  45: 'minus' # -
  109: 'minus' # -
  9: 'switch_next_side' # tab
  98: 'delete'  # b
  66: 'delete'  # B
  119: 'switch_prev' # w
  115: 'switch_next' # s
  87: 'switch_prev' # W
  83: 'switch_next' # S
  13: 'save' # enter
  65: 'previous' # a
  97: 'previous' # A
  100: 'pass' # d
  68: 'pass' # D
```
- Here are some explanations:
  - Mouse Actions:
    - Left-Click (No need to hold): Simply left-click to enter marking mode, where you can annotate rotation boxes on the image.
    - Mouse Movement: Move the mouse to adjust the size of the rotation box.
    - Right-Click (No need to hold): After entering marking mode by left-clicking, a single right-click (no need to hold) allows you to adjust the relative position of the rotation box. When you perform a single right-click and move the mouse, you can adjust the position of the rotation box.
    - Confirm Annotation: In marking mode, once you are satisfied with the position and size of the rotation box, you can confirm the annotation to complete the annotation of the rotation box. 
  - Key Actions:
    - Esc : Exit annotation mode. Pressing the 'esc' key allows you to exit the annotation process.
    - = or +: Increase the size of a selected side of the bounding box. This key allows you to expand a specific side of the bounding box.
    - (-) : Decrease the size of a selected side of the bounding box. Use this key to reduce the size of a particular side of the bounding box.
    - Tab : Switch to the next side for adjustment. Press 'tab' to toggle between sides for resizing.
    - B or b : Delete the currently selected bounding box. 'B' and 'b' keys can be used to remove the active bounding box.
    - W or w : Switch to the previous class label. These keys allow you to cycle through class labels in reverse order.
    - S or s : Switch to the next class label. 'S' and 's' keys help you navigate through class labels in forward order.
    - Enter : Save the annotation coordinates. Press 'enter' to save the current annotation, ensuring that the changes are recorded.
    - A or a : Move to the previous image. These keys allow you to navigate to the previous image in your dataset.
    - D or d : Move to the next image. Use 'D' and 'd' keys to proceed to the next image in your dataset.

- Usage Example: Annotating a Rotation Box:
  1. Launch the YOLOOBBAutoLabeler tool.
  2. Open an image that needs annotation. This image contains an object that needs to be annotated, such as a traffic sign.
  3. Use the left mouse button to click (no need to hold), which will put you in marking mode, allowing you to start annotating a rotation box on the image.
  4. Move the mouse to adjust the size of the rotation box to ensure it accurately surrounds the object, such as the traffic sign.
  5. If you need to adjust the relative position of the rotation box, press the right mouse button (no need to hold) and then move the mouse to adjust the position of the rotation box.
  6. Click the left mouse button again (no need to hold) to fix the position of the rotation box.
  7. If you realize that a previously annotated rotation box is incorrect, press the "B" key to delete the previously marked rotation box.
  8. Use the "Tab" key to switch to the next edge of the bounding box for resizing (using the +- keys) if necessary.
  9. When you are satisfied with the annotation result, press the Enter key to save the annotation.
  10. If you have multiple objects of different classes to annotate, use the W key or S key to switch to different class labels.
  11. Repeat steps 3 through 8 until you have completed annotating the image.
  12. If you need to annotate the next image, press the D key.
  13. If you need to annotate the previous image, press the A key.
  14. When you have annotated all the images, press the Esc key to exit the tool. You now have a dataset with accurately annotated rotation boxes that can be used for model training.

### Launching the Main Program and Setting Parameters
```python=
python main.py --mode [choose mode(OBB)] --last_time_num [last image number] --weights [model weights file path] --source [image folder path] --imagesz [image width] [image height] --conf_thres [object confidence threshold] --iou_thres [NMS IOU threshold] --max_det [maximum detections per image] --store [label storage path] --images_store [images that have been labeled]
```
- You don't need to configure all parameters every time. Here are detailed explanations of some important parameters such as mode, last_time_num, weights, source, store, images_store:
  - mode (Mode):
    - OBB (Normal Mode): This is the general mode for multi-class labeling where you can quickly switch object categories using the keyboard.
  - last_time_num (Last Image Number):
    - This parameter allows you to quickly jump to a specific image number. If you have previously labeled the first five images, you can input last_time_num 6 to directly jump to labeling the sixth image.
  - weights (Model Weights File):
    - If you wish to perform pre-labeling using the YoloV5OBB model, provide the path to the weight file. The program will automatically switch to pre-labeling mode and give you the option to modify labels or save them directly. 
  - source (Image Folder Path):
    - This is a mandatory parameter that specifies the path to the folder containing the images you want to label. 
  - store (Label Storage Path):
    - This is also a mandatory parameter that specifies where you want to save the label information.
  - images_store
    - if you want to make store the images(because if you skip the images, the labels file won't match the original images file), use this parameter. 
  
Using these parameters, you can easily configure the main program to label your image dataset and perform labeling in different modes according to your needs.

