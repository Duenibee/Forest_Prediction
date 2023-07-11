import rotated_distribution as rd
import walking 
import pyzed.sl as sl
import info_function as info
from ultralytics import YOLO
import math
import argparse
import os
import sys
import random
from pathlib import Path
from ultralytics.yolo.utils.ops import non_max_suppression,scale_boxes
from ultralytics.nn.autobackend import AutoBackend
import numpy as np
import cv2
import torch
from utils.datasets import letterbox
import torch.backends.cudnn as cudnn
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly
from ultralytics.yolo.utils.ops import Profile

WEIGHTS = './my_weight/best11.pt'
IMG_SIZE = (640,640)
DEVICE = '0'
AUGMENT = False
CONF_THRES = 0.3
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False
dnn=False
classes=None # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False,  # class-agnostic NMS
weights, imgsz =  WEIGHTS, IMG_SIZE
max_det=10 # maximum detections per image
line_thickness=1
hide_labels=False,  # hide labels
hide_conf=False,  # hide confidences

# Initialize
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
print('device:', device)

# Load model
model = DetectMultiBackend(weights, device=device, dnn=dnn)
# model_v8 = AutoBackend(weights="best_0.937.pt", device=device, dnn=dnn, fp16=half)

if half:
    model.half()  # to FP16

# Get names and colors
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)
half &= (pt or jit or engine) and device.type != 'cpu'
if pt or jit:
    model.model.half() if half else model.model.float()

# model_v8.warmup(imgsz=(1 if pt or model_v8.triton else bs, 3, *imgsz))
model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

def detect(image , model, AUGMENT = False):
    
    x , y, w, h, conf = 0, 0, 0, 0, 0.
    
    # Load image
    frame = image # BGR
    xyxy =[]
    label=[]

    # Padded resize
    im = letterbox(frame, imgsz, stride=stride)[0]

    # Convert
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    # Inference
    t0 = time_sync()
    pred = model(im, augment=False, visualize=False)
    # print('pred shape:', pred.shape)

    # Apply NMS
    pred = non_max_suppression_obb(pred, CONF_THRES, IOU_THRES, classes, agnostic_nms, multi_label=True, max_det=max_det)

    # Process detections
    det = pred[0]
    # print('det shape:', det.shape)
    s = ''
    s += '%gx%g ' % im.shape[2:]  # print string
    
    original = [0 , 0, 0, 0]
    for i, det in enumerate(pred):  # per image
        pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
        # seen += 1
        annotator = Annotator(im, line_width=line_thickness, example=str(names))

        if len(det):
            # Rescale polys from img_size to im0 size
            # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            pred_poly = scale_polys(im.shape[2:], pred_poly, [376,672,3])
            det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

            # # Print results
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class
            #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    return det

import math
def get_info(det):

    rock_info=[]
    foot_info=[]
    j_r=0
    j_f=0

    for *bbox, conf2, cls2 in reversed(det):
        if 168<bbox[0]<504:
            c2 = int(cls2)
            if names[c2]=='rock':
                print(names[c2])
                r_center_coord=[float((bbox[0]+bbox[4])/2),float((bbox[1]+bbox[5])/2)]
                r_height=math.dist((bbox[0],bbox[1]),(bbox[6],bbox[7]))
                r_width=math.dist((bbox[0],bbox[1]),(bbox[2],bbox[3]))
                r_rad=math.atan2(abs(bbox[7]-bbox[1]),abs(bbox[6]-bbox[0]))
                rock_info.append([r_center_coord[0],r_center_coord[1],r_height,r_width,r_rad]) # default left foot
                j_r+=1
            else:
                print(names[c2])

                f_center_coord=[float((bbox[0]+bbox[4])/2),float((bbox[1]+bbox[5])/2)]
                f_height=math.dist((bbox[0],bbox[1]),(bbox[6],bbox[7]))
                f_width=math.dist((bbox[0],bbox[1]),(bbox[2],bbox[3]))
                f_rad=math.atan2(abs(bbox[7]-bbox[1]),abs(bbox[6]-bbox[0]))
                foot_info.append([f_center_coord[0],f_center_coord[1],f_height,f_width,f_rad]) # default left foot
                j_r+=1

    return rock_info,j_r,foot_info,j_f

blue_color = (255, 0, 0)

print("Running...")
# zed parameter setting
# VGA	1344x376 / HD720 2560x720
# check in https://www.stereolabs.com/docs/video/camera-controls/
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.VGA	
init.camera_fps = 30
cam = sl.Camera()

if not cam.is_opened():
    print("Opening ZED Camera...")
status = cam.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit()
runtime = sl.RuntimeParameters()
mat = sl.Mat()

key = ''
while key != 113:  # for 'q' key
    err = cam.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        frame=mat.get_data()[:,:,:3]
        new_im=frame
        det = detect(frame, model)
        coord=[]
        labels=[]
        nums=0
        rocks,j_r,foots,j_f=get_info(det)
        print(j_r)
        print(rocks)
        print(j_f)
        print(foots)

        for *poly, conf, cls in reversed(det):
            c = int(cls)  # integer class
            nums+=1
            label2=(f'{names[c]+str(nums)} {conf:.2f}')
            coord.append(poly)
            labels.append(label2)
        if j_r>0:
            new_im = np.ascontiguousarray(new_im, dtype=np.uint8)
            pts=np.array([[poly[0].cpu().numpy()*1.875,poly[1].cpu().numpy()*2],[poly[2].cpu().numpy()*1.875,poly[3].cpu().numpy()*2],[poly[4].cpu().numpy()*1.875,poly[5].cpu().numpy()*2],[poly[6].cpu().numpy()*1.875,poly[7].cpu().numpy()*2]])
            new_im=cv2.polylines(new_im,np.int32([pts]),isClosed=True,color=blue_color,thickness=4) 
        else:
            new_im=frame
        cv2.imshow("ZED", new_im)
        key = cv2.waitKey(5)
    else:
        key = cv2.waitKey(5)
cv2.destroyAllWindows()

cam.close()
print("\nFINISH")

