import rotated_distribution as rd
import walking 
import torch
import numpy as np
from utils.datasets import letterbox
from utils.datasets import letterbox
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression_obb, scale_polys)
from utils.torch_utils import select_device
from utils.rboxs_utils import rbox2poly
WEIGHTS = '/home/aims/obb_contents/weights/circle_foot.pt'
IMG_SIZE = (640,640)
DEVICE = '0'
AUGMENT = False
CONF_THRES = 0.7
IOU_THRES = 0.25
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




class Obb:
    def __init__(self, weights,device) -> None:
        self.device = device
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn)
        self.weights, self.imgsz =  weights, IMG_SIZE
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.half &= (self.pt or self.jit or self.engine) and self.device.type != 'cpu'
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()

    def detect(self, image , model, AUGMENT = False):
        
        # Load image
        frame = image # BGR


        # Padded resize
        im = letterbox(frame, imgsz, stride=self.stride)[0]

        # Convert
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        # Inference
        pred = model(im, augment=False, visualize=False)
        # print('pred shape:', pred.shape)

        # Apply NMS
        pred = non_max_suppression_obb(pred, CONF_THRES, IOU_THRES, classes, agnostic_nms, multi_label=True, max_det=max_det)

        # Process detections
        det = pred[0]
        # print('det shape:', det.shape)
        s = ''
        s += '%gx%g ' % im.shape[2:]  # print string
        
        for i, det in enumerate(pred):  # per image
            pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
            # seen += 1

            if len(det):

                pred_poly = scale_polys(im.shape[2:], pred_poly, [376,672,3])
                det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

        return det