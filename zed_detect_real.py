import rotated_distribution as rd
import walking 
import pyzed.sl as sl
import info_function as info
import math
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from my_model import CNN
from my_obb_model import Obb

from utils.datasets import letterbox
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression_obb , scale_polys)
from utils.torch_utils import select_device
from utils.rboxs_utils import rbox2poly
import math

WEIGHTS = '/home/aims/obb_contents/weights/circle_foot.pt'
PATH_weight_x="/home/aims/obb_contents/weights/only_x/240.pth"
PATH_weight_y="/home/aims/obb_contents/weights/only_y/100.pth" 

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

# Initialize
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
print('device:', device)

# Load model 
# v5_obb
model = DetectMultiBackend(weights, device=device, dnn=dnn)

# gait_cycle_model x,y
model_x = CNN().to(device)
model_y = CNN().to(device)

model_x.load_state_dict(torch.load(PATH_weight_x))
model_x.eval()

model_y.load_state_dict(torch.load(PATH_weight_y))
model_y.eval()

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
    # Load image
    frame = image # BGR

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

        if len(det):

            pred_poly = scale_polys(im.shape[2:], pred_poly, [376,672,3])
            det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

    return det

def get_info(det):

    rock_info=[]
    foot_info=[]
    j_r=0
    j_f=0

    for *bbox, conf2, cls2 in reversed(det):
        if 168<bbox[0]<504:
            c2 = int(cls2)
      

            if c2==0:
                r_center_coord=[float((bbox[0]+bbox[4])/2),float((bbox[1]+bbox[5])/2)]
                r_height=math.dist((bbox[0],bbox[1]),(bbox[6],bbox[7]))
                r_width=math.dist((bbox[0],bbox[1]),(bbox[2],bbox[3]))
                r_rad=math.atan2(abs(bbox[7]-bbox[1]),abs(bbox[6]-bbox[0]))
                rock_info.append([r_center_coord[0],r_center_coord[1],r_height,r_width,r_rad]) # default left foot
                j_r+=1
            else:
                f_center_coord=[float((bbox[0]+bbox[4])/2),float((bbox[1]+bbox[5])/2)]
                f_height=math.dist((bbox[0],bbox[1]),(bbox[6],bbox[7]))
                f_width=math.dist((bbox[0],bbox[1]),(bbox[2],bbox[3]))
                f_rad=0
                foot_info.append([f_center_coord[0],f_center_coord[1],f_height,f_width,f_rad]) # default left foot
                j_f+=1

    return rock_info,j_r,foot_info,j_f

blue_color = (255, 0, 0)
green_color = (0, 255, 0)


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

# cap = cv2.VideoCapture("/home/aims/obb_contents/circle/video/circle1.avi")
# fps = cap.get(cv2.CAP_PROP_FPS)
delay=round(1000/init.camera_fps) 



class real_time_data_pre:
    def __init__(self) -> None:
        # 10 frame 씩 저장
        self.num_frames_per_segment = 20
        self.input_arr=np.zeros((0,3))

        # reference 설정
        self.ref_left=(315,340)
        self.ref_right=(385,340)
        self.count=0
        self.center_axis=340
    def first_pre(self, x, y, l_r):
        in_data=[x, y, l_r]
        self.input_arr=np.insert(self.input_arr,self.count,in_data,axis=0)
        self.count+=1

    def data_pre(self, x_1, y_1, l_r_1, x_2, y_2, l_r_2):
        data_np_2=np.array([[x_1, y_1, l_r_1],[x_2, y_2, l_r_2]])
        temp_arr=self.input_arr
        index_0=np.where(self.input_arr[:,2]==0)[0]
        index_1=np.where(self.input_arr[:,2]==1)[0]
        # 발이 하나도 안잡혔을때
        if  data_np_2[0,0]==0:
            # 왼발부터 
            x_vel=0
            y_vel=0
            temp_x=temp_arr[index_0[-7],0]
            temp_y=temp_arr[index_0[-7],1]
            for i22 in index_0[-6:-1]:
                x_vel+=temp_arr[i22,0]-temp_x
                y_vel+=temp_arr[i22,1]-temp_y
                temp_x=temp_arr[i22,0]
                temp_y=temp_arr[i22,1]
            x_vel/=4
            y_vel/=4
            x_temp=np.round(temp_arr[index_0[-1],0]+x_vel,0)
            y_temp=np.round(temp_arr[index_0[-1],1]+y_vel,0)

            # extrapolation의 제한을 둔다.
            data_np_2[0,0]=x_temp
            data_np_2[0,1]=(lambda y: y if y<340 else self.ref_left[1]) (y_temp)

            # 오른발 extrapolation 
            x_vel=0
            y_vel=0
            temp_x=temp_arr[index_1[-7],0]
            temp_y=temp_arr[index_1[-7],1]
            for i22 in index_1[-6:-1]:
                x_vel+=temp_arr[i22,0]-temp_x
                y_vel+=temp_arr[i22,1]-temp_y
                temp_x=temp_arr[i22,0]
                temp_y=temp_arr[i22,1]
            x_vel/=4
            y_vel/=4
            x_temp=np.round(temp_arr[index_0[-1],0],0)
            y_temp=np.round(temp_arr[index_0[-1],1]+y_vel,0)

            # extrapolation의 제한을 둔다.
            data_np_2[1,0]=x_temp
            data_np_2[1,1]=(lambda x: x if x<340 else self.ref_right[1]) (y_temp)

            insert1=np.zeros((0,3))
            insert2=np.zeros((0,3))

            insert1=np.insert(insert1,0,[data_np_2[0,0],data_np_2[0,1],data_np_2[0,2]],axis=0)
            insert2=np.insert(insert2,0,[data_np_2[1,0],data_np_2[1,1],data_np_2[1,2]],axis=0)
    
            # temp_arr=np.r_[temp_arr,insert1]
            # temp_arr=np.r_[temp_arr,insert2]
        # data 하나만 0배열인경우
        elif  data_np_2[1,0]==0:
            if data_np_2[1,2]==0:
                x_vel=0
                y_vel=0
                temp_x=temp_arr[index_0[-7],0]
                temp_y=temp_arr[index_0[-7],1]
                for i22 in index_0[-6:-1]:
                    x_vel+=temp_arr[i22,0]-temp_x
                    y_vel+=temp_arr[i22,1]-temp_y
                    temp_x=temp_arr[i22,0]
                    temp_y=temp_arr[i22,1]
                x_vel/=4
                y_vel/=4

                x_temp=np.round(temp_arr[index_0[-1],0],0)
                y_temp=np.round(temp_arr[index_0[-1],1]+y_vel,0)

                # extrapolation의 제한을 둔다.
                data_np_2[1,0]=x_temp
                data_np_2[1,1]=(lambda x: x if x<340 else self.ref_left[1]) (y_temp)
        
            if data_np_2[1,2]==1:
                x_vel=0
                y_vel=0
                temp_x=temp_arr[index_1[-7],0]
                temp_y=temp_arr[index_1[-7],1]
                for i22 in index_1[-6:-1]:
                    x_vel+=temp_arr[i22,0]-temp_x
                    y_vel+=temp_arr[i22,1]-temp_y
                    temp_x=temp_arr[i22,0]
                    temp_y=temp_arr[i22,1]
                x_vel/=4
                y_vel/=4

                x_temp=np.round(temp_arr[index_1[-1],0],0)
                y_temp=np.round(temp_arr[index_1[-1],1]+y_vel,0)
                # extrapolation의 제한을 둔다.
                data_np_2[1,0]=x_temp
                data_np_2[1,1]=(lambda x: x if x<340 else self.ref_right[1]) (y_temp)

            insert1=np.zeros((0,3))
            insert2=np.zeros((0,3))
            insert1=np.insert(insert1,0,[data_np_2[0,0],data_np_2[0,1],data_np_2[0,2]],axis=0)
            insert2=np.insert(insert2,0,[data_np_2[1,0],data_np_2[1,1],data_np_2[1,2]],axis=0)
            # temp_arr=np.r_[temp_arr,insert1]
            # temp_arr=np.r_[temp_arr,insert2]
            # print(len(temp_arr))
        # 데이터가 왼/오 둘다 있을때
        elif  data_np_2[1,0]!=0:
            insert1=np.zeros((0,3))
            insert2=np.zeros((0,3))
            insert1=np.insert(insert1,0,[data_np_2[0,0],data_np_2[0,1],data_np_2[0,2]],axis=0)
            insert2=np.insert(insert2,0,[data_np_2[1,0],data_np_2[1,1],data_np_2[1,2]],axis=0)

            # temp_arr=np.r_[temp_arr,insert1]
            # temp_arr=np.r_[temp_arr,insert2]

        self.input_arr=np.delete(self.input_arr,0,axis=0)
        self.input_arr=np.delete(self.input_arr,0,axis=0)

        self.input_arr=np.append(self.input_arr,insert1,axis=0)
        self.input_arr=np.append(self.input_arr,insert2,axis=0)

real_data=real_time_data_pre()
loop_count_2=0
key = ''
print("delay", delay)
with torch.no_grad():
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            frame=mat.get_data()[:,:,:3]
            new_im=frame.copy()
            det = detect(frame, model)
            coord=[]
            labels=[]
            nums=0
            temp_img=new_im.copy()
            inference=0
            rocks,j_r,foots,j_f=get_info(det)
            for *poly, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label2=(f'{names[c]}')
                coord.append(poly)
                labels.append(label2)
                inference=1
     
            if j_f==0:
                if real_data.count<20:
                    real_data.first_pre(real_data.ref_left[0],real_data.ref_left[1],0)
                    real_data.first_pre(real_data.ref_right[0],real_data.ref_right[1],1)
                else:
                    real_data.data_pre(0, 0, 0, 0 ,0 , 1)
            if j_f==1:
                center_x = foots[0][0]
                center_y = foots[0][1]
    
                if real_data.count<20:
                    if center_x>real_data.center_axis:
                        real_data.first_pre(center_x,center_y,1)
                        real_data.first_pre(real_data.ref_left[0],real_data.ref_left[1],0)
                    else:
                        real_data.first_pre(center_x,center_y,0)
                        real_data.first_pre(real_data.ref_right[0],real_data.ref_right[1],1)
                else:
                    if center_x>real_data.center_axis:
                        real_data.data_pre(center_x,center_y,1, 0,0,0)
                    else:
                        real_data.data_pre(center_x,center_y,0, 0,0,1)
        

            elif j_f==2:
                center_x_1 = foots[0][0]
                center_y_1 = foots[0][1]
                center_x_2 = foots[1][0]
                center_y_2 = foots[1][1]

                if real_data.count<20:
                    if center_x_1>real_data.center_axis:
                        real_data.first_pre(center_x_2,center_y_2,0)
                        real_data.first_pre(center_x_1,center_y_1,1)
                    else:
                        real_data.first_pre(center_x_1,center_y_1, 0)
                        real_data.first_pre(center_x_2,center_y_2, 1)
                else:
                    if center_x_1>real_data.center_axis:
                        real_data.data_pre(center_x_2,center_y_2,0, center_x_1,center_y_1,1)
                    else:
                        real_data.data_pre(center_x_1,center_y_1, 0 , center_x_2,center_y_2, 1)

            # print(real_data.count)
            if inference!=0:
                for i2,j2 in zip(coord,label2):
                    pts=np.array([[i2[0].cpu().numpy(),i2[1].cpu().numpy()],[i2[2].cpu().numpy(),i2[3].cpu().numpy()],[i2[4].cpu().numpy(),i2[5].cpu().numpy()],[i2[6].cpu().numpy(),i2[7].cpu().numpy()]])
                    temp_img=cv2.polylines(temp_img,np.int32([pts]),isClosed=True,color=blue_color,thickness=3) 

            if real_data.count >= 20:
                arr=real_data.input_arr.reshape(1,1,10,6)
                x_in=torch.from_numpy(arr).to(device).float()
                outputs_x = model_x(x_in)
                outputs_y = model_y(x_in)
                x=int(outputs_x[:,0])
                y=int(outputs_y[:,0])
                temp_img=cv2.circle(temp_img, (x,y), 5, green_color, -1)
            
            # input data check
            # x_l=int(real_data.input_arr[-1,0])
            # y_l=int(real_data.input_arr[-1,1])
            # x_r=int(real_data.input_arr[-2,0])
            # y_r=int(real_data.input_arr[-2,1])
            # print((x_l,y_l))q
            # print((x_r,y_r))

            # temp_img=cv2.circle(temp_img, (x_l,y_l), 5, green_color, -1)
            # temp_img=cv2.circle(temp_img, (x_r,y_r), 5, green_color, -1)
            cv2.imshow("ZED", temp_img)
            key = cv2.waitKey(delay)
            loop_count_2+=1
        else:
            break
cv2.destroyAllWindows()
cam.close()
print("\nFINISH")

