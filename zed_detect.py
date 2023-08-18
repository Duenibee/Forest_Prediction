import rotated_distribution as rd
import walking 
import pyzed.sl as sl
import info_function as info

import numpy as np
import cv2
import torch
from my_model import CNN
from my_obb_model import Obb
from data_pre import real_time_data_pre

from utils.torch_utils import select_device
import math
def main():
    # parameters
    WEIGHTS = '/home/aims/obb_contents/weights/circle_foot.pt'
    PATH_weight_x="/home/aims/obb_contents/weights/only_x/240.pth"
    PATH_weight_y="/home/aims/obb_contents/weights/only_y/100.pth" 
    DEVICE = '0'
    blue_color = (255, 0, 0)
    red_color = (0, 0, 255)

    # Initialize
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print('device:', device)

    # Load model 
    # v5_obb
    model_obb= Obb(WEIGHTS,device)

    # gait_cycle_model x,y
    model_x = CNN().to(device)
    model_y = CNN().to(device)

    model_x.load_state_dict(torch.load(PATH_weight_x))
    model_x.eval()

    model_y.load_state_dict(torch.load(PATH_weight_y))
    model_y.eval()

    # if half:
    #     model_obb.model.half()  # to FP16

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

    # video
    # cap = cv2.VideoCapture("/home/aims/obb_contents/circle/video/circle1.avi")
    # fps = cap.get(cv2.CAP_PROP_FPS)

    delay=round(1000/init.camera_fps) 

    real_data=real_time_data_pre()
    key = ''
    print("delay", delay)


    with torch.no_grad():
        while key != 113:  # for 'q' key
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                frame=mat.get_data()[:,:,:3]
                new_im=frame.copy()
                det = model_obb.detect(frame, model_obb.model)
                coord=[]
                labels=[]
                nums=0
                temp_img=new_im.copy()
                inference=0
                rocks,j_r,foots,j_f=real_time_data_pre.get_info(det)
                for *poly, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label2=(f'{model_obb.names[c]}')
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
                    temp_img=cv2.circle(temp_img, (x,y), 5, red_color, -1)
                
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
            else:
                break
    cv2.destroyAllWindows()
    cam.close()
    print("\nFINISH")

if __name__=="__main__":
    main()