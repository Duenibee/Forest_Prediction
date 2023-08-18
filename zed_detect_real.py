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
                temp_img=new_im.copy()
                rocks,j_r,foots,j_f,rock_info_xyxy ,foot_info_xyxy =real_time_data_pre.get_info(det)
                real_data.data_foot(foots,j_f,real_data)

                temp_img=real_data.draw_rect(temp_img,rock_info_xyxy,foot_info_xyxy)

                if real_data.count >= 20:
                    arr=real_data.input_arr.reshape(1,1,10,6)
                    x_in=torch.from_numpy(arr).to(device).float()
                    outputs_x = model_x(x_in)
                    outputs_y = model_y(x_in)
                    x=int(outputs_x[:,0])
                    y=int(outputs_y[:,0])
                    temp_img=cv2.circle(temp_img, (x,y), 5, red_color, -1)
                # input_data_check
                # temp_img=real_data.input_data_check(real_data,temp_img)
                
                cv2.imshow("ZED", temp_img)
                key = cv2.waitKey(delay)
            else:
                break
    cv2.destroyAllWindows()
    cam.close()
    print("\nFINISH")

if __name__=="__main__":
    main()