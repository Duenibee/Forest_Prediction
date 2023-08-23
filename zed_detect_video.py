from ultralytics import YOLO
import pyzed.sl as sl
import info_function as info

import numpy as np
import cv2
import torch
from my_model import CNN
# from my_obb_model import Obb
from data_pre import real_time_data_pre
from utils.torch_utils import select_device
def main():
    # parameters
    WEIGHTS = '/home/aims/obb_contents/weights/v5_obb/0821.pt'
    PATH_weight_x="/home/aims/obb_contents/weights/only_x/1000.pth"
    PATH_weight_y="/home/aims/obb_contents/weights/only_y/300.pth" 
    DEVICE = '0'
    blue_color = (255, 0, 0)
    red_color = (0, 0, 255)
    center_axis=340
    # Initialize
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print('device:', device)

    # Load model 
    # v8_segment
    model_v8= YOLO('/home/aims/obb_contents/weights/v8/best.pt') 

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

    cap = cv2.VideoCapture("/home/aims/obb_contents/rock/rock7.avi")
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay=round(1000/fps) 


    # video
    # cap = cv2.VideoCapture("/home/aims/obb_contents/circle/video/circle1.avi")
    # fps = cap.get(cv2.CAP_PROP_FPS)


    real_data=real_time_data_pre()
    key = ''



    with torch.no_grad():
        while key != 113:  # for 'q' key
            ret, frame =cap.read()
            # print(ret)
            if ret:
                temp_img=frame.copy()
                results = model_v8.predict(source=frame)
                rocks,j_r,foots,j_f =real_time_data_pre.get_info(results)
                real_data.data_foot(foots,j_f,real_data)

                temp_img=real_data.draw_rect(temp_img,rocks,foots,j_r,j_f)

                if real_data.count >= 60:
                    arr=real_data.input_arr.reshape(1,1,30,6)
                    x_in=torch.from_numpy(arr).to(device).float()
                    outputs_x = model_x(x_in)
                    outputs_y = model_y(x_in)
                    x=int(outputs_x[:,0])
                    y=int(outputs_y[:,0])
                    temp_img=cv2.circle(temp_img, (x,y), 5, red_color, -1)
                    temp_img=cv2.line(temp_img,(center_axis,0),(center_axis,400),color=red_color,thickness=1)
                # input_data_check
                # temp_img=real_data.input_data_check(real_data,temp_img)
                
                cv2.imshow("ZED", temp_img)
                key = cv2.waitKey(delay)
            else:
                break
    cv2.destroyAllWindows()
    cap.release()
    print("\nFINISH")

if __name__=="__main__":
    main()