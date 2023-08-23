import numpy as np
import torch
import cv2

import math
blue_color = (255, 0, 0)
red_color = (0, 0, 255)
green_color = (0, 255, 0)

class real_time_data_pre:
    def __init__(self) -> None:
        # 10 frame 씩 저장
        self.num_frames_per_segment = 60
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


    def get_info(results):
        j_f=0
        j_r=0
        if len(results)!=0:
            for r in results:                                                                                                                                                                                                            
                boxes = r.boxes  # Boxes object for bbox outputs                                                                                                                                                                         
                masks = r.masks  # Masks object for segment masks outputs 
                check=boxes.cls==0
                check_arr=np.array(check.cpu().numpy())
                check_index_rock=np.where(check_arr==True)
                check_index_foot=np.where(check_arr==False)
                cx_cy_w_h_r =np.zeros((0,4))
                cx_cy_w_h_f =np.zeros((0,4))
                # 각 배열에서 cx cy 좌표 추출
                if len(boxes.xywh) !=0:
                    for index1, c in enumerate(boxes.xywh):
                        c_np=np.array(c.cpu().numpy())
                        if index1 in check_index_rock[0]:
                            print(index1)
                            cx_cy_w_h_r=np.insert(cx_cy_w_h_r,j_r,[c_np[0],c_np[1],c_np[2],c_np[3]],axis=0)
                            j_r+=1
                        elif index1 in check_index_foot[0] and 200<c_np[0]<450:
                            cx_cy_w_h_f=np.insert(cx_cy_w_h_f,j_f,[c_np[0],c_np[1],c_np[2],c_np[3]],axis=0)
                            j_f+=1
                x_coordinates_mask = []
                y_coordinates_mask = []
                # 각 배열에서 mask x y 좌표 추출
                # for index, arr in enumerate(masks.xy):
                #     if index in check_index_rock[0]:
                #         x = [point[0] for point in arr]  # 각 배열 내 모든 좌표의 첫 번째 요소가 x 좌표
                #         x_coordinates_mask.append(x)
                #         y = [point[1] for point in arr]  # 각 배열 내 모든 좌표의 첫 번째 요소가 x 좌표
                #         y_coordinates_mask.append(y)
        return cx_cy_w_h_r,j_r,cx_cy_w_h_f,j_f


    def data_foot(self,foots,j_f,real_data):
        if j_f==0:
            if real_data.count<60:
                real_data.first_pre(real_data.ref_left[0],real_data.ref_left[1],0)
                real_data.first_pre(real_data.ref_right[0],real_data.ref_right[1],1)
            else:
                real_data.data_pre(0, 0, 0, 0 ,0 , 1)
        if j_f==1:
            center_x = foots[0][0]
            center_y = foots[0][1]

            if real_data.count<60:
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

            if real_data.count<60:
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

    def draw_rect(self,temp_img,rocks,foots,j_r,j_f):
        # draw rocks
        if j_r>0:
            for rock_xywh in rocks:
                x1 = int(rock_xywh[0] - rock_xywh[2] / 2)
                y1 = int(rock_xywh[1] - rock_xywh[3] / 2)
                x2 = int(rock_xywh[0] + rock_xywh[2] / 2)
                y2 = int(rock_xywh[1] + rock_xywh[3] / 2)
                temp_img=cv2.rectangle(temp_img, (x1, y1), (x2 ,y2), blue_color, 2) 
        if j_f>0:
            for foot_xywh in foots:
                x1 = int(foot_xywh[0] - foot_xywh[2] / 2)
                y1 = int(foot_xywh[1] - foot_xywh[3] / 2)
                x2 = int(foot_xywh[0] + foot_xywh[2] / 2)
                y2 = int(foot_xywh[1] + foot_xywh[3] / 2)
                temp_img=cv2.rectangle(temp_img, (x1, y1), (x2 ,y2), red_color, 2) 
       
        return temp_img
    

    def input_data_check(self,real_data,temp_img):
        # input data check
        x_l=int(real_data.input_arr[-1,0])
        y_l=int(real_data.input_arr[-1,1])
        x_r=int(real_data.input_arr[-2,0])
        y_r=int(real_data.input_arr[-2,1])
        temp_img=cv2.circle(temp_img, (x_l,y_l), 5, blue_color, -1)
        temp_img=cv2.circle(temp_img, (x_r,y_r), 5, red_color, -1)
        return temp_img