import numpy as np
import math
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


