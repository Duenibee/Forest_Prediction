import rotated_distribution as rd
import csv
def walking_lr(foot_info,rock_info,j_f,j_r,foot_y_l,foot_y_r,start,left_right,new_im,frame_count,previous_foot,previous_foot_detect,min_foot,foot_y_l_all,foot_y_r_all):
    final_res=[0,0]
    
    if j_f==0:
        final_res=[0,0]
    # foot_info sorting (x value small)
    temp=0
    if j_f==2: # sort only when 2 foot detect
        if foot_info[4]-foot_info[0]<0:
            temp=foot_info[0:4]
            foot_info[0:4]=foot_info[4:]
            foot_info[4:]=temp
    
    # parameter 
    center_axis=340
    foot_cutline=260
    # 시작 발 결정
    if start==0:
        final_res=[0,0]
        if j_f==1:  # 1 foot
            if foot_info[0]<center_axis: # 중심기준 왼쪽
                foot_y_l.append(foot_info[1]) 
                foot_y_l_all.append(foot_info[1])
                left_right=0 # left
                start=1
                frame_count=1
                print("start1: left")
                previous_foot=foot_info[0:4]
                previous_foot_detect=left_right
                previous_frame_count=frame_count
            else:
                foot_y_r.append(foot_info[1]) 
                foot_y_r_all.append(foot_info[1]) 
                left_right=2 # right
                frame_count=1
                start=1
                previous_foot=foot_info[0:4]
                previous_foot_detect=left_right
                previous_frame_count=frame_count
                print("start2: right",len(previous_foot))


        elif j_f==2: # 2 foot
            # 2개 발감지시 사이 거리가 멀어져야 시작
            if abs(foot_info[1]-foot_info[5])>40:
                # 그때 발의 y값을 비교, ex) 왼발이 앞에있으면 y값 뺄때 음수 
                if foot_info[1]-foot_info[5]<0:
                    foot_y_l.append(foot_info[1]) 
                    left_right=0 # left
                    frame_count=1
                    start=1
                    print("start3: left")
                    previous_foot=foot_info[0:4]
                    previous_foot_detect=left_right
                    previous_frame_count=frame_count


                else:
                    foot_y_r.append(foot_info[5])
                    foot_y_r_all.append(foot_info[5]) 

                    left_right=2 # right
                    frame_count=1
                    start=1
                    previous_foot=foot_info[4:]
                    previous_foot_detect=left_right
                    previous_frame_count=frame_count
                    print("start4: right")

    else: # start=1 : walking
        if j_f==0:
            foot_distribution=rd.stride_distribution([],previous_foot,min_foot,left_right,previous_foot_detect,frame_count) # empty array -> ref point
            rock_distribution=rd.rotated_distribution(rock_info,[],min_foot,previous_foot,left_right,previous_foot_detect,j_r)
            final_res,final_array=rd.step_point(rock_distribution,foot_distribution,j_r)
        elif j_f==1:
            if foot_info[0]<center_axis: # 감지된 발이 왼발
                foot_y_l.append(foot_info[1])
                foot_y_l_all.append(foot_info[1])
                if left_right==0: # left
                    foot_distribution=rd.stride_distribution(foot_info[0:4],previous_foot,min_foot,left_right,previous_foot_detect,frame_count)
                    rock_distribution=rd.rotated_distribution(rock_info,foot_info[0:4],min_foot,previous_foot,left_right,previous_foot_detect,j_r)
                    final_res,final_array=rd.step_point(rock_distribution,foot_distribution,j_r)
                    if foot_info[1]>foot_cutline and foot_y_l[-1]-foot_y_l[-2]>-5: # 발 y의 현재값과 직전값을 비교함
                        print("change1")
                        left_right=2 # change
                        frame_count=1
                        min_foot=min(foot_y_l)
                        del foot_y_l[:-1]
                        # new_im=cv2.line(new_im,(0,foot_cutline),(700,foot_cutline),color=blue_color,thickness=1)
    
                else: # 감지된 발이 왼발인데 left_right가 왼발이 아닐때-> ref point
                    foot_distribution=rd.stride_distribution([],previous_foot,min_foot,left_right,previous_foot_detect,frame_count)
                    rock_distribution=rd.rotated_distribution(rock_info,[],min_foot,previous_foot,left_right,previous_foot_detect,j_r)
                    final_res,final_array=rd.step_point(rock_distribution,foot_distribution,j_r)

            if foot_info[0]>center_axis:
                foot_y_r.append(foot_info[1])
                foot_y_r_all.append(foot_info[1])
                
                if left_right==2:
                    foot_distribution=rd.stride_distribution(foot_info[0:4],previous_foot,min_foot,left_right,previous_foot_detect,frame_count)
                    rock_distribution=rd.rotated_distribution(rock_info,foot_info[0:4],min_foot,previous_foot,left_right,previous_foot_detect,j_r)
                    final_res,final_array=rd.step_point(rock_distribution,foot_distribution,j_r)
                    
                    if foot_info[1]>foot_cutline and foot_y_r[-1]-foot_y_r[-2]>-5:
                        print("change2")
                        left_right=0 # change
                        min_foot=min(foot_y_r)
                        frame_count=1
                        del foot_y_r[:-1]
                        # new_im=cv2.line(new_im,(0,foot_cutline),(700,foot_cutline),color=red,thickness=1)

                else:
                    foot_distribution=rd.stride_distribution([],previous_foot,min_foot,left_right,previous_foot_detect,frame_count)
                    rock_distribution=rd.rotated_distribution(rock_info,[],min_foot,previous_foot,left_right,previous_foot_detect,j_r)
                    final_res,final_array=rd.step_point(rock_distribution,foot_distribution,j_r)
            previous_foot=foot_info[0:4]
            previous_foot_detect=left_right       
            previous_frame_count=frame_count

        elif j_f==2:
            foot_y_l.append(foot_info[1])
            foot_y_l_all.append(foot_info[1])
            foot_y_r.append(foot_info[5])
            foot_y_r_all.append(foot_info[5])
            if foot_info[0+left_right*2]<=center_axis: #left
                foot_distribution=rd.stride_distribution(foot_info[0:4],previous_foot,min_foot,left_right,previous_foot_detect,frame_count)
                rock_distribution=rd.rotated_distribution(rock_info,foot_info[0:4],min_foot,previous_foot,left_right,previous_foot_detect,j_r)
                final_res=rd.step_point(rock_distribution,foot_distribution,j_r)
                final_res,final_array=rd.step_point(rock_distribution,foot_distribution,j_r)
                if foot_info[1]>foot_cutline and foot_y_l[-1]-foot_y_l[-2]>-5:
                    print("change3")
                    left_right=2 # change
                    frame_count=1
                    # new_im=cv2.line(new_im,(0,foot_cutline),(700,foot_cutline),color=blue_color,thickness=1)
                    min_foot=min(foot_y_l)
                    del foot_y_l[:-1]
                
                previous_foot=foot_info[0:4]
                previous_foot_detect=left_right
                previous_frame_count=frame_count

            elif foot_info[0+left_right*2]>center_axis: # right
                foot_distribution=rd.stride_distribution(foot_info[4:],previous_foot,min_foot,left_right,previous_foot_detect,frame_count)
                rock_distribution=rd.rotated_distribution(rock_info,foot_info[4:],min_foot,previous_foot,left_right,previous_foot_detect,j_r)
                final_res=rd.step_point(rock_distribution,foot_distribution,j_r)
                final_res,final_array=rd.step_point(rock_distribution,foot_distribution,j_r)
                if foot_info[1+left_right*2]>foot_cutline and foot_y_r[-1]-foot_y_r[-2]>-5:
                    left_right=0 # change
                    frame_count=1
                    print("change4")
                    min_foot=min(foot_y_r)
                    del foot_y_r[:-1]
                    # new_im=cv2.line(new_im,(0,foot_cutline),(700,foot_cutline),color=red,thickness=1)
                
                previous_foot=foot_info[4:]
                previous_foot_detect=left_right
                previous_frame_count=frame_count
        # 발이 잘못나온경우 수정 + 보폭 추가보정
        if left_right==0: # 왼발인데 오른발로 나온경우
            if foot_y_r[-1]<250:
                left_right==2
                frame_count=1

        else:
            if foot_y_l[-1]<250:
                left_right==0
                frame_count=1


    return final_res,frame_count, new_im, left_right,foot_y_l,foot_y_r,frame_count,start,previous_foot,previous_foot_detect, min_foot,foot_y_l_all,foot_y_r_all