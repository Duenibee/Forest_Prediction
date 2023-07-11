import math
def get_info(det,det2):

    rock_info=[]
    foot_info=[]
    j_r=0
    for rock in reversed(det):
        if 168 < rock[0]<504:
            r_center_coord=[float((rock[0]+rock[4])/2),float((rock[1]+rock[5])/2)]
            r_height=math.dist((rock[0],rock[1]),(rock[6],rock[7]))
            r_width=math.dist((rock[0],rock[1]),(rock[2],rock[3]))
            r_rad=math.atan2(abs(rock[7]-rock[1]),abs(rock[6]-rock[0]))
            rock_info.append([r_center_coord[0],r_center_coord[1],r_height,r_width,r_rad]) # default left foot
            j_r+=1


    j_f=0 
    for foot in reversed(det2):
        f_center_coord=[(foot[0]+foot[2])/2,(foot[1]+foot[3])/2]
        f_height=abs(foot[1]-foot[3])
        f_width=abs(foot[2]-foot[0])
        foot_info.append(f_center_coord[0].cpu().numpy())
        foot_info.append(f_center_coord[1].cpu().numpy())
        foot_info.append(f_height.cpu().numpy())
        foot_info.append(f_width.cpu().numpy())
        j_f+=1

    # walking algorithm
    # 0개 발이 감지될 상황을 위해 저장  
    if j_f==0:
        final_res=[0,0]
    # foot_info sorting (x value small)
    temp=0
    if j_f==2: # sort only when 2 foot detect
        if foot_info[4]-foot_info[0]<0:
            temp=foot_info[0:4]
            foot_info[0:4]=foot_info[4:]
            foot_info[4:]=temp
    

    return foot_info, rock_info , j_f, j_r