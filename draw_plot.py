import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import numpy as np
pd.set_option('display.max_rows', None)  # 행의 수를 제한 없이 출력
pd.set_option('display.max_columns', None)  # 열의 수를 제한 없이 출력
# for k in range(1,17):
# df = pd.read_csv(f"video/0508/foot_csv/no_rock_{k}.csv")
for k in range(1,26):
    df = pd.read_csv(f"/home/aims/yolov8_8_0_10/ultralytics/yolov5_obb/video/0703/foot_csv/no_rock_{k}.csv")
    df2=np.where(df['data']>0)[0]
    x=df2[0::2]
    y=df2[1::2]
    # plt.plot(x,y,'o')

    # print(y['data'][3])
    # print(y['data'][1])
    # temp=y['data'][1]
    a=[]
    left=[]
    right=[]
    left_y_index=[]
    right_y_index=[]

    color=[255, 0, 0]

    for j in x:
        if df['data'][j]<340:
            left.append(j)
            left_y_index.append(j+1)
        else:
            right.append(j)
            right_y_index.append(j+1)

    new_index=np.asarray([],np.int64)
    peaks,properties=find_peaks(-df['data'][y],distance=70,height=-280)
    left_y=[]
    right_y=[]
    # print(left)

    # print(peaks)

    # temp=peaks[0]
    # temp_dic=dict()
    # for i in range(1,len(peaks)-1):
    #     if ((peaks[i]-temp)<10):
    #         temp_dic[f"{peaks[i]}"]=y['data'][peaks[i]*2+1]
            
    #     if((peaks[i]-temp)>10 or i==(len(peaks)-2)):
    #         if len(temp_dic)!=0:
    #             index=min(temp_dic, key=temp_dic.get)
    #             new_index=np.append(new_index,[index,peaks[i]])
    #         else:
    #             new_index=np.append(new_index,int(peaks[i]))
    #         temp=peaks[i+1]
    #         temp_dic=dict()
    # new_index=np.array(new_index, dtype=np.float64)
    # print(new_index*2+1)

    temp=peaks[0]
    temp_dic=dict()
    i=1
    while True:
        if (abs(peaks[i]-temp)<6):
            temp_dic[f"{temp}"]=df['data'][y[temp]]
            temp_dic[f"{peaks[i]}"]=df['data'][y[peaks[i]]]

        if(abs(peaks[i]-temp)>6 or i==(len(peaks)-1)):   
            # print(temp_dic)
            # print(peaks[i])        
            # print("")
            if i==(len(peaks)-1) and len(temp_dic)==0:
                new_index=np.append(new_index,int(peaks[i])) 
            if i==(len(peaks)-1) and len(temp_dic)!=0 and peaks[i]-temp>6:
                new_index=np.append(new_index,int(peaks[i])) 

            if len(temp_dic)!=0:
                index=min(temp_dic, key=temp_dic.get)
                new_index=np.append(new_index,int(index))
                temp_dic=dict()
            else:
                new_index=np.append(new_index,int(temp)) 

        temp=peaks[i]
        i+=1
        if i==len(peaks):
            break



    x_2=y[new_index]

    for i in left:
        left_y.append(df['data'][i+1])

    for i in right:
        right_y.append(df['data'][i+1])

    plt.figure(figsize=(10,6))
    plt.plot(left,left_y,'o',color='r',label='left')
    plt.plot(right,right_y,'o',label='right')
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_label_position('top') 
    plt.gca().xaxis.tick_top()
    plt.xlabel('Frame',loc='right')
    plt.ylabel('Y')
    plt.legend(['left','right'], loc='upper right')
    plt.plot(df['data'][x_2],"x",color='black')

    # x_1=range(0,300,20)
    # plt.xticks(x_1)
    # plt.show()
    # temp=left_y[0]

    # for i in left_y:
    #     a.append(i-temp)
    #     temp=i
    # plt.plot(left,a,'o')
    # x_1=range(0,900,50)
    # plt.xticks(x_1)

    # print(x)
    # print(y)

    # plt.plot(df,'o',label="left")
    # plt.plot(df2,'o',label="right",color='r')
    # plt.legend()
    plt.savefig(f'./plot/my_plot{k}.png')
    plt.close()