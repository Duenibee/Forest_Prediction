import cv2
import numpy as np
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import math

# default value
 
stride_y = 0
stride_half = 250

# previous_foot=[]  # [x,y,w,h]

global previous_foot_detect  # 0 or 2
    

def distance_from_foot_point(ref_point, points):
    distances = []
    for point in points:
        distance = math.sqrt((point[0] - ref_point[0])**2 + (point[1] - ref_point[1])**2)
        
        distances.append(distance)
    if len(distances)==0 or np.max(distances)==0:
        pass
    else:
        distances = distances/np.max(distances)    

    return distances



def rotated_distribution(point_array, foot, stride_min,previous_foot, foot_detect, previous_foot_detect, rock_num) :
    global stride_y
    arr_rock=np.ones((376, 672))
    
         
    # point_array2 = point_array
    if len(point_array)!=0:
        i=len(point_array)-1
        while True:
            if 168 > point_array[i][0] or  point_array[i][0] > 504 or point_array[i][1]>250 or point_array[i][1]<(stride_min-80): 
                point_array=np.delete(point_array, i, axis = 0) 
            i-=1
            if(i<0):
                break

    # index=[]              
    # if len(point_array) > 5 :
    #     point_array = sorted(point_array, key = lambda x: x[1], reverse = True)[0:5]
     
    # if len(point_array)!=0:
    #     for x in range(len(point_array)):
    #         index.append(np.where((168 > point_array[x][0] or  point_array[x][0] > 504 or point_array[x][1]>330 or point_array[x][1]<(stride_min-10))))
        
    #     for x in index:
    #         point_array=np.delete(point_array, x, axis = 0) 

             
            
    # point_array=point_array2
    

    
    if len(foot) == 0  and (foot_detect -  previous_foot_detect) != 0  : # change, no detect foot
       
       if foot_detect ==0 :  # left foot
       
           foot = [290, previous_foot[1], 40, 25]  # left reference point (450, 376)
        
       else :  # right foot
       
           foot = [390, previous_foot[1], 40, 25]  # right reference point (900, 376)      
       
    
    elif len(foot) == 0  and (foot_detect - previous_foot_detect) == 0 :
        if foot_detect==0:
            foot = [270, previous_foot[1], 40, 25]
        else:
            foot = [410, previous_foot[1], 40, 25]
            
    
    if rock_num ==0:
        if foot_detect==0:
            point_array=[[int(foot[0]),int(stride_min)-10,40,25,0]]
        else:
            point_array=[[int(foot[0]),int(stride_min)-10,40,25,0]]    
    
    
    distance_weight = distance_from_foot_point(foot[0:2], point_array)
        
    # Create the 2D array - image size
    arr = np.zeros((672, 376))

    # Create a meshgrid of x and y coordinates
    x, y = np.meshgrid(np.arange(672), np.arange(376))
    
    result = []
    
    for i in range(len(point_array)) :
        
        point = point_array[i]

        # Subtract the mean from each coordinate
        x_diff1 = x - point[0]
        y_diff1 = y -point[1]
        
        
        # Calculate the Gaussian probability distribution for each element
        arr_point = np.exp(-0.5 * (np.power(x_diff1, 2) / (point[2] ** 2) + np.power(y_diff1, 2) / (point[3] ** 2)))
        
        # Sum the two distributions to create the mixture
        arr = arr_point
        
        # Make rotated distribution
        arr_rotation =  cv2.getRotationMatrix2D((point[0],point[1]), angle =point[4]*180/(np.pi), scale =5)
        arr_result = cv2.warpAffine(arr,arr_rotation,(672, 376))
        
        # Foot weight scale calcultation
        if distance_weight[i]!=0:
            foot_weight = distance_weight[i]
        else:
            foot_weight=1
        rotated_arr = (1/ foot_weight)*(rotate(arr_result, axes = (0,1), angle=0, reshape=False, mode='nearest'))   # (1+i*const) 값은 거리에 따른 가중치
        
        #rotated_arr =  rotated_arr/np.sum(rotated_arr)  ##
        result.append(rotated_arr)
        
            
        # Combine all the arrays 
        combined_array = np.sum(result, axis=0)

        # Divide the combined array by the number of arrays to get the final array
    if len(point_array)!=0:
        arr_rock = combined_array / len(point_array)
        arr_rock=1 # 에러 나와서 고쳤음 !!!
        arr_rock = (arr_rock/np.sum(arr_rock))
    else:
        arr_rock=np.ones((376,672))

    return arr_rock




def stride_distribution(foot, previous_foot, stride_min, foot_detect, previous_foot_detect,frame_count=1):

    global stride_half
    # stride_min-=30
    stride_half = stride_min
    
    # update stride with stride_min
    
    
    if len(foot)==0: # no detect foot
        
        if foot_detect ==0 :  # left foot
            foot_ref = [290, 376, 40, 25]# left reference point (450, 376)
            foot=foot_ref

        else :  # right foot
            foot_ref =[390, 376, 40, 25] 
            foot=foot_ref
        if foot[1]-previous_foot[1]>0:

            stride_y = stride_half+frame_count*1.5
        else:

            stride_y=stride_half
    else : # detect foot
        if foot[1]-previous_foot[1]>0:
            stride_y = stride_half+frame_count*3
        else:

            stride_y=stride_half

        
            
            
    # print(len(previous_foot))
    # calculation sigma_y
    x_sigma =foot[2]                     # w 보다 2배 크게 하여, 가로 방향으로 표준 편차를 키워 평평하게 만듦
    y_sigma = (stride_y)

    # Create a meshgrid of x and y coordinates
    x, y = np.meshgrid(np.arange(672), np.arange(376))
    
    # Subtract the mean from each coordinate
    x_diff1 = x - foot[0]
    y_diff1 = y - stride_y    ## 빼주는 값은 항상 상수 (보폭 길이 고려하여 결정)

    # Calculate the Gaussian probability distribution for each element

    arr_foot = np.exp(-0.5 * (np.power(x_diff1, 2) / (x_sigma ** 2) + np.power(y_diff1, 2) / ( y_sigma ** 2))) ## y축 sigma는 보폭 사이즈에서 발 끝 쪽 좌표 빼준 값

    
    arr_foot = arr_foot/np.sum(arr_foot)

        
    return arr_foot
    
    

# Final point of maximum probability 
    

def step_point( arr_rock, arr_foot,rock_num):
    if rock_num ==0:
        final_distribution = arr_foot 
        
    else:
        final_distribution = arr_rock * arr_foot


    max_index = np.argmax(final_distribution)
    
    
    y, x = np.unravel_index(max_index, final_distribution.shape)
    
    x_coord = x      
    y_coord = y
    
    return [x_coord, y_coord],final_distribution


def save_numpy_as_image(final_array, frame_count):
    # Create a colormap that maps values between 0 and 1 to a range of colors
    cmap = plt.get_cmap('jet')
    output_folder = "./distribution_save"
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Display the image with the colormap applied
    final_array = (final_array-np.min(final_array))/(np.max(final_array)-np.min(final_array))

    im = ax.imshow(final_array, cmap=cmap)

    # Add a colorbar to the image to show the mapping of values to colors
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set the title and axis labels
    fontprops_title = { 'size': 13, 'weight': 'bold'}
    fontprops_axis= { 'size': 10, 'weight': 'bold'}
    ax.set_title('Final distribution', fontdict=fontprops_title)
    ax.set_xlabel('X axis pixel', fontdict=fontprops_axis)
    ax.set_ylabel('Y axis pixel', fontdict=fontprops_axis)

    # Set output file name with count variable appended
    output_filename = f"image_{frame_count}.png"

    # Construct full output file path
    output_path = f"{output_folder}/{output_filename}"

    # Save the image as a file
    plt.savefig(output_path)


def final_rotated_distribution(final_rock,rock_num) :  # final_rock = [x, y ,w, h, rad]
    if rock_num!=0:
        # Create the 2D array - image size
        arr = np.zeros((672, 376))

        # Create a meshgrid of x and y coordinates
        x, y = np.meshgrid(np.arange(672), np.arange(376))
        
        x_diff1 = x - final_rock[0]
        y_diff1 = y -final_rock[1]

        x_sigma = final_rock[2] *0.6          # w 보다 2배 크게 하여, 가로 방향으로 표준 편차를 키워 평평하게 만듦
        y_sigma = final_rock[3] *0.6

        arr_point = np.exp(-0.5 * (np.power(x_diff1, 2) / (x_sigma ** 2) + np.power(y_diff1, 2) / ( y_sigma ** 2)))

        # Calculate the Gaussian probability distribution for each element
        
        
        # Sum the two distributions to create the mixture
        arr = arr_point
        
        # Make rotated distribution
        arr_rotation =  cv2.getRotationMatrix2D((final_rock[0],final_rock[1]), angle =final_rock[4]*180/(np.pi), scale =2)
        arr_result = cv2.warpAffine(arr,arr_rotation,(672, 376))
        
        result = (rotate(arr_result, axes = (0,1), angle=0, reshape=False, mode='nearest'))   # (1+i*const) 값은 거리에 따른 가중치

    else:
        result = np.ones((376,672))
    return result