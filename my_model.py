import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch
import random
from torch.utils.data import DataLoader, Dataset

np.set_printoptions(threshold=np.inf, linewidth=np.inf) 
num_video=20

# 데이터 로드
input_arr=np.load(f"/home/aims/obb_contents/annotation/circle_gt/numpy/input_arr_{num_video}.npy")
input_label=np.load(f"/home/aims/obb_contents/annotation/circle_gt/numpy/input_label_{num_video}.npy")
print(input_arr.shape)
print(input_label.shape)
shape=input_arr.shape

input_arr=input_arr.reshape((shape[0],10,6))

# 데이터 정규화
# Data_Pre.min_max_normalize(input_arr)

# ---------학습------------
#gpu설정
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else 'cpu')
device='cuda'

print("다음 기기로 학습합니다:", device)
random.seed(777)
torch.manual_seed(777)
if device=='cuda':
    torch.cuda.manual_seed_all(777)


class CustomDataset(Dataset): 
  def __init__(self,x_data):
    self.x = x_data

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self,idx):
    print(self.x[idx].shape)

    arr=self.x[idx].reshape(1,10,6)
    x=torch.from_numpy(arr).to(device).float()
    return x

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels=16, in_channels=1 ,kernel_size=(6,3), stride=1),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1) #pooling layer
            )
        
        # self.layer2 = torch.nn.Sequential(
            # torch.nn.Conv2d(in_channels=20, out_channels=256,kernel_size=2, stride=1),
            # torch.nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2) #pooling layer
            # )
        
        self.fc1 = torch.nn.Linear(96, 32)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.layer3 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=1 - self.keep_prob)
            )

        self.fc2 = torch.nn.Linear(32, 16)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.layer4 = torch.nn.Sequential(
            self.fc2,
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=1 - self.keep_prob)
            )

        self.fc3 = torch.nn.Linear(16, 8)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


        self.layer5 = torch.nn.Sequential(
            self.fc3,
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=1 - self.keep_prob)
            )
        
        self.fc4 = torch.nn.Linear(8, 1)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        
    def forward(self, x):
        out = self.layer1(x)
        # out= self.layer2(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC

        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = self.fc4(out)
        return out
def saveModel(net): 
    path = "/home/aims/obb_contents/weights/train_xy_ver_fast/final.pth" 
    torch.save(net.state_dict(), path) 


# model_x = CNN().to(device)
# model_y = CNN().to(device)


# testsets =CustomDataset(input_arr,input_label)
# test_loader= DataLoader(testsets, batch_size=1, shuffle=False)
# model_x.load_state_dict(torch.load(PATH_weight_x))
# model_x.eval()

# model_y.load_state_dict(torch.load(PATH_weight_y))
# model_y.eval()


# batch_size = test_loader.batch_size
# correct = 0
# total = 0
# error1=0
# error2=0

# center_axis=340
# red = (0, 0, 255)
# green = (0, 255, 0)
# sky=(255, 255,0)
# white=(255, 255,255)
# ref_left=(315,340)
# ref_right=(385,340)

# loop_count=1
# loop_count_2=0
# a=list(test_loader)
# with torch.no_grad():
# outputs_x = model_x(a[loop_count_2][0])
# outputs_y = model_y(a[loop_count_2][0])
# # print(a[loop_count-1])
# x=int(outputs_x[:,0])
# y=int(outputs_y[:,0])

# # y=int(outputs[:,1])
# print(x)

# frame=cv2.circle(frame, (x,y), 5, red, -1)
