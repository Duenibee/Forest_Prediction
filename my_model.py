import numpy as np
from torch import nn
import torch
import random
from torch.utils.data import Dataset

np.set_printoptions(threshold=np.inf, linewidth=np.inf) 
num_video=20

# 데이터 로드
input_arr=np.load(f"/home/aims/obb_contents/annotation/circle_gt/numpy/input_arr_{num_video}.npy")
input_label=np.load(f"/home/aims/obb_contents/annotation/circle_gt/numpy/input_label_{num_video}.npy")
print(input_arr.shape)
print(input_label.shape)
shape=input_arr.shape

input_arr=input_arr.reshape((shape[0],30,6))

# 데이터 정규화
# Data_Pre.min_max_normalize(input_arr)

# ---------학습------------
#gpu설정
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else 'cpu')
device='cuda'

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

    arr=self.x[idx].reshape(1,30,6)
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
        
        self.fc1 = torch.nn.Linear(736, 32)
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
