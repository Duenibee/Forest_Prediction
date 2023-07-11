import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("foot_all.csv")
# df2 = pd.read_csv("y_r.csv")
x=df[0::2]
y=df[1::2]
# plt.plot(x,y,'o')

# print(y['data'][3])
# print(y['data'][1])
# temp=y['data'][1]
a=[]
left=[]
right=[]
color=[255, 0, 0]

# for j in x.index:
#     if x['data'][j]<340:
#         left.append(j)
#     else:
#         right.append(j)

# left_y=[]
# right_y=[]
# print(left)

# for i in left:
#     left_y.append(y['data'][i+1])

# for i in right:
#     right_y.append(y['data'][i+1])

fig=plt.figure(figsize=(10,6))
ax = fig.add_subplot(projection='3d')

# plt.plot(left,left_y,'o',color='r',label='left')
# plt.plot(right,right_y,'o',label='right')
t=range(len(x))
# ax.scatter(x[140:160], t[140:160], y[140:160], marker='o')
# plt.plot(y,'o')
# plt.gca().invert_yaxis()
# plt.gca().xaxis.set_label_position('top') 
# plt.gca().xaxis.tick_top()
plt.xlabel('x',loc='right')
plt.ylabel('frame')



# plt.legend(['left','right'], loc='upper right')
# x_1=range(0,950,50)
# plt.xticks(x_1)
temp=0
for i in range(10,len(x),10):
    ax.clear()
    ax.scatter(0,0,0,'r')
    ax.scatter(0,0,500,'r')
    ax.set_xticks(range(0,400,30))
    ax.set_zticks(range(0,400,30))
    ax.axes.set_xlim3d(left=200, right=400) 
    ax.axes.set_zlim3d(bottom=100, top=400)
    ax.scatter(x[temp:i], range(0,10), y[temp:i], marker='o')
    
    temp=i
    plt.savefig(f'./plot/my_plot{i}.png')

# plt.show()
