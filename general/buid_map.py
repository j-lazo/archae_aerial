import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2

img = cv2.imread('../bunny.jpg')
print(np.shape(img))
img_r = img[:,:,0]
img_g = img[:,:,1]
img_b = img[:,:,2]
#plt.figure()
#plt.imshow(img)
#plt.figure()
#img[:,:,0] = np.multiply(img[:,:,0],0.2)
#img[:,:,1] = np.multiply(img[:,:,1],0.9)
#img[:,:,2] = np.multiply(img[:,:,2],0.2)
#plt.imshow(img)

canvas = np.ones([10,10,3])

directory = os.getcwd()
only_images = [f for f in os.listdir(directory) if f[-4:] =='.jpg']

vis = []
maxes = []
for i in range(18):
    vis.append(cv2.imread("".join([directory, '/', only_images[14*i]])))
    for cont, image in enumerate(only_images[i*14+1:i*14+13]):
        ima = cv2.imread("".join([directory, '/', image]))
        vis[i] = np.concatenate((vis[i], ima), axis=1)
    maxes.append((max(np.shape(vis[i]))))

img_show = vis[1]
shape = np.shape(img_show)
if shape[1] != max(maxes):
    canvas = np.ones([200, max(maxes) - shape[1], 3])
    img_show = np.concatenate((img_show, canvas), axis=1)

for element in vis[2:]:
    shape = np.shape(element)

    if shape!= (200, max(maxes), 3):
        canvas = np.ones([200, max(maxes) - shape[1], 3])
        element = np.concatenate((element, canvas ), axis=1)
    print(np.shape(img_show), np.shape(element))
    img_show = np.concatenate((img_show, element), axis=0)


plt.figure()
plt.imshow(img_show)
    

file = open('results', 'r') 
A = np.zeros(270)
r = re.compile('.*:.*')
for count, line in enumerate(file):
    if line[:7] =='tiling_':
        tile = line[7:] 
        tile = int(re.sub('\.jpg$', '', tile))
    
    if count%2 != 0: 
        if line[:3] == 'not':
            A[tile] = -1*float(line[16:])
        elif line[:3] != 'not':
            A[tile] = 1*float(line[12:])

I = np.reshape(A, (18,15))
fig = plt.figure()
plt.imshow(I)

plt.show()
