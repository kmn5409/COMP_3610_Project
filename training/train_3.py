import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join



img = cv2.imread('training_images/full_color_training_8088.TIF' ,0)
blur = cv2.bilateralFilter(img,10,90,90)
edges = cv2.Canny(blur,100,200)
img = Image.fromarray(edges)

plt.imshow(img)
plt.show()
'''
full_color_images_folder = 'training_images/'
files = [f for f in listdir(full_color_images_folder) if isfile(join(full_color_images_folder, f))]
num = 0
for f in files:
    if(num%1000 == 0):
        print(num)
    img = cv2.imread('training_images/' + f,0)
    edges = cv2.Canny(img,100,200)
    img = Image.fromarray(edges)
    num+=1
    img.save('training_edge_images/edges_' + str(f.split('_')[3]) + '.TIF')
    
    #print(f)
'''

'''
'''
#plt.imsave('training_edge_images/1.TIF',edges,cmap='gray')
#plt.imshow(edges,cmap='gray')
#plt.show()
'''
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
'''
