import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image
import os
from os import listdir
from os.path import isfile, join

def split_channels(file_, output_file, file_num):
    image = Image.open(file_)
    output_image = Image.open(output_file)
    array = asarray(image)
    out_array = asarray(output_image)


    patch_size = 384

    #channels = ['train_red', 'train_green', 'train_blue']
    channels = ['training_images', 'output_images']
    #color = ['red', 'green', 'blue']
    color = ['full_color']
    for channel in range(1):
        width_after = patch_size
        num = file_num
        #num = 1 
        for width in range(0, image.size[0], patch_size):
            height_after = patch_size
            for height in range(0, image.size[1], patch_size):
                #print(height)
                patch = array[height:height_after,width:width_after, :]
                out_patch = out_array[height:height_after,width:width_after]  
                height_img = patch[:,:,0].shape[0]
                if(height_img % patch_size != 0):
                    height_after+=patch_size
                    continue
                width_img = patch[:,:,0].shape[1] 
                if(height_img != 0 and width_img !=0):
                    if((np.count_nonzero(patch[:,:,0] == 255) / (height_img*width_img)) > 0.01):
                        height_after+=patch_size
                        continue 
                patch = Image.fromarray(patch)
                output_patch = Image.fromarray(out_patch)
                patch.save(channels[channel] + '/' + color[channel] + '_training_' + str(num) + '.TIF')
                output_patch.save(channels[channel+1] + '/' + 'output_' + str(num) + '.TIF')
                #plt.savefig('../training/train_gt/ground_truth_' + str(num) + '.tif')
                num+=1
                if(num%1000 == 0):
                    print(num)
                height_after+=patch_size
                if(height_after > image.size[1]):
                    break

            width_after+=patch_size
            if(width_after > image.size[0]):
                break
    return num


#full_color_images_folder = 'full_color_images/'
full_color_images_folder = 'input/'
output_images_folder = 'output/'
files = [f for f in listdir(full_color_images_folder) if isfile(join(full_color_images_folder, f))]
files_ = ['img-' + str(num) + '.tiff' for num in range(1,len(files)+1)]
file_num = 1
#folders = ['train_red', 'train_green', 'train_blue']
folders = ['training_images', 'output_images']
files.sort()
print(files_)

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

for file_ in files_:
    training_img = full_color_images_folder + file_
    output_img = output_images_folder + file_
    file_num = split_channels(training_img, output_img, file_num) 
    #if(file_num > 445):
    #    break

