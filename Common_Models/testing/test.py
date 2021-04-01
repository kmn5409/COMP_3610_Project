import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd
import os
from os import listdir
from os.path import isfile, join


def read_input(file_):
    #print(file_)
    #img = cv2.imread(folder + 'train_full_color/full_color_training_' + file_ + '.TIF')
    img = cv2.imread(folder + file_)
    new_img = np.reshape(img, (points,3))


    return new_img
    #X_train = np.concatenate((x,y))
    #print(new_img.shape)

#file_name = '../training/logistic_regression_8001_images_' + '.sav'
file_name = '../training/linearsvm_3001_images_' + '.sav'
loaded_model = pickle.load(open(file_name, 'rb'))
#pred = loaded_model.predict(new_img)
#logic_reg = LogisticRegression(fit_intercept = False, C=1e12, solver='lbfgs')
#folder = '../Training/k_fold/'
folder = 'testing_full_color_images/'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
files = ['full_color_testing_' + str(num) + '.TIF' for num in range(1,len(files)+1)]
#files = pd.read_csv(fold + '_fold_test.csv')
height = 384
width = 384
points = height*width
#num = files['name'][0].split('_')[1]
#X_train = read_input(num)

predictions_folder = 'Predictions_linearsvm_3001' + '/'
if not os.path.exists(predictions_folder):
    os.makedirs(predictions_folder)

num = 1

for file_ in files:
    #num = files['name'][file_].split('_')[1]

    #if(file_ % 100 == 0):
    #    print(file_)

    pred = loaded_model.predict(read_input(file_))
    pred_img = np.reshape(pred,(height,width,1))
    
    final_img = np.stack((pred_img,pred_img,pred_img),axis=2)
    final_img = final_img[:,:,:,0]
    
    cv2.imwrite(predictions_folder + 'pred_' + str(num) + '.TIF', final_img)
    num+=1 
