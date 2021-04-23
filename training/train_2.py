import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression,SGDClassifier
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from datetime import datetime
import os
from os import listdir
from os.path import isfile, join

"""
logic_reg = LogisticRegression(fit_intercept = False, C=1e12, solver='lbfgs')
img = cv2.imread(folder + 'full_color_images/2019-12-26, Landsat 8 USGS, True color.tiff')
height = img.shape[0]
width = img.shape[1]
points = height*width

new_img = np.reshape(img, (points,3))
orig_img = np.reshape(new_img, (height,width,3))

label_img = cv2.imread(folder + 'ground_truth_full_images/ground_truth_1.tif')
label = label_img[:,:,0]
labels = np.reshape(label, (height*width))

model_log = logic_reg.fit(new_img,labels)
pred = model_log.predict(new_img)
pred_img = np.reshape(pred,(height,width,1))

final_img = np.stack((pred_img,pred_img,pred_img),axis=2)
final_img = final_img[:,:,:,0]

plt.imshow(final_img)
plt.show()
"""

def read_input(file_):
    #print(file_)
    img = cv2.imread(folder + file_)
    new_img = np.reshape(img, (points,3))    
    #print(new_img.shape)


    return new_img
    #X_train = np.concatenate((x,y))
    #print(new_img.shape)

def read_labels(file_):
    label_img = cv2.imread('output_images/' + file_)
    #print(label_img.shape)
    label = label_img[:,:,0]
    #print(label.shape)
    labels = np.reshape(label, (height*width))
    return labels

def to_string(files,i):
    return files['name'][i].split('_')[1]

start_time = datetime.now()
print(str(start_time))


#fold = '5'
#folder = '../Training/k_fold/'
folder = 'training_images/'
#files = pd.read_csv(fold + '_fold_train.csv')
files = [f for f in listdir(folder) if isfile(join(folder, f))]
files = ['full_color_training_' + str(num) + '.TIF' for num in range(1,len(files)+1)]
output_files = ['output_' + str(num) + '.TIF' for num in range(1,len(files)+1)]
#print(files)

#print(files)

#print('Fold ' + fold)
height = 384
width = 384
points = height*width
#num = to_string(files,0)
num = 0
X_train = read_input(files[num])
y_train = read_labels(output_files[num])
model = SGDClassifier(loss='log',fit_intercept=False)
#logistic regression
#model = SGDClassifier(loss='hinge', penalty='l2')
#linear svm

values_ = [1001,2001,3001,4001,5001,6001,7001,8001]
for file_ in range(1,len(files)):
    num+=1
    #num = to_string(files,file_)
    #if(file_ == 1001 or file_ == 2001):
    
    
    if(file_ == 6001):
        X_train = read_input(files[num])
        y_train = read_labels(output_files[num])
        break
        #continue
     

    if(file_ in values_):
        X_train = read_input(files[num])
        y_train = read_labels(output_files[num])
        #break
        continue
        
    if(file_%100 == 0):
        #print(num)
        print(file_)
    #print(num)
    X_train = np.concatenate((X_train,read_input(files[num])))
    y_train = np.concatenate((y_train,read_labels(output_files[num])))
    #break
    if( file_%1000==0 and file_!=0):
        print('Done Loading:')
        model_log = model.partial_fit(X_train,y_train,classes=np.unique(y_train))
        #break

print('Model trained on ' + str(file_) + ' patches')
#model = SGDClassifier(loss='log',fit_intercept=False)
model_log = model.partial_fit(X_train,y_train,classes=np.unique(y_train))

print(X_train.shape)
print(y_train.shape)
pickle.dump(model_log, open('logistic_regression_3_' + str(file_) + '_images_' +'.sav', 'wb'))

end_time = datetime.now()
print('\nDuration: {}'.format(end_time - start_time))
print (str(end_time))

