from os import listdir
from os.path import isfile, join
import pandas as pd


ground_truth_images_folder = 'output_images/'

files = [f for f in listdir(ground_truth_images_folder) if isfile(join(ground_truth_images_folder, f))]

training_image_names = pd.DataFrame(columns=['name'])
for file_ in range(1, len(files)+1):
    name = str(file_)
    training_image_names = training_image_names.append({'name': name}, ignore_index=True)

print(training_image_names)
training_image_names.to_csv('train.csv',index=False)

