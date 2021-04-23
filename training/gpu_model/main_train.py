from __future__ import print_function
from sklearn.model_selection import train_test_split
import os
import numpy as np
from utils import ADAMLearningRateTracker
import cloud_net_model
from losses import jacc_coef
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from generators import mybatch_generator_train, mybatch_generator_validation
import pandas as pd
from utils import get_input_image_names
from datetime import datetime

start_time = datetime.now()
print(str(start_time))

def train():
    model = cloud_net_model.model_arch(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.compile(optimizer=Adam(lr=starting_learning_rate), loss=jacc_coef, metrics=[jacc_coef])
    # model.summary()

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=end_learning_rate, verbose=1)
    csv_logger = CSVLogger('unet_2400_images_5_fold_150_epochs.log')

    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_img, train_msk,
                                                                                      test_size=val_ratio,
                                                                                      random_state=42, shuffle=True)

    if train_resume:
        model.load_weights(weights_path)
        print("\nTraining resumed...")
    else:
        print("\nTraining started from scratch... ")

    print("Experiment name: ", experiment_name)
    print("Input image size: ", (in_rows, in_cols))
    print("Number of input spectral bands: ", num_of_channels)
    print("Learning rate: ", starting_learning_rate)
    print("Batch size: ", batch_sz, "\n")

    model.fit_generator(
        generator=mybatch_generator_train(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        steps_per_epoch=np.ceil(len(train_img_split) / batch_sz), epochs=max_num_epochs, verbose=1,
        validation_data=mybatch_generator_validation(list(zip(val_img_split, val_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        validation_steps=np.ceil(len(val_img_split) / batch_sz),
        callbacks=[model_checkpoint, lr_reducer, ADAMLearningRateTracker(end_learning_rate), csv_logger])


#GLOBAL_PATH = 'path to 38-cloud dataset'
#GLOBAL_PATH = '../Cloud_Net/38_Cloud_Training_2'
#GLOBAL_PATH = '../training_images'
GLOBAL_PATH = 'gdrive/My Drive/Research/Road_Detection/Training'
#TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'Training')
TRAIN_FOLDER = GLOBAL_PATH
#TEST_FOLDER = os.path.join(GLOBAL_PATH, 'Test')
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'Training')

in_rows = 192
in_cols = 192
num_of_channels = 3
num_of_classes = 1
#starting_learning_rate = 1e-4
#starting_learning_rate = 6.9999995e-05
starting_learning_rate = 3.4299996e-05
end_learning_rate = 1e-8
max_num_epochs = 1  # just a huge number. The actual training should not be limited by this value
val_ratio = 0.2
patience = 15
decay_factor = 0.7
batch_sz = 6
max_bit = 65535  # maximum gray level in landsat 8 images
experiment_name = "unet_2400_images_100_epochs_5_fold"
weights_path = os.path.join(GLOBAL_PATH, experiment_name + '.h5')
train_resume = False
#train_resume = True

# getting input images names
#train_patches_csv_name = 'training_patches_38-cloud.csv'
#train_patches_csv_name = '5_fold_train.csv'
train_patches_csv_name = 'train.csv'
df_train_img = pd.read_csv(os.path.join(TRAIN_FOLDER,train_patches_csv_name))
train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)

train()

end_time = datetime.now()
print('\nDuration: {}'.format(end_time - start_time))
print (str(end_time))

#Unet
#Duration: 0:21:01.466512
#Duration: 0:21:08.774101
#Duration: 0:21:06.091605
#Duration: 0:21:17.089469
