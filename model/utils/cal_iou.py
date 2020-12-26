import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from keras import optimizers
## Import usual libraries
import tensorflow as tf
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras import backend
from datetime import datetime #DB
import gc #DB
from tqdm import tqdm
from data_loader import verify_segmentation_dataset
from config import fcn8_cnn as cnn
from more_custom_unet import UNET_v2
from create_fine_model import fine_model
from sklearn.utils import shuffle
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
warnings.filterwarnings("ignore")

HEIGHT = int(400)
WIDTH  = int(640)
print(HEIGHT, WIDTH)
N_CLASSES = 5
FINE_N_CLASSES=5

weight_path='keras_model/ep80_trained_unet_model2_640x400.hdf5'
model_type = 2
model = UNET_v2(N_CLASSES, HEIGHT, WIDTH)
model.load_weights('../U1Pretrain/keras_model/ep150pre.hdf5')
model.summary()

fmodel = fine_model(model, nClasses=FINE_N_CLASSES)
fmodel.load_weights('keras_model/ep70fine.hdf5')
fmodel.summary()




path='../finetune_4cls/4cls/'
dir_train_img=path+'train'
dir_train_seg=path+'anno'
dir_valid_img = path+'val'
dir_valid_seg = path+'val_anno'

def verify_datasets(validate=True):
    n_classes=5
    print("Verifying training dataset")
    verified, tr_len, _ = verify_segmentation_dataset(dir_train_img,
                                           dir_train_seg,
                                           n_classes)
    assert verified
    if validate:
        print("Verifying validation dataset")
        verified, val_len, _ = verify_segmentation_dataset(dir_valid_img,
                                               dir_valid_seg,
                                               n_classes)
        assert verified
#verify_datasets(validate=True)


# load training images
train_images = os.listdir(dir_train_img)
train_images.sort()
train_segmentations  = os.listdir(dir_train_seg)
train_segmentations.sort()
X_train, Y_train=[], []

for im , seg in tqdm(zip(train_images,train_segmentations)):
    X_train.append(cnn.NormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT))
    Y_train.append(cnn.LoadSegmentationArr( os.path.join(dir_train_seg,seg) , FINE_N_CLASSES, WIDTH, HEIGHT)  )



print('train load')
X_train, Y_train = np.array(X_train), np.array(Y_train)


print(X_train.shape,Y_train.shape)
print(X_train.max(), X_train.min())
print(Y_train.max(), Y_train.min())

# In[13]:


# load validation images
valid_images = os.listdir(dir_valid_img)
valid_images.sort()
valid_segmentations  = os.listdir(dir_valid_seg)
valid_segmentations.sort()
X_valid, Y_valid = [], []

for im , seg in tqdm(zip(valid_images,valid_segmentations)):
    X_valid.append(cnn.NormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT) )
    Y_valid.append( cnn.LoadSegmentationArr( os.path.join(dir_valid_seg,seg) , FINE_N_CLASSES, WIDTH, HEIGHT))

X_valid, Y_valid = np.array(X_valid), np.array(Y_valid)

print(X_valid.shape,Y_valid.shape)
print(X_valid.max(),X_valid.min())



print("\ncomputing IoU over testing data set:")
X_test =X_train
Y_test = Y_train
y_pred1   = fmodel.predict(X_test, batch_size=2)
y_pred1_i = np.argmax(y_pred1, axis=3)
y_test1_i = np.argmax(Y_test, axis=3)
#print(y_test1_i.shape,y_pred1_i.shape)
cnn.IoU(y_test1_i, y_pred1_i)

print("\nnow computing IoU over validation data set:")
y_pred2 = fmodel.predict(X_valid, batch_size=2)
y_pred2_i = np.argmax(y_pred2, axis=3)
y_test2_i = np.argmax(Y_valid, axis=3)
#print(y_test2_i.shape,y_pred2_i.shape)
cnn.IoU(y_test2_i,y_pred2_i)
UPSCALE=True
cnn.visualize_model_performance(X_valid, y_pred2_i, y_test2_i, FINE_N_CLASSES)
