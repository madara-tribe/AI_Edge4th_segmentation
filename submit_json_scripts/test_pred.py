import cv2
import os
import numpy as np
from keras import optimizers
## Import usual libraries
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from keras import backend
from datetime import datetime #DB
from keras.utils import plot_model #DB
from keras.preprocessing.image import ImageDataGenerator #DB
import gc #DB
from tqdm import tqdm
from config import fcn_config as cfg
from config import fcn8_cnn as cnn

from config import unet as unet
import pandas as pd
warnings.filterwarnings("ignore")



HEIGHT = 224
WIDTH  = 224
N_CLASSES = 11

######################################################################
# model a
######################################################################

model_type = 2
if (model_type==1):
        model = unet.UNET_v1(N_CLASSES, HEIGHT, WIDTH)
elif (model_type==2):
        model = unet.UNET_v2(N_CLASSES, HEIGHT, WIDTH)
else:
        model = unet.UNET_v3(N_CLASSES, HEIGHT, WIDTH)

checkpoint_dir = "ep1000_trained_unet_model2_224x224.hdf5" 
model.load_weights(checkpoint_dir)
model.summary()

FINE_N_CLASSES = 5
def create_fintune_model(model):
    IMAGE_ORDERING =  "channels_last"
    nClasses = 5
    inputs_ = model.inputs
    dense = model.get_layer(index=-2).output
    c10 = Conv2D(filters=nClasses, kernel_size=1, data_format=IMAGE_ORDERING, activation="sigmoid")(dense)
    models = Model(inputs=inputs_, outputs=c10)
    return models

finetune_model = create_fintune_model(model)
checkpoint_dir = "keras_model/ep500_trained_unet_model2_224x224.hdf5"
finetune_model.load_weights(checkpoint_dir)
#finetune_model.load_weights(ck)
finetune_model.summary()
#sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
#finetune_model.compile(loss='categorical_crossentropy',
 #             optimizer=sgd,
          #    metrics=['accuracy'])

dir_train_img='seg_test_images'
# load training images
train_images = os.listdir(dir_train_img)
train_images.sort()
X_test = []
for im in tqdm(train_images):
    X_test.append(cnn.NormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT ))

X_test = np.array(X_test)
print(X_test.shape)




print("\nnow computing IoU over testing data set:")
y_pred1 = finetune_model.predict(X_test)
y_pred1_i = np.argmax(y_pred1, axis=3)

print(X_test.shape, y_pred1_i.shape)
print(np.unique(y_pred1_i))
import numpy as np
np.save('test_prediction', y_pred1_i)





