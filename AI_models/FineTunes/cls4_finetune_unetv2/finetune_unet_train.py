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
from config import fcn_config as cfg
from config import fcn8_cnn as cnn

from config import unet as unet
from sklearn.utils import shuffle
import pandas as pd
warnings.filterwarnings("ignore")

HEIGHT = 224
WIDTH  = 224 
print(HEIGHT, WIDTH)
N_CLASSES = 11

BATCH_SIZE = 5
EPOCHS = 500


# In[4]:


model_type = 2
if (model_type==1):
        model = unet.UNET_v1(N_CLASSES, HEIGHT, WIDTH)
elif (model_type==2):
        model = unet.UNET_v2(N_CLASSES, HEIGHT, WIDTH)
else:
        model = unet.UNET_v3(N_CLASSES, HEIGHT, WIDTH)
model.load_weights('train_ck/cp-0763.ckpt')
model.summary()


# In[14]:


FINE_N_CLASSES = 5
def create_fintune_model(model, nClasses=FINE_N_CLASSES):
    IMAGE_ORDERING =  "channels_last"
    inputs_ = model.inputs
    dense = model.get_layer(index=-2).output
    c10 = Conv2D(filters=nClasses, kernel_size=1, data_format=IMAGE_ORDERING, activation="sigmoid")(dense)
    models = Model(inputs=inputs_, outputs=c10)
    return models

fmodel = create_fintune_model(model, FINE_N_CLASSES)
fmodel.load_weights('keras_model/ep500_trained_unet_model2_224x224.hdf5')
fmodel.summary()


# In[7]:


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


# In[12]:


# load training images
train_images = os.listdir(dir_train_img)
train_images.sort()
train_segmentations  = os.listdir(dir_train_seg)
train_segmentations.sort()
X_train, Y_train=[], []


for im , seg in tqdm(zip(train_images,train_segmentations)):
    X_train.append(cnn.NormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT))
    Y_train.append(cnn.LoadSegmentationArr( os.path.join(dir_train_seg,seg) , FINE_N_CLASSES, WIDTH, HEIGHT)  )


for im , seg in tqdm(zip(train_images,train_segmentations)):
    X_train.append(cnn.ContrastNormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT, gamma=0.5))
    Y_train.append(cnn.LoadSegmentationArr( os.path.join(dir_train_seg,seg) , FINE_N_CLASSES, WIDTH, HEIGHT)  )


for im , seg in tqdm(zip(train_images,train_segmentations)):
    X_train.append(cnn.ContrastNormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT, gamma=0.3))
    Y_train.append(cnn.LoadSegmentationArr( os.path.join(dir_train_seg,seg) , FINE_N_CLASSES, WIDTH, HEIGHT)  )


X_train, Y_train = np.array(X_train), np.array(Y_train)
#X_train, Y_train = augmentation(X_train, Y_train)

print(X_train.shape,Y_train.shape)
print(X_train.max(), X_train.min())


# In[13]:


# load validation images
valid_images = os.listdir(dir_valid_img)
valid_images.sort()
valid_segmentations  = os.listdir(dir_valid_seg)
valid_segmentations.sort()
X_valid, Y_valid = [], []

for im , seg in tqdm(zip(valid_images,valid_segmentations)):
    X_valid.append( cnn.NormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT) )
    Y_valid.append( cnn.LoadSegmentationArr( os.path.join(dir_valid_seg,seg) , FINE_N_CLASSES, WIDTH, HEIGHT))

for im , seg in tqdm(zip(valid_images,valid_segmentations)):
    X_valid.append(cnn.ContrastNormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT, gamma=0.5) )
    Y_valid.append(cnn.LoadSegmentationArr( os.path.join(dir_valid_seg,seg) , FINE_N_CLASSES, WIDTH, HEIGHT)  )


for im , seg in tqdm(zip(valid_images,valid_segmentations)):
    X_valid.append(cnn.ContrastNormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT, gamma=0.3) )
    Y_valid.append(cnn.LoadSegmentationArr( os.path.join(dir_valid_seg,seg) , FINE_N_CLASSES, WIDTH, HEIGHT)  )

X_valid, Y_valid = np.array(X_valid) , np.array(Y_valid)
#X_valid, Y_valid = augmentation(X_valid, Y_valid)


print(X_valid.shape,Y_valid.shape)
print(X_valid.max(),X_valid.min())


# In[16]:


checkpoint_path = "train_ck/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)

sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
fmodel.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# In[18]:


startTime1 = datetime.now() #DB
hist1 = fmodel.fit(X_train,Y_train, validation_data=(X_valid,Y_valid), batch_size=BATCH_SIZE,epochs=EPOCHS, callbacks=[cp_callback], verbose=2)
endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Keras training (s): ", diff1.total_seconds())
print("\n")

for key in ["loss", "val_loss"]:
    plt.plot(hist1.history[key],label=key)
plt.legend()

plt.savefig("keras_model/unet_model" + str(model_type) + "_training_curves_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")

fmodel.save("keras_model/ep" + str(EPOCHS) + "_trained_unet_model" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5")
print("\nEnd of UNET training\n")
