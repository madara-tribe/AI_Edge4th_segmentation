import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from keras import optimizers
import tensorflow as tf
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras import backend
from datetime import datetime #DB
import gc #DB
from tqdm import tqdm
from utils.data_loader import verify_segmentation_dataset
from config import fcn8_cnn as cnn
from config import aug as au
from unet.more_custom_unet import UNET_v2
from unet.create_fine_model import fine_model
from sklearn.utils import shuffle
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
warnings.filterwarnings("ignore")




BATCH_SIZE = 2
EPOCHS = 70
model_type =2
HEIGHT = int(400)
WIDTH  = int(640)
print(HEIGHT, WIDTH)
N_CLASSES = 34
FINE_N_CLASSES = 5

model = UNET_v2(N_CLASSES, HEIGHT, WIDTH)
model.load_weights('../U1Pretrain/keras_model/ep150pre.hdf5')
model.summary()

FINE_N_CLASSES = 5
f1model = fine_model(model, nClasses=FINE_N_CLASSES)
#f1model.load_weights('keras_model/fine.hdf5')
f1model.summary()




path='../finetune_4cls/4cls/'
dir_train_img=path+'train'
dir_train_seg=path+'anno'
dir_valid_img = path+'val'
dir_valid_seg = path+'val_anno'


# load training images
train_images = os.listdir(dir_train_img)
train_images.sort()
train_segmentations  = os.listdir(dir_train_seg)
train_segmentations.sort()
X_train, Y_train=[], []

for im , seg in tqdm(zip(train_images, train_segmentations)):
    X_train.append(cnn.NormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT),)
    Y_train.append(cnn.LoadSegmentationArr( os.path.join(dir_train_seg,seg), N_CLASSES, WIDTH, HEIGHT)  )

def fliplr_image(X, Y):
    Xflip =np.array([img[:, ::-1] for img in X])
    Yflip =np.array([img[:, ::-1] for img in Y])
    return Xflip, Yflip



print('train load')
#X_train, Y_train = np.array(X_train), np.array(Y_train)
X1, Y1 = fliplr_image(X_train, Y_train)
X_train = np.vstack([X_train, X1])
Y_train = np.vstack([Y_train, Y1])

#X_train, Y_train = shuffle(X_train, Y_train)
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
    X_valid.append(cnn.NormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT) )
    Y_valid.append(cnn.LoadSegmentationArr(os.path.join(dir_valid_seg,seg), N_CLASSES, WIDTH, HEIGHT))


X_valid, Y_valid = np.array(X_valid), np.array(Y_valid)
#Xv1, Yv1 = fliplr_image(X_valid, Y_valid)
#X_valid = np.vstack([X_valid, Xv1])
#Y_valid = np.vstack([Y_valid, Yv1])
print(X_valid.shape,Y_valid.shape)
print(X_valid.max(),X_valid.min())




checkpoint_path = "train_ck/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1)
callback = [cp_callback, reduce_lr]

adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
f1model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


# In[18]:


startTime1 = datetime.now() #DB
hist1 = f1model.fit(X_train,Y_train, validation_data=(X_valid,Y_valid), batch_size=BATCH_SIZE,epochs=EPOCHS, callbacks=callback)
endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Keras training (s): ", diff1.total_seconds())
print("\n")

for key in ["loss", "val_loss"]:
    plt.plot(hist1.history[key],label=key)
plt.legend()

plt.savefig("keras_model/unet_model" + str(model_type) + "_training_curves_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")

f1model.save("keras_model/ep" + str(EPOCHS) + "_trained_unet_model" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5")
print("\nEnd of UNET training\n")
