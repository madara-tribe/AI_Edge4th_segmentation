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
from random_erasing import sift_angle
from generate_unet import UNET_v2
from sklearn.utils import shuffle
import pandas as pd
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
warnings.filterwarnings("ignore")

HEIGHT = int(1216/4)
WIDTH  = int(1920/4) 
print(HEIGHT, WIDTH)
N_CLASSES = 3
FINE_N_CLASSES=3
BATCH_SIZE = 2
EPOCHS = 10
weight_path='keras_model/ep102_trained_unet_model2_480x304.hdf5'

model_type = 2
model = UNET_v2(N_CLASSES, HEIGHT, WIDTH)
model.load_weights(weight_path)
model.summary()

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

colormaps={0:'None', 1:'car', 2:'signal', 3:'pedestrian', 4:'lane'}
colorR = [0, 0,     255, 255, 69]
colorG = [0, 0,     255, 0,   47]
colorB = [0, 255,   0,   0,   142]


CLASS_COLOR = list()
for i in colormaps.keys():
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
RGB_COLORS = np.array(CLASS_COLOR, dtype=np.uint8)
print(RGB_COLORS)


def NormalizeImageArr(path, H, W):
    NORM_FACTOR = 255
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (H, W), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32)
    img = img/NORM_FACTOR
    return img

def LoadSegmentationArr( path , nClasses,  width ,height):
    seg_labels = np.zeros((height, width, nClasses))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, : , 0]
    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)
    return seg_labels
    
def give_color_to_seg_img(path, width, height, colormaps):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    seg = img[:, : , 0]
    seg_img = np.zeros((seg.shape[0],seg.shape[1],3)).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = RGB_COLORS #DB
    for c in colormaps.keys():
        segc = (seg == c) # True or False
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return seg_img.astype(np.float32)/255





# In[12]:


# load training images
train_images = os.listdir(dir_train_img)
train_images.sort()
train_segmentations  = os.listdir(dir_train_seg)
train_segmentations.sort()
X_train, Y_train=[], []
X_train2, Y_train2=[], []
for im , seg in tqdm(zip(train_images,train_segmentations)):
    X_train.append(NormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT))
    Y_train.append(give_color_to_seg_img(os.path.join(dir_train_seg,seg), WIDTH, HEIGHT, colormaps))

def flip_image(X, Y):
    Xflip =np.array([img[::-1] for img in X])
    Yflip =np.array([img[::-1] for img in Y])
    return Xflip, Yflip

def flipud_image(X, Y):
    Xflip =np.array([np.flipud(img) for img in X])
    Yflip =np.array([np.flipud(img) for img in Y])
    return Xflip, Yflip
    
def fliplr_image(X, Y):
    Xflip =np.array([np.fliplr(img) for img in X])
    Yflip =np.array([np.fliplr(img) for img in Y])
    return Xflip, Yflip



print('train load')
X_train, Y_train = np.array(X_train), np.array(Y_train)
X1, Y1 = fliplr_image(X_train, Y_train)
X_train = np.vstack([X_train, X1])
Y_train = np.vstack([Y_train, Y1])
#rotX, rotY = RotationAugment(X_train, Y_train)

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
    X_valid.append(NormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT))
    Y_valid.append(give_color_to_seg_img(os.path.join(dir_valid_seg,seg), WIDTH, HEIGHT, colormaps))
X_valid, Y_valid = np.array(X_valid), np.array(Y_valid)
print(X_valid.shape,Y_valid.shape)
print(X_valid.max(),X_valid.min())



print('call back')
checkpoint_path = "train_ck/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)

def wrap_scheduler(epoch, initial_lr=2e-5, use_warm_up=True):
    if use_warm_up and epoch <= 5:
        return 10**(-2.0+0.4*epoch)*initial_lr
    x = initial_lr
    if epoch >= 60: x /= 10.0
    if epoch >= 85: x /= 10.0
    return x
lr_scheduler = LearningRateScheduler(wrap_scheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
callback = [cp_callback]


print('loss opt')
adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=['accuracy'])


# In[18]:

print('train')
startTime1 = datetime.now() #DB
hist1 = model.fit(X_train,Y_train, validation_data=(X_valid,Y_valid), batch_size=BATCH_SIZE,epochs=EPOCHS, callbacks=callback, verbose=2)
endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Keras training (s): ", diff1.total_seconds())
print("\n")

for key in ["loss", "val_loss"]:
    plt.plot(hist1.history[key],label=key)
plt.legend()

plt.savefig("keras_model/unet_model" + str(model_type) + "_training_curves_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")

model.save("keras_model/ep" + str(EPOCHS) + "_trained_unet_model" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5")
print("\nEnd of UNET training\n")
