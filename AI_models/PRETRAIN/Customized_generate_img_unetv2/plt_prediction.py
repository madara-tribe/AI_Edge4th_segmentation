import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from datetime import datetime
from generate_unet import UNET_v2
HEIGHT = 304
WIDTH  = 480 
print(HEIGHT, WIDTH)
FINE_N_CLASSES = 3


weight_path='keras_model/ep102_trained_unet_model2_480x304.hdf5'
model_type = 2
model = UNET_v2(FINE_N_CLASSES, HEIGHT, WIDTH)
model.load_weights(weight_path)
model.summary()



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


path='../finetune_4cls/4cls/'
dir_train_img=path+'train'
dir_train_seg=path+'anno'
dir_valid_img = path+'val'
dir_valid_seg = path+'val_anno'

def flip_image(X, Y):
    Xflip =np.array([img[::-1, ::-1] for img in X])
    Yflip =np.array([img[::-1, ::-1] for img in Y])
    return Xflip, Yflip



# load training images
train_images = os.listdir(dir_train_img)
train_images.sort()
train_segmentations  = os.listdir(dir_train_seg)
train_segmentations.sort()
X_train, Y_train=[], []

for im , seg in tqdm(zip(train_images[:10],train_segmentations[:10])):
    X_train.append(NormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT))
    Y_train.append(give_color_to_seg_img(os.path.join(dir_train_seg,seg), WIDTH, HEIGHT, colormaps))


print('train load')
augX, augY = flip_image(X_train, Y_train)
#udX, udY = flipud_image(X_train, Y_train)
#lrX, lrY = fliplr_image(X_train, Y_train)
#rotX, rotY = RotationAugment(
X_train = np.vstack([X_train, augX])
Y_train = np.vstack([Y_train, augY])

print(X_train.shape, Y_train.shape)
print(X_train.max(), X_train.min())



print('load validation images')
valid_images = os.listdir(dir_valid_img)
valid_images.sort()
valid_segmentations = os.listdir(dir_valid_seg)
valid_segmentations.sort()
X_valid, Y_valid = [], []

for im , seg in tqdm(zip(valid_images[:10],valid_segmentations[:10])):
    X_valid.append(NormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT))
    Y_valid.append(give_color_to_seg_img(os.path.join(dir_valid_seg,seg), WIDTH, HEIGHT, colormaps))

X_valid, Y_valid = np.array(X_valid), np.array(Y_valid)
print(X_valid.shape,Y_valid.shape)
print(X_valid.max(),X_valid.min())


plt.imshow(X_train[0]),plt.show()
plt.imshow(Y_train[0]),plt.show()
preds = model.predict(X_train)
print(preds.shape)

plt.imshow(preds[0]),plt.show()
plt.imshow(X_train[0]),plt.show()
plt.imshow(Y_train[0]),plt.show()
