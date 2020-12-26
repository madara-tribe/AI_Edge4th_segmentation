
import cv2, os
import numpy as np
import random
import tensorflow as tf
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from config import fcn_config as cfg #DB

def salt(img):
    row,col,ch = img.shape
    s_vs_p = 0.5
    amount = 0.004
    sp_img = img.copy()
    # pepper
    num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in img.shape]
    sp_img[coords[:-1]] = (0,0,0)
    return sp_img
	
	
def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    #img = img.astype(np.uint8)
    return img

def contrast(img, gamma = 0.5):
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')
 
    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
    return cv2.LUT(img, gamma_cvt)

def sift_angle(image, y_move_ratio=0, x_move_ratio=0, angle_ratio=float(np.pi/60)):
    h, w, _ = np.shape(image)
    size = tuple(np.array([w, h]))
    print(size)
    #np.pi=3.141592653589793
    rad=angle_ratio
    move_x = x_move_ratio
    move_y = w * y_move_ratio

    matrix = [[np.cos(rad), -1 * np.sin(rad), move_x],
                   [np.sin(rad), np.cos(rad), move_y]]

    affine_matrix = np.float32(matrix)
    chage_angle = cv2.warpAffine(image, affine_matrix, size, flags=cv2.INTER_LINEAR)
    return chage_angle


def clahe(bgr):
    #plt.imshow(bgr),plt.show()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    #plt.imshow(lab),plt.show()
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def noise(img):
	ksize = 3
	for j in range(3):
		img[:,:,j]=cv2.medianBlur(img[:, :, j], ksize)
	return img


def NormalizeImageArr(path, H, W):
    NORM_FACTOR = 255
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (H, W), interpolation=cv2.INTER_NEAREST)
    if img.mean()<80:
        img = clahe(img)
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


CLASS_NAMES = ("None", "Sky",
               "Wall",
               "Pole",
               "Road",
               "Sidewalk",
               "Vegetation",
               "Sign",
               "Fence",
               "vehicle",
               "Pedestrian", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'AS', 'BS', 'CS', 'DS', 'Es', 'Fs', 'Gs', 'Hs', 'Is', 'Js', 'Ks', 'Ls', 'Ms', 'Ns', 'Os')

BATCH_SIZE = 32
EPOCHS = 200




# colors for segmented classes
colorB = [128, 232, 70, 156, 153, 153,  30,   0,  35, 152, 180,  60,   0, 142, 70, 100, 100, 230,  32,
          128, 232, 70, 156, 153, 153,  30,   0,  35, 152, 180,  60,   0, 142, 70, 100, 100, 230,  32]
colorG = [ 64,  35, 70, 102, 153, 153, 170, 220, 142, 251, 130,  20,   0,   0,  0,  60,  80,   0,  11,
          128, 244, 70, 102, 190, 153, 250, 220, 107, 152,  70, 220, 255,   0,  0,   0,   0,   0, 119]
colorR = [128, 244, 70, 102, 190, 153, 250, 220, 107, 152,  70, 220, 255,   0,  0,   0,   0,   0, 119,
          64,  35, 70, 102, 153, 153, 170, 220, 142, 251, 130,  20,   0,   0,  0,  60,  80,   0,  11]
CLASS_COLOR = list()
for i in range(0, 34):
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
COLORS = np.array(CLASS_COLOR, dtype="float32")

def give_color_to_seg_img(seg,n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = COLORS #DB
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0]/255.0 ))
        seg_img[:,:,1] += (segc*( colors[c][1]/255.0 ))
        seg_img[:,:,2] += (segc*( colors[c][2]/255.0 ))

    return (seg_img)




def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)
    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi == c))
        FP = np.sum( (Yi != c)&(y_predi == c))
        FN = np.sum( (Yi == c)&(y_predi != c))
        IoU = TP/float(TP + FP + FN)
        #print("class {:02.0f}: #TP={:7.0f}, #FP={:7.0f}, #FN={:7.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        print("class (%2d) %12.12s: #TP=%7.0f, #FP=%7.0f, #FN=%7.0f, IoU=%4.3f" % (c, cfg.CLASS_NAMES[c],TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))
    return


def get_images_and_labels(dir_img, dir_seg, n_classes, width, height ):
    images = os.listdir(dir_img)
    images.sort()
    segmentations  = os.listdir(dir_seg)
    segmentations.sort()
    X = [] # list of input images
    Y = [] # list of ground truth (gt)
    I = [] # List of image Filenames
    S = [] # List of gt filenames
    for im , seg in zip(images,segmentations) :
        img_filename = os.path.join(dir_img, im)
        seg_filename = os.path.join(dir_seg, seg)
        S.append(seg_filename)
        I.append(img_filename)
        X.append( NormalizeImageArr(img_filename)  )
        Y.append( LoadSegmentationArr( seg_filename, n_classes , width , height )  )
    X, Y = np.array(X) , np.array(Y)
    print("X tensor shape: ", X.shape, " Y tensor shape: ", Y.shape)
    return X, Y, I, S

def get_images(dir_img):
    images = os.listdir(dir_img)
    images.sort()
    X = []
    for im in images :
        img_filename = os.path.join(dir_img, im)
        X.append( NormalizeImageArr(img_filename)  )
    X = np.array(X)
    return X

#########################################################################################################
# Visualize the model performance
def visualize_model_performance(X_test, y_pred1_i, y_test1_i, n_classes):

    for k in range(10):

        i = k
        img_is  = (X_test[i] + 1)*(255.0/2)
        seg = y_pred1_i[i]
        segtest = y_test1_i[i]

        fig = plt.figure(figsize=(10,30))
        ax = fig.add_subplot(1,3,1)
        ax.imshow(img_is/255.0)
        ax.set_title("original")

        ax = fig.add_subplot(1,3,2)
        ax.imshow(give_color_to_seg_img(seg,n_classes))
        ax.set_title("predicted class")

        ax = fig.add_subplot(1,3,3)
        ax.imshow(give_color_to_seg_img(segtest,n_classes))
        ax.set_title("true class")

        plt.savefig("rpt/unet_model_performance_image_" + str(i) + ".png")

    plt.show()





def getImageArr( path , width , height ):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width, height ))
    return img

def getSegmentationArr( path , nClasses ,  width , height  ):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width, height ))
    img = img[:, : , 0]
    return img

######################################################################
# plot an image and its 12 classes
######################################################################
import seaborn as sns

def plot_image_with_classes(dir_train_seg_inp, dir_train_img_inp):

    # seaborn has white grid by default so I will get rid of this.
    sns.set_style("whitegrid", {'axes.grid' : False})
    ldseg = np.array(os.listdir(dir_train_seg_inp))

    ## pick the first image file
    fnm = "0016E5_01860.png" #"0001TP_008040.png" #ldseg[0]
    print(fnm)

    ## read in the original image and segmentation labels
    seg2 = cv2.imread(os.path.join(dir_train_seg_inp, fnm) ) # (360, 480, 3)
    img_is = cv2.imread(os.path.join(dir_train_img_inp, fnm) )
    print("seg.shape={}, img_is.shape={}".format(seg2.shape,img_is.shape))

    ## Check the number of labels
    mi, ma = np.min(seg2), np.max(seg2)
    n_classes = ma - mi + 1
    print("minimum seg = {}, maximum seg = {}, Total number of segmentation classes = {}".format(mi,ma, n_classes))

    # plot the image
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(img_is)
    ax.set_title("original image")
    #plt.savefig("../rpt/original_image.png")
    plt.show()

    # plot the classes
    fig = plt.figure(figsize=(15,10))

    for kk in range(mi,ma+1):
        k  = kk #new_classes[kk]
        ax = fig.add_subplot(3,n_classes/3,k+1)
        seg_image = (seg2 == k)*cfg.COLORS[k]/255.0
        ax.imshow(seg_image)
        #ax.set_title("label = {}".format(k))
        ax.set_title("label = {}".format(cfg.CLASS_NAMES[k]))

    plt.savefig("rpt/segmentation_classes.png")
    plt.show()

######################################################################
# Plot some images with classes coded in colors
######################################################################


def plot_some_images(dir_train_seg_inp, dir_train_img_inp):

    ldseg = np.array(os.listdir(dir_train_seg_inp))
    input_height , input_width  = cfg.HEIGHT, cfg.WIDTH
    output_height, output_width = cfg.HEIGHT, cfg.WIDTH

    for fnm in ldseg[np.random.choice(len(ldseg),30,replace=False)]:
        fnm = fnm.split(".")[0]
        seg_filename = os.path.join(dir_train_seg_inp, fnm) + ".png"
        seg3 = cv2.imread(seg_filename) # (360, 480, 3)
        img_filename = os.path.join(dir_train_img_inp, fnm) + ".png"
        img_is = cv2.imread(img_filename)
        seg_img = give_color_to_seg_img(seg3,cfg.NUM_CLASSES)
        print(img_filename, seg_filename)

        fig = plt.figure(figsize=(20,40))
        ax = fig.add_subplot(1,4,1)
        ax.imshow(seg_img)

        ax = fig.add_subplot(1,4,2)
        ax.imshow(img_is/255.0)
        ax.set_title("original image {}".format(img_is.shape[:2]))
        ax = fig.add_subplot(1,4,3)
        ax.imshow(cv2.resize(seg_img,(input_width, input_height)))
        ax = fig.add_subplot(1,4,4)
        tmp_img = cv2.resize(img_is,(output_width, output_height))
        ax.imshow(tmp_img/255.0)
        ax.set_title("resized to {}".format((output_width, output_height)))

    plt.savefig("rpt/segmentation_example_" + fnm + ".png")
    plt.show()


