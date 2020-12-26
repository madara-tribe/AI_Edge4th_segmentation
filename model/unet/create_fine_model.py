import tensorflow as tf
import sys, time, warnings
import keras as keras
from keras.models import *
from keras.layers import *
#from keras.utils import plot_model #DB
warnings.filterwarnings("ignore")

IMAGE_ORDERING =  "channels_last"

def fine_model(premodel, nClasses):
    inputs_ = premodel.inputs
    x = premodel.get_layer(index=-5).output
    o = Conv2DTranspose(nClasses, kernel_size=(1,1), strides=(1,1), use_bias=False)(o)
    o = (Activation("softmax"))(o)
    models = Model(inputs=inputs_, outputs=o)
    return models
