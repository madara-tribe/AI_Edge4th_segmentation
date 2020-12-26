import os
import sys
import shutil
from keras import backend as K
#from tensorflow.keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
from more_custom_unet import UNET_v2
from create_fine_model import fine_model
from config import fcn_config as cfg

model_name = "unet2"
HEIGHT = int(400)
WIDTH  = int(640)
print(HEIGHT, WIDTH)
N_CLASSES = 5
FINE_N_CLASSES = 5
##############################################
# Set up directories
##############################################

KERAS_MODEL_DIR = "keras_model"

WEIGHTS_DIR = KERAS_MODEL_DIR

CHKPT_MODEL_DIR = "train_ck"


# set learning phase for no training: This line must be executed before loading Keras model
K.set_learning_phase(0)

# load weights & architecture into new model

model = UNET_v2(N_CLASSES, HEIGHT, WIDTH)
model.load_weights('../U1Pretrain/keras_model/ep150pre.hdf5')
f1model = fine_model(model, nClasses=FINE_N_CLASSES)
f1model.load_weights('keras_model/ep70fine.hdf5')
f1model.load_weights('keras_model/ep50fine2.hdf5')

##print the CNN structure
#model.summary()

# make list of output node names
output_names=[out.op.name for out in f1model.outputs]

# set up tensorflow saver object
saver = tf.train.Saver()

# fetch the tensorflow session using the Keras backend
sess = K.get_session()

# get the tensorflow session graph
graph_def = sess.graph.as_graph_def()


# Check the input and output name
print ("\n TF input node name:")
print(f1model.inputs)
print ("\n TF output node name:")
print(f1model.outputs)

# write out tensorflow checkpoint & inference graph (from MH's "MNIST classification with TensorFlow and Xilinx DNNDK")
save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "fine.ckpt"))
tf.train.write_graph(graph_def, CHKPT_MODEL_DIR + "/", "infer_graph.pb", as_text=False)

print ("\nFINISHED CREATING TF FILES\n")
