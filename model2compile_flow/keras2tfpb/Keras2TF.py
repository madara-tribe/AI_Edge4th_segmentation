import os
import sys
import shutil
from keras import backend as K
#from tensorflow.keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf

#from config import fcn_config as cfg

model_name = "unet2"

##############################################
# Set up directories
##############################################

KERAS_MODEL_DIR = "keras_model"

WEIGHTS_DIR = KERAS_MODEL_DIR

CHKPT_MODEL_DIR = "train_ck"


# set learning phase for no training: This line must be executed before loading Keras model
K.set_learning_phase(0)

# load weights & architecture into new model
if model_name=="fcn8ups" :
        weights= "ep100_trained_fcn8_224x224.hdf5"
elif model_name=="fcn8" :
        weights= "ep200_trained_fcn8_224x224.hdf5"
elif model_name=="unet1" :
        weights= "ep200_trained_unet_model1_224x224.hdf5"
elif model_name=="unet2" :
        weights= "ep2_trained_unet_model2_224x224.hdf5"
elif model_name=="unet3" :
        weights= "ep2_trained_unet_model3_224x224.hdf5"

print("model name = ", model_name)
filename = os.path.join(WEIGHTS_DIR,weights)

assert os.path.isdir(WEIGHTS_DIR)
assert os.path.isfile(filename)
model = load_model(filename)

##print the CNN structure
#model.summary()

# make list of output node names
output_names=[out.op.name for out in model.outputs]
print(output_names)
# set up tensorflow saver object
saver = tf.train.Saver()

# fetch the tensorflow session using the Keras backend
sess = K.get_session()

# get the tensorflow session graph
graph_def = sess.graph.as_graph_def()


# Check the input and output name
print ("\n TF input node name:")
print(model.inputs)
print ("\n TF output node name:")
print(model.outputs)

# write out tensorflow checkpoint & inference graph (from MH's "MNIST classification with TensorFlow and Xilinx DNNDK")
save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "cp-0002.ckpt"))
tf.train.write_graph(graph_def, CHKPT_MODEL_DIR + "/", "infer_graph.pb", as_text=False)

print ("\nFINISHED CREATING TF FILES\n")
