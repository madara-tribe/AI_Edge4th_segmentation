import cv2
import os
import numpy as np

#from keras.preprocessing.image import img_to_array
#from config import fcn_config as cfg
#from config import fcn8_cnn as cnn
NORM_FACTOR = 255

def NormalizeImageArr( path ):
    img = cv2.imread(path, 1)
    #img = cv2.resize(img, (256, 256))
    img = cv2.resize(img, (480, 304))
    img = img.astype(np.float32)
    img = img/NORM_FACTOR
    return img

def get_script_directory():
    path = os.getcwd()
    return path

SCRIPT_DIR = get_script_directory()
calib_image_dir  = "workspace/dataset1/img_calib"
calib_image_list = "workspace/dataset1/calib_list.txt"
print("script running on folder ", SCRIPT_DIR)
print("CALIB DIR ", calib_image_dir)


calib_batch_size = 10

def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  #print(line)
  for index in range(0, calib_batch_size):
      curline = line[iter*calib_batch_size + index]
      #print("iter= ", iter, "index= ", index, "sum= ", int(iter*calib_batch_size + index), "curline= ", curline)
      calib_image_name = curline.strip()

      image_path = os.path.join(calib_image_dir, calib_image_name)
      image2 = NormalizeImageArr(image_path)

      #image2 = image2.reshape((image2.shape[0], image2.shape[1], 3))
      images.append(image2)

  return {"input_1": images}


#######################################################

def main():
  calib_input()


if __name__ == "__main__":
    main()
