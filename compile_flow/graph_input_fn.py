import cv2
import os
import numpy as np

#from keras.preprocessing.image import img_to_array
#from config import fcn_config as cfg
#from config import fcn8_cnn as cnn
def clahe(bgr):
    #plt.imshow(bgr),plt.show()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    #plt.imshow(lab),plt.show()
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def contrast(img, gamma = 0.5):
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')

    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
    return cv2.LUT(img, gamma_cvt)

def PArr(path):
    NORM_FACTOR = 255
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (640, 400), interpolation=cv2.INTER_NEAREST)
    if img.mean()<80:
        img = clahe(img)
    img = contrast(img, gamma = 1.3)
    img = img.astype(np.float32)
    img = img/NORM_FACTOR
    return img


def NormalizeImageArr(path):
    NORM_FACTOR = 255
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (640, 400), interpolation=cv2.INTER_NEAREST)
    if img.mean()<80:
        img = clahe(img)
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
      image3 = image2[:, ::-1]
      im4 = PArr(image_path)
      #image2 = image2.reshape((image2.shape[0], image2.shape[1], 3))
      images.append(image2)
      images.append(image3)
      images.append(im4)
  return {"input_1": images}

#######################################################

def main():
  calib_input()


if __name__ == "__main__":
    main()