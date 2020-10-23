import cv2
import os
import numpy as np
HEIGHT = int(1216/4)
WIDTH  = int(1920/4)
N_CLASSES = 5

def sift_angle(image, y_move_ratio=0, x_move_ratio=0, angle_ratio=float(np.pi/60)):
    h, w, _ = np.shape(image)
    size = tuple(np.array([w, h]))
    #np.pi=3.141592653589793
    rad=angle_ratio
    move_x = x_move_ratio
    move_y = w * y_move_ratio

    matrix = [[np.cos(rad), -1 * np.sin(rad), move_x],
                   [np.sin(rad), np.cos(rad), move_y]]

    affine_matrix = np.float32(matrix)
    chage_angle = cv2.warpAffine(image, affine_matrix, size, flags=cv2.INTER_LINEAR)
    return chage_angle



def RandomErasing(x):
      image = np.zeros_like(x)
      size = x.shape[2]
      offset = np.random.randint(-4, 5, size=(2,))
      mirror = np.random.randint(2)
      remove = np.random.randint(2)
      top, left = offset
      left = max(0, left)
      top = max(0, top)
      right = min(size, left + size)
      bottom = min(size, top + size)
      if mirror > 0:
          x = x[:,:,::-1]
      image[:,size-bottom:size-top,size-right:size-left] = x[:,top:bottom,left:right]
      if remove > 0:
          while True:
              s = np.random.uniform(0.02, 0.4) * size * size
              r = np.random.uniform(-np.log(3.0), np.log(3.0))
              r = np.exp(r)
              w = int(np.sqrt(s / r))
              h = int(np.sqrt(s * r))
              left = np.random.randint(0, size)
              top = np.random.randint(0, size)
              if left + w < size and top + h < size:
                  break
          c = np.random.randint(-128, 128)
          image[:, top:top + h, left:left + w] = c
      return image


def flip_rotate_image(image):
    # flip
    flip =np.array([cv2.flip(img, 1) for img in image])
    # flip_up
    flip_up =np.array([cv2.flip(img, 0) for img in image])
    #print(flip_up.shape, np.unique(flip_up[0]))
    #plt.imshow(flip_up[0], "gray"),plt.show()

    # flip_up_lr
    flip_up_lr =np.array([cv2.flip(img, -1) for img in image])

    # right_90rotate
    right_90rotate = np.array([np.rot90(img) for img in image])

    # left_90rotate
    left_90rotate = np.array([np.rot90(img, 3) for img in image])

    return image, flip, flip_up, flip_up_lr, right_90rotate, left_90rotate


def augmentation(image, annos):

    img, flip_img, flip_up_img, flip_up_lr_img, right_90rotate_img, left_90rotate_img = flip_rotate_image(image)

    ano, flip_anno, flip_up_anno, flip_up_lr_anno, right_90rotate_anno, left_90rotate_anno = flip_rotate_image(annos)
    #plt.imshow(annos[0], "gray"),plt.show()
    #eannos = RandomErasing(annos)
    #eflip = RandomErasing(flip)
    train_images = np.vstack([img, flip_img, flip_up_img, flip_up_lr_img,
                     right_90rotate_img, left_90rotate_img])
    train_annos = np.vstack([ano, flip_anno, flip_up_anno, flip_up_lr_anno,
                     right_90rotate_anno, left_90rotate_anno])
    return flip_img, flip_anno





