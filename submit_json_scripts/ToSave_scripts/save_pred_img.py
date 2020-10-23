import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

#RGB
#0 (0,0,0) None
#1 (0,0,255) car
# 2 (0, 255, 255) signal
# 3(255, 0, 0) pedestrian
#4 (142,47,69) lane

colormaps={0:'None', 1:'car', 2:'signal', 3:'pedestrian', 4:'lane'}
colorR = [0, 0,     255, 255, 69]
colorG = [0, 0,     255, 0,   47]
colorB = [0, 255,   0,   0,   142]


CLASS_COLOR = list()
for i in colormaps.keys():
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
RGB_COLORS = np.array(CLASS_COLOR, dtype=np.uint8)
print(RGB_COLORS)

def give_color_to_seg_img(seg, colormaps):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros((seg.shape[0],seg.shape[1],3)).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = RGB_COLORS #DB
    for c in colormaps.keys():
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return seg_img.astype(np.uint8)

# (1216, 1936, 3)
H=1216
W=1936
path = '/Users/hagi/downloads/VITIS_SEG_results/submit_json_scripts/test_prediction.npy'
masks = np.load(path)
masks = [cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) for mask in masks]
print(len(masks), masks[1].shape, np.unique(masks[1]))



color_pred = [give_color_to_seg_img(mask, colormaps) for mask in masks]
plt.imshow(color_pred[1]),plt.show()
print(np.unique(color_pred), color_pred[1].max(), color_pred[1].min())


f = open('test_name_list.txt')
names = f.readlines()  # ファイル終端まで全て読んだデータを返す
f.close()
print(len(names))


save_path = '/Users/hagi/desktop/S'
for name, im in zip(names, color_pred):
    save_name = name.split('.')[0]
    #im = im.astype(np.int8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #plt.imshow(im),plt.show()
    print(np.unique(im), save_name, im.shape)
    cv2.imwrite(os.path.join(save_path, save_name+'.png'), im)


