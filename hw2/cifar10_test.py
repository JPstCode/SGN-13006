import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

def unpickle(file):

    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict


def convert_imgs(raw_imgs):

    raw = np.array(raw_imgs, dtype= float) /255.0
    rawint = np.array(raw_imgs,dtype=int)

    images = raw.reshape([-1,3,32,32])
    images = images.transpose([0,2,3,1])
    return rawint, images

batch_1 = unpickle('data_batch_1')
#print(batch_1)

images_array = batch_1[b'data']
img_labels = batch_1[b'labels']


empty_img = np.ones((32,32,3),np.uint8)

intimgs, images = convert_imgs(images_array)
ex_img = np.array(intimgs[3],np.uint8)


im_r = ex_img[0:1024].reshape(-1)
im_g = ex_img[1024:2048].reshape(-1)
im_b = ex_img[2048:].reshape(-1)


rows = 32
columns = 32
colors = 3
count = 0

for j in range(0,rows):
    for i in range(0,columns):
        #print(count)
        for color in range(0,colors):

            if color == 0:
                empty_img[i][j][color] = im_b[count]
            elif color == 1:
                empty_img[i][j][color] = im_g[count]
            elif color == 2:
                empty_img[i][j][color] = im_r[count]

        count = count + 1


#plt.imshow(empty_img)
#plt.show()
