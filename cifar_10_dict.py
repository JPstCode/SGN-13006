import numpy as np
from matplotlib import pyplot as plt
import pickle
from random import randint


def unpickle(file):

    with open(file, 'rb') as fo:

        #dict has 4 keys: batch_label, labels, data, filenames
        dict = pickle.load(fo,encoding='bytes')
    return dict


#return train images in (5,10000,3072) and train labels (5,10000)
def get_train_data():

    raw_img = []
    labels = []

    for i in range(0,5):
        data = unpickle("data_batch_"+ str(i+1))

        #store data and labels to lists
        raw_img.append(data[b'data'])
        labels.append(data[b'labels'])

    return raw_img, labels

#returns test images in (1,10000,3072) and test labels (1,10000)
def get_test_data():

    raw_test_data= []

    data = unpickle("test_batch")

    raw_test_data.append(data[b'data'])
    raw_test_data.append(data[b'labels'])

    return raw_test_data


def list2array(train_imgs,train_labels,test_imgs,test_labels):

    raw_tr_imgs = np.array(train_imgs,dtype=int)
    raw_test_imgs = np.array(test_imgs,dtype=int)

    tr_arr_imgs = raw_tr_imgs.reshape([-1,3072])
    test_array_i = raw_test_imgs.reshape([-1,3072])


    tr_arr_labels = np.array(train_labels,np.uint8)
    test_array_l = np.array(test_labels,np.uint8)

    tr_arr_labels = tr_arr_labels.reshape(([-1]))
    test_array_l = test_array_l.reshape(([-1]))

    return tr_arr_imgs, tr_arr_labels, test_array_i, test_array_l

#convert images from 3072 -> 32x32x3
def convert_imgs(data):

    raw = np.array(data,dtype=int)

    img = raw.reshape([-1,32,32])
    img = img.transpose([1,2,0])

    return img


def get_labels():

    labels = []

    data = unpickle('batches.meta')
    label_names = data[b'label_names']
    for i in range(0,len(label_names)):
        var = label_names[i]
        label_name = var.decode()
        labels.append(label_name)

    return labels

#data = list containing 3072 pixel values
def plot_image(img):

    plot_img = np.ones((32, 32, 3), np.uint8)

    #display = np.array(data,np.uint8)

    img_red_val = img[0:1024].reshape(-1)
    img_green_val = img[1024:2048].reshape(-1)
    img_blue_val = img[2048:].reshape(-1)

    count = 0
    for row in range(0,32):
        for column in range(0,32):
            for color in range(0,3):

                if color == 0:
                    plot_img[row][column][color] = img_red_val[count]
                elif color == 1:
                    plot_img[row][column][color] = img_green_val[count]
                elif color == 2:
                    plot_img[row][column][color] = img_blue_val[count]

            count = count + 1

    plt.imshow(plot_img)
    plt.show()


def cifar_10_rand(data):

    random_list = []


    for i in range(len(data)):
        random_label = randint(0,9)
        random_list.append(random_label)

    return random_list


def cifar_10_eval(pred,gt):

    difference = np.subtract(pred,gt)

    percent = ((len(pred)-np.count_nonzero(difference))/len(pred))*100
    format(percent, ".2f")

    print("Correct classifications: " ,percent,"%")


def cifar_10_1NN_for(test,trimg_array,trlabel):


    label = []
    pred_label = 0


    for i in range(0,len(test)):
        prev_result = np.sum(test[i])

        for j in range(0,len(trimg_array)):
            difference = np.abs(np.subtract(test[i],trimg_array[j]))
            result = np.sum(difference)

            if result < prev_result:
                pred_label = trlabel[j]
                prev_result = result


            if j == (len(trimg_array)-1):
                print(len(test),": ", i)
                label.append(pred_label)

    return label

def main():

    raw_train_img, train_labels = get_train_data()
    raw_test_data = get_test_data()

    tr_img_arr, tr_label_arr, test_img_arr, test_label_arr = list2array(raw_train_img,
                                                            train_labels, raw_test_data[0],
                                                                        raw_test_data[1])

    #labels = get_labels()


    #randoms = cifar_10_rand(test_img_arr)
    #cifar_10_eval(randoms,test_label_arr)


    #plot_image(tr_img_array[950])

    #print("train: ")
    #predict_train = cifar_10_1NN_for(tr_img_arr[42000:43000],tr_img_arr[42000:43000],tr_label_arr[42000:43000])

    #print(np.sum(np.subtract(predict_train,tr_label_arr[0:1000])))
    #cifar_10_eval(predict_train,tr_label_arr[42000:43000])

    #print("test: ")
    #predict = cifar_10_1NN_for(test_img_arr[200:300],tr_img_arr[3000:10000],tr_label_arr[3000:10000])

    #cifar_10_eval(predict,test_label_arr[200:300])

main()
