import numpy as np
import pickle


def unpickle(file):

    with open(file, 'rb') as fo:

        #dict has 4 keys: batch_label, labels, data, filenames
        dict = pickle.load(fo,encoding='bytes')
    return dict


def get_train_data():

    raw_img = []
    labels = []

    for i in range(0,5):
        data = unpickle("data_batch_"+ str(i+1))

        #store data and labels to lists
        raw_img.append(data[b'data'])
        labels.append(data[b'labels'])

    return raw_img, labels



def get_test_data():

    raw_test_data = []

    data = unpickle("test_batch")

    raw_test_data.append(data[b'data'])
    raw_test_data.append(data[b'labels'])

    return raw_test_data


def list2array(train_imgs,train_labels,test_imgs,test_labels):

    raw_tr_imgs = np.array(train_imgs,dtype=int)
    raw_test_imgs = np.array(test_imgs,dtype=int)


    tr_arr_imgs = raw_tr_imgs.reshape([-1,3, 1024])
    test_array_i = raw_test_imgs.reshape([-1,3, 1024])


    tr_arr_labels = np.array(train_labels,np.uint8)
    test_array_l = np.array(test_labels,np.uint8)

    tr_arr_labels = tr_arr_labels.reshape(([-1]))
    test_array_l = test_array_l.reshape(([-1]))

    return tr_arr_imgs, tr_arr_labels, test_array_i, test_array_l


#returns an array for mean color values for every image
def cifar_10_features(img_array):

    mean_values = np.ones([len(img_array),3],dtype=float)

    for i in range(0,len(img_array)):
        mean_values[i] = np.mean(img_array[i], axis=1)

    return mean_values



#forms vector for mean colors for every feature
def cifar_10_feature_means(img_array,label_array):

    feature_means = np.ones([10,4], dtype=float)

    for i in range(0,len(img_array)):

        img_class = int(label_array[i])
        img_r_mean = np.mean(img_array[i][0])
        img_g_mean = np.mean(img_array[i][1])
        img_b_mean = np.mean(img_array[i][2])

        feature_means[img_class][0] = feature_means[img_class][0] + img_r_mean
        feature_means[img_class][1] = feature_means[img_class][1] + img_g_mean
        feature_means[img_class][2] = feature_means[img_class][2] + img_b_mean
        feature_means[img_class][3] = feature_means[img_class][3] + 1



    for row in range(0,len(feature_means)):
        for element in range(0,3):

            feature_means[row][element] = feature_means[row][element] / (feature_means[row][3]-1)


    feature_means = np.delete(feature_means,3,1)

    return feature_means


def cifar_bayes_learn(img_means,labels):


    feature_means = np.ones([10,3],dtype=float)
    p = np.ones([10,1],dtype=float)
    _, counts = np.unique(labels, return_counts=True)


    for i in range(0,len(img_means)):

       img_class = labels[i]
       feature_means[img_class][0] = feature_means[img_class][0] + img_means[i][0]
       feature_means[img_class][1] = feature_means[img_class][1] + img_means[i][1]
       feature_means[img_class][2] = feature_means[img_class][2] + img_means[i][2]


    for i in range(0,len(feature_means)):
        feature_means[i] = np.divide(feature_means[i],counts[i])

    print(feature_means)
    input("sad")

    for i in range(0,len(counts)):
        p[i] = counts[i]/len(img_means)


    return mu, sigma, p


def main():

    raw_train_img, train_labels = get_train_data()

    raw_test_data = get_test_data()

   #(50000,3,1024) (50000,)   (10000,3,1024)  (10000,)
    tr_img_arr, tr_label_arr, test_img_arr, test_label_arr = list2array(raw_train_img,
                                                            train_labels, raw_test_data[0],
                                                                        raw_test_data[1])

    #feature_means = cifar_10_feature_means(tr_img_arr,tr_label_arr)
    #print(feature_means)

    tr_img_means = cifar_10_features(tr_img_arr)
    cifar_bayes_learn(tr_img_means,tr_label_arr)


main()