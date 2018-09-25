import numpy as np
import pickle
from scipy.stats import norm, multivariate_normal
from collections import Counter
import random

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


def all_means(img_array,label_array):

    #This is stupid?
    mean_list = []
    for i in range(0,10):
        mean_list.append([])
        for j in range(0,3):
            mean_list[i].append([])


    for i in range(0,len(img_array)):

        img_class = label_array[i]

        mean_list[img_class][0].append(img_array[i][0])
        mean_list[img_class][1].append(img_array[i][1])
        mean_list[img_class][2].append(img_array[i][2])


    mean_arr = np.asarray(mean_list)

    return mean_arr


#class_means = 10,sub_img,samples*[mr,mg,mb]
def feature_means(class_means):


    class_color_means2 = np.ones([10,len(class_means[0]),3],dtype=float)

    for cl in range(0,10):
        for sub_img in range(0,len(class_means[0])):

            class_color_means2[cl][sub_img][0] = np.std(class_means[cl][sub_img][0])
            class_color_means2[cl][sub_img][1] = np.std(class_means[cl][sub_img][1])
            class_color_means2[cl][sub_img][2] = np.std(class_means[cl][sub_img][2])

    return class_color_means2


def cifar_10_features_simple(img_array):

    mean_values = np.ones([len(img_array),3],dtype=float)

    for i in range(0,len(img_array)):

        mean_values[i] = np.mean(img_array[i], axis=1)

    return mean_values


#img_arr = 50000*3*1024
def cifar_10_features(img_arr,N):

    nsub_img = int(1024/(N**2))
    nsub_pixels = int(N**2)
    colors = []

    for pic in range(0,len(img_arr)):

        colors.append([])
        for sub_num in range(0,nsub_img):

            colors[pic].append([])
            for color in range(0,3):
                colors[pic][sub_num].append([])
                colors[pic][sub_num][color].append(np.mean(img_arr[pic][color][sub_num*nsub_pixels:(sub_num+1)*nsub_pixels]))


    sub_mean_colors = np.asarray(colors)

    return sub_mean_colors

                    #[images,sub,color]
def class_sub_means(sub_img_means,labels):

    #stupid?
    class_means = []
    for cl in range(0,10):
        class_means.append([])
        for sub_img in range(len(sub_img_means[0])):
            class_means[cl].append([])


    for i in range(0,len(sub_img_means)):
        class_ = labels[i]

        for sub_img in range(0,len(sub_img_means[0])):
            class_means[class_][sub_img].append(sub_img_means[i][sub_img])


    class_means = np.asarray(class_means)


    return class_means


def class_sub_means2(sub_img_means,labels):


    class_means = []
    for cl in range(0,10):
        class_means.append([])
        for sub_img in range(0,len(sub_img_means[0])):
            class_means[cl].append([])
            for color in range(0,3):
                class_means[cl][sub_img].append([])


    for i in range(0,len(sub_img_means)):
        cl = labels[i]
        for sub_img in range(0,len(sub_img_means[0])):
            for color in range(0,3):

                class_means[cl][sub_img][color].append(float(sub_img_means[i][sub_img][color]))


    class_means = np.asarray(class_means)

    return class_means


def cifar_bayes_learn(img_arr,N,labels):


    #sub_img_means = (samples,sub_imgs,colors)
    #vector has all the color mean values for every picture
    sub_img_means = cifar_10_features(img_arr,N)

    #class_means = (10,sub_img,samples*[mr,mg,mb]
    #class_means = class_sub_means(sub_img_means,labels)


    #class_means2 = (10,sub_img,3,samples*mcolor)
    class_means2 = class_sub_means2(sub_img_means,labels)


    #_, counts = np.unique(labels[0:1000], return_counts=True)


    p = 1/(10*(1024/N**2))
    mu = feature_means(class_means2)


    #SIG(10,sub_images,3,3)
    SIG = calc_cov(class_means2)

    return mu, SIG, p


def calc_cov(class_means):

    SIG = []

    for cl in range(0,len(class_means)):
        SIG.append([])
        for sub_img in range(0,len(class_means[0])):
            SIG[cl].append([])
            #for sample in range(0,len(class_means[cl][sub_img])):
            red = np.asarray(class_means[cl][sub_img][0])
            green = np.asarray(class_means[cl][sub_img][1])
            blue = np.asarray(class_means[cl][sub_img][2])
            cov_array = np.array([red,green,blue])

            SIG[cl][sub_img].append(np.cov(cov_array))

    SIG = np.asarray(SIG)

    return SIG

def classify(test_img_arr,N,mu,SIG,p):

    prob_total = 1.0
    c = []
    class_ = 0

    test_means = cifar_10_features(test_img_arr,N)
    test_means = test_means.reshape([len(test_means),len(test_means[0]),-1])


    for i in range(0,len(test_means)):
        print("Picture: ",i)
        last_prob = 0.0
        c.append([])

        for test_sub in range(0,len(test_means[0])):

            img = test_means[i][test_sub]

            for cl in range(0,10):
                for sub in range(0,len(test_means[0])):

                    probability = multivariate_normal.pdf(img,mu[cl][sub],SIG[cl][sub][0])

                    if probability > last_prob:

                        last_prob = probability
                        class_ = cl


            c[i].append(class_)


    c2 = []
    for i in range(0,len(c)):

        counter = Counter(c[i])
        maxV = max(counter.values())
        jaa = max(counter, key=counter.get)
        choice = random.choice(
            [key for key, value in counter.items() if value == maxV])

        c2.append(choice)
#        cl_old = 0
#        class_value = 0
#
#        for classes in range(0,10):
#            cl = c[i].count(classes)
#
#            if cl > cl_old:
#                class_value = classes
#                cl_old = cl
#
#        print(c[i], "appended", class_value)
#        input("moot")
#        c2.append(class_value)

    return c2

def cifar_10_eval(pred,gt):

    difference = np.subtract(pred,gt)

    percent = ((len(pred)-np.count_nonzero(difference))/len(pred))*100
    format(percent, ".2f")

    print("Correct classifications: " ,percent,"%")


def main():

    print("get data")
    raw_train_img, train_labels = get_train_data()

    raw_test_data = get_test_data()

    # (50000,3,1024) (50000,)   (10000,3,1024)  (10000,)
    tr_img_arr, tr_label_arr, test_img_arr, test_label_arr = list2array(
        raw_train_img,
        train_labels, raw_test_data[0],
        raw_test_data[1])

    N = 16


    print("Calculating parameters")
    mu,SIG,p = cifar_bayes_learn(tr_img_arr[0:20000],N,tr_label_arr)

    print("Start classifying")
    c = classify(test_img_arr[0:100],N,mu,SIG,p)

    print("Evaluate")
    cifar_10_eval(c,test_label_arr[0:100])


main()
