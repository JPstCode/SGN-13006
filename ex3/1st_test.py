import numpy as np
import pickle
from scipy.stats import norm, multivariate_normal

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

#returns all the mean values of given images 10*3*5000
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

#returns an array for mean color values for every image
def cifar_10_features(img_array):

    mean_values = np.ones([len(img_array),3],dtype=float)

    for i in range(0,len(img_array)):

        mean_values[i] = np.mean(img_array[i], axis=1)

    return mean_values


def feature_means(class_means,counts):

    feature_means_ = np.zeros([10,3],dtype=float)

    #add all image feature means
    for i in range(0,len(class_means)):

        occurance = counts[i]
        feature_means_[i][0] = np.divide(np.sum(class_means[i][0]),occurance)
        feature_means_[i][1] = np.divide(np.sum(class_means[i][1]),occurance)
        feature_means_[i][2] = np.divide(np.sum(class_means[i][2]),occurance)

    return feature_means_


def cifar_bayes_learn(img_arr,labels):


    img_means = cifar_10_features(img_arr)
    class_means = all_means(img_means, labels)

    #count how many times different classes occur in labels
    _, counts = np.unique(labels, return_counts=True)

    class_color_means = feature_means(class_means,counts)


    #variance = np.zeros([10,3],dtype=float)
    sigma = np.ones([10, 3], dtype=float)
    p = np.ones([10,1],dtype=float)

    #calculate the standard deviaton of each color in every class
    for row in range(0,10):
        for column in range(0,3):
            sigma[row][column] = np.std(class_means[row][column])
            #variance[row][column] = np.var(class_means[row][column])


    #count the probability for every class
    for i in range(0,len(counts)):
        p[i] = counts[i]/len(img_means)

    return class_color_means, sigma, p


def cifar_classify(img_means,mu,sigma,p):

    prob_total = 1.0
    c = []
    class_ = 0

    for i in range(0,len(img_means)):

        print("pic",i)
        img = img_means[i]
        last_prob = 0.0

        for cl in range(0,10):

            for color in range(0,3):
                probability = norm.pdf(img[color],mu[cl][color],sigma[cl][color])
                prob_total = prob_total*probability

            prob_total = prob_total*p[cl]

            if prob_total > last_prob:
                last_prob = prob_total
                class_ = cl

            prob_total = 1.0


        c.append(class_)

    c = np.asarray(c)

    return c

def classify_multi(img_means,mu,SIG,p):

    prob_total = 1.0
    c = []
    class_ = 0


    for i in range(0,len(img_means)):
        print("picture: ",i)
        img = img_means[i]
        last_prob = 0.0


        for cl in range(0,10):

            probability = multivariate_normal.pdf(img,mu[cl],SIG[cl])

            if probability > last_prob:
                last_prob = probability
                class_ = cl

        c.append(class_)

    c = np.asarray(c)

    return c

def calc_cov(img_arr,labels):

    img_means_arr = cifar_10_features(img_arr)
    class_means = all_means(img_means_arr,labels)

    SIG = []

    for i in range(0,len(class_means)):
        SIG.append(np.cov(class_means[i]))

    SIG = np.asarray(SIG)

    return SIG


def cifar_10_eval(pred,gt):

    difference = np.subtract(pred,gt)

    percent = ((len(pred)-np.count_nonzero(difference))/len(pred))*100
    format(percent, ".2f")

    print("Correct classifications: " ,percent,"%")


def main():

    raw_train_img, train_labels = get_train_data()

    raw_test_data = get_test_data()

   #(50000,3,1024) (50000,)   (10000,3,1024)  (10000,)
    tr_img_arr, tr_label_arr, test_img_arr, test_label_arr = list2array(raw_train_img,
                                                            train_labels, raw_test_data[0],
                                                                        raw_test_data[1])


    test_img_means = cifar_10_features(test_img_arr)



    mu,sigma,p = cifar_bayes_learn(tr_img_arr,tr_label_arr)

    SIG = calc_cov(tr_img_arr,tr_label_arr)

    c = classify_multi(test_img_means,mu,SIG,p)
    c2 = cifar_classify(test_img_means,mu,sigma,p)

    cifar_10_eval(c,test_label_arr)
    cifar_10_eval(c2, test_label_arr)


main()