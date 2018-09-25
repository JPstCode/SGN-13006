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



def feature_means(class_means,counts):

    feature_means_ = np.zeros([10,len(class_means[0]),3],dtype=float)

    #add all image feature means
    for i in range(0,len(class_means)):

        occurance = counts[i]

        for sub_img in range(0,len(class_means[0])):

            print(class_means[i][sub_img][0])
            #feature_means_[i][sub_img][0] =
            #class_means[i][sub_img]


            input("s")

    print(feature_means_.shape)
    input("aha")
    return feature_means_




#img_arr = 50000*3*1024
def cifar_10_features(img_arr,N):

    nsub_img = int(1024/(N**2))
    nsub_pixels = int(N**2)
    sub_features = np.zeros([nsub_img,3,1],dtype=float)
    sub_ft_list = []
    colors = []



    for pic in range(0,len(img_arr)):

        colors.append([])
        for sub_num in range(0,nsub_img):

            colors[pic].append([])
            for color in range(0,3):
                colors[pic][sub_num].append([])
                colors[pic][sub_num][color].append(np.mean(img_arr[pic][color][sub_num*nsub_pixels:(sub_num+1)*nsub_pixels]))


    sub_mean_colors = np.asarray(colors)
    print(sub_mean_colors.shape)
    input("nih")
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


def cifar_bayes_learn(img_arr,N,labels):

    #jokaiselle classille tarvis tehdä geneerinen sub image, jota voi verrata sit cifar sub imageihin


    sub_img_means = cifar_10_features(img_arr[0:1000],16)
    class_means = class_sub_means(sub_img_means,labels)

    _, counts = np.unique(labels[0:1000], return_counts=True)

    print(class_means[0][3][9])
    input("tas")
    class_color_means = feature_means(class_means,counts)

    #return class_color_mean, SIG, p


def calc_cov(img_arr,labels):

    img_means_arr = cifar_10_features(img_arr,16)
    class_means = all_means(img_means_arr,labels)

    SIG = []

    for i in range(0,len(class_means)):
        SIG.append(np.cov(class_means[i]))

    SIG = np.asarray(SIG)

    return SIG

def main():

    raw_train_img, train_labels = get_train_data()

    raw_test_data = get_test_data()

    # (50000,3,1024) (50000,)   (10000,3,1024)  (10000,)
    tr_img_arr, tr_label_arr, test_img_arr, test_label_arr = list2array(
        raw_train_img,
        train_labels, raw_test_data[0],
        raw_test_data[1])


    N = 16
    #features = [images, sub_images, colors, mean]
    #features = cifar_10_features(tr_img_arr[0:20], N)


    cifar_bayes_learn(tr_img_arr,N,tr_label_arr)

main()
