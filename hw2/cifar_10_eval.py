import pickle
import numpy as np


#CONSTANTS
#path = glob.glob(r'C:\Users\Juho\Desktop\koulu\SGN13006\hw2\*')

# Height or width of the images (32 x 32)
size = 32

# 3 channels: Red, Green, Blue (RGB)
channels = 3

# Number of classes
num_classes = 10

# Each file contains 10000 images
image_batch = 10000

# 5 training files
num_files_train = 5

# Total number of training images
images_train = image_batch * num_files_train


def unpickle(file):
    # Convert byte stream to object
    with open(file, 'rb') as fo:
        print("Decoding file: %s" % (file))
        dict = pickle.load(fo, encoding='bytes')

        # Dictionary with images and labels
    return dict


def load_data(file):
    # Load file, unpickle it and return images with their labels

    data = unpickle(file)
    # Get raw images
    images_array = data[b'data']

    # Convert image
    #images = convert_images(images_array)

    # Convert class number to numpy array
    labels = np.array(data[b'labels'])

    # Images and labels in np array form
    return images_array, labels


#def convert_images(raw_images):
    # Convert images to numpy arrays

    # Convert raw images to numpy array and normalize it
    #raw = np.array(raw_images, dtype=float) / 255.0

    # Reshape to 4-dimensions - [image_number, channel, height, width]
    #images = raw.reshape([-1, channels, size, size])

    #images = images.transpose([0, 2, 3, 1])

    # 4D array - [image_number, height, width, channel]
    #return images


def get_train_data():
    # Load all training data in 5 files

    # Pre-allocate arrays
    images = np.zeros(shape=[images_train, size, size, channels], dtype=float)
    labels = np.zeros(shape=[images_train], dtype=int)

    # Starting index of training dataset
    start = 0

    # For all 5 files
    for i in range(0,num_files_train):
        # Load images and labels

        images_batch, labels_batch = load_data("data_batch_" + str(i+1))

        # Calculate end index for current batch
        end = start + image_batch

        # Store data to corresponding arrays
        images[start:end, :] = images_batch
        labels[start:end] = labels_batch

        # Update starting index of next batch
        start = end

        # Images, their labels and
        # corresponding one-hot vectors in form of np arrays
    return images, labels

def get_test_data():

    # Load all test data
    images, labels = load_data(file="test_batch")
    # Images, their labels and
    # corresponding one-hot vectors in form of np arrays
    return images, labels


train_img_data, train_label_data = get_train_data()
test_img_data, test_label_data = get_test_data()

#print(len(train_img_data))
#print(len(test_img_data))

