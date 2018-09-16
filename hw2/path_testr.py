import glob
import pickle
import numpy as np


#path = (r'C:\Users\Juho\Desktop\koulu\SGN13006\')

def unpickle(file):
    # Convert byte stream to object
    with open(file, 'rb') as fo:
        print("Decoding file: %s" % (file))
        dict = pickle.load(fo, encoding='bytes')

        # Dictionary with images and labels
    return dict

unpickle("data_batch_1")