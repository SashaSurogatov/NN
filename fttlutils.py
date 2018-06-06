from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils

#vgg16 - 0
#inception_v3 - 1
#Xception - 2

mode = 1

DATA_DIR = "/home/sasha/PycharmProjects/mag"
MODEL_DIR = os.path.join(DATA_DIR, "models")
NUM_CLASSES = 2
IMAGE_WIDTH = 0

if mode == 0:
    IMAGE_WIDTH = 224
else:
    IMAGE_WIDTH = 299

def train_test_split(X, Y, test_size, random_state):
    # using regular train_test_split results in classes not being represented
    splitter = StratifiedShuffleSplit(n_splits=1,
                                      test_size=test_size,
                                      random_state=random_state)
    for train, test in splitter.split(X, Y):
        Xtrain, Xtest, Ytrain, Ytest = X[train], X[test], Y[train], Y[test]
        break
    return Xtrain, Xtest, Ytrain, Ytest

def plot_loss(history):
    # visualize training loss and accuracy
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], color="r", label="Train")
    plt.plot(history.history["val_acc"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="r", label="Train")
    plt.plot(history.history["val_loss"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.tight_layout()
    #plt.show()
    plt.savefig("history.png")


def print_stats(ytest, ytest_, model_name):
    print(model_name)
    print("Accuracy: {:.5f}, Cohen's Kappa Score: {:.5f}".format(
        accuracy_score(ytest, ytest_),
        cohen_kappa_score(ytest, ytest_, weights="quadratic")))
    print("Confusion Matrix:")
    print(confusion_matrix(ytest, ytest_))
    print("Classification Report:")
    print(classification_report(ytest, ytest_))


def get_data(path, all=False):
    ys, fs = [], []
    f_train = open(path, "r")
    for line in f_train:
        #line = line.split(",")
        #ys.append(int(line[1]))
        #fs.append(line[0])
        line = line.split()
        ys.append(int(line[0]))
        fs.append(line[1])

        if all == False and len(ys) > 3000:
            break
    f_train.close()

    xs = []
    for y, f in zip(ys, fs):
        img = image.load_img(f, target_size=(IMAGE_WIDTH, IMAGE_WIDTH))
        #img = img.crop((55, 55, 200, 200))###########REMOVE THIS
        img = img.resize((IMAGE_WIDTH, IMAGE_WIDTH))
        img4d = image.img_to_array(img)
        img4d = np.expand_dims(img4d, axis=0)
        img4d = preprocess_input(img4d)
        xs.append(img4d[0])

    X = np.array(xs)
    y = np.array(ys)
    Y = np_utils.to_categorical(y, num_classes=NUM_CLASSES)
    return X, Y

def get_data_generator(path):
    f_train = open(path, "r")
    lines = f_train.readlines()
    last_pos = 0
    while True:
        ys, fs = [], []
        for line in lines[last_pos:]:
            line = line.split()
            ys.append(int(line[0]))
            fs.append(line[1])

            if len(ys) > 31:
                break
        last_pos = last_pos + 31

        xs = []
        for y, f in zip(ys, fs):
            img = image.load_img(f, target_size=(IMAGE_WIDTH, IMAGE_WIDTH))
            img4d = image.img_to_array(img)
            img4d = np.expand_dims(img4d, axis=0)
            img4d = preprocess_input(img4d)
            xs.append(img4d[0])

        X = np.array(xs)
        y = np.array(ys)
        Y = np_utils.to_categorical(y, num_classes=NUM_CLASSES)
        print ("last_pos", last_pos, len(X), len(Y))
        yield X, Y
        if last_pos >= len(lines):
            last_pos = 0

    f_train.close()