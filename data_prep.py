# will create train, val, test folders under data
"""" assumes structure
data
    class1
        img
        ...
    class2
        img
        ...

generates stucture:
data
    train
        class1
            img
            ...
        class2
            img
            ...
    val
        class1
            img
            ...
        class2
            img
            ...
    test
"""""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator


def image_data_generator(train_path, val_path, test_path, target_size, classes, batch_size):
    train_datagen = ImageDataGenerator().flow_from_directory(train_path,
                                                             target_size=target_size,
                                                             classes=classes,
                                                             batch_size=batch_size)
    val_datagen = ImageDataGenerator().flow_from_directory(val_path,
                                                           target_size=target_size,
                                                           classes=classes,
                                                           batch_size=batch_size)
    test_datagen = ImageDataGenerator().flow_from_directory(test_path,
                                                            target_size=target_size,
                                                            classes=classes,
                                                            batch_size=batch_size)
    return train_datagen, val_datagen, test_datagen


def create_batches(image_path, train_val_test_ratio):
    '''
    Create batches of data
    :param image_path:
    :param train_val_test_ratio:
    :return: 3 ImageDataGenerator() objects for train, validation
             and test batches
    '''

    if not os.path.exists(image_path):
        raise Exception("image path: {} does not exist".format(image_path))

    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/train'):
        os.makedirs('data/train')
    if not os.path.exists('data/val'):
        os.makedirs('data/val')
    if not os.path.exists('data/test'):
        os.makedirs('data/test')

    filenames = []
    classnames = []
    for (dirpath, dirnames, files) in os.walk(image_path):
        for dirname in dirnames:
            classnames.append(dirname)

        subfiles = []
        for file in files:
            if file.endswith('jpg') or file.endswith('png') or file.endswith('jpeg'):
                subfiles.append(dirpath + "/" + file)
        if len(subfiles) > 0:
            filenames.append(subfiles)

    for directory in filenames:
        np.random.shuffle(directory)

    legend = {}
    for i in range(len(classnames)):
        legend[classnames[i]] = filenames[i]

    for classname in legend:
        # split into train, val, test

        # train
        if not os.path.exists('data/train/{}'.format(classname)):
            os.makedirs('data/train/{}'.format(classname))

        for i in range(int(len(legend[classname]) * train_val_test_ratio[0])):  # training data
            shutil.copy(legend[classname][i], "data/train/{}".format(classname))

        # val
        if not os.path.exists('data/val/{}'.format(classname)):
            os.makedirs('data/val/{}'.format(classname))

        for i in range(int(len(legend[classname]) * train_val_test_ratio[0]), int(
                len(legend[classname]) * (train_val_test_ratio[0] + train_val_test_ratio[1]))):  # training data
            shutil.copy(legend[classname][i], "data/val/{}".format(classname))

        # test
        if not os.path.exists('data/test/{}'.format(classname)):
            os.makedirs('data/test/{}'.format(classname))

        for i in range(int(len(legend[classname]) * (train_val_test_ratio[0] + train_val_test_ratio[1])),
                       len(legend[classname])):  # training data
            shutil.copy(legend[classname][i], "data/test/{}".format(classname))

        train_path = "data/train"
        val_path = "data/val"
        test_path = "data/test"

    return image_data_generator(train_path, val_path, test_path, target_size=(150, 150), classes=classnames,
                                batch_size=32)


def plot(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if ims.shape[-1] != 3:
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()


train_val_test_ratio = (0.85, 0.1, 0.05)
create_batches("pictures", train_val_test_ratio)
