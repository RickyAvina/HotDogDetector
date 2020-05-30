import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import data_prep

train_path = "data/train"
val_path = "data/val"
test_path = "data/test"
image_size = (150, 150)
batch_size = 32

train_batches, val_batches, test_batches = data_prep.image_data_generator(train_path=train_path, val_path=val_path, test_path=test_path,
                                                                          target_size=image_size, classes=['hot_dog', 'not_hot_dog'],
                                                                          batch_size=batch_size)


# ims, labels = next(train_batches)
# data_prep.plot(ims, titles=labels)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, 3, strides=2, activation='relu',
                     padding='same', input_shape=(150, 150, 3)))
    model.add(MaxPool2D(3, strides=2, padding='same'))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    return model

def create_model2():
    model = Sequential()
    model.add(Conv2D(32, 3, strides=2, activation='relu',
                     padding='same', input_shape=(150, 150, 3)))
    for size in [32, 64, 128]:
        model.add(Conv2D(size, 3, padding='same', activation='relu'))
        model.add(MaxPool2D(3, strides=2, padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    return model


model = create_model2()

callbacks = [
    keras.callbacks.ModelCheckpoint("models/save_at_{epoch}.h5")
]

model.compile(Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_batches, epochs=30, callbacks=callbacks, validation_data=val_batches, verbose=2)