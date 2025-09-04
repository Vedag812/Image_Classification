import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_data(data_dir, img_size=(150, 150), batch_size=32):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
    )

    return train_gen, val_gen, test_gen
