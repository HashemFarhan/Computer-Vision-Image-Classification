import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetV2L, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
import tensorflow as tf

def preprocess_image(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def augment_image(image_path, class_0):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        preprocessing_function=preprocess_image
    )

    it = datagen.flow(img_array, batch_size=1)
    augmented_images = []

    for i in range(2):
        augmented_image = next(it)
        augmented_images.append(augmented_image[0].astype(np.uint8))

    return augmented_images

# def create_model(input_shape=(299, 299, 3)):
#     base_model = EfficientNetB0(
#         weights='imagenet',
#         include_top=False,
#         input_shape=input_shape
#     )
    
#     # Simplified and more focused architecture
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
    
#     # Enhanced dense layers with L2 regularization
#     x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
    
#     x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.4)(x)
    
#     x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
    
#     output = Dense(1, activation='sigmoid')(x)
    
#     model = Model(inputs=base_model.input, outputs=output)
#     return model

def create_model(input_shape=(299, 299, 3)):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.1)(x)
    
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

def prepare_data(train_csv_path, train_img_path):
    df = pd.read_csv(train_csv_path)
    df["image_path"] = df["id"].apply(lambda x: os.path.join(train_img_path, x))
    return df

def create_generators(balanced_df, target_size=(299, 299), batch_size=32):
    balanced_df['label'] = balanced_df['label'].astype(str)
    train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['label'])

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,

        brightness_range=[0.8, 1.2],
        rescale=1.0 / 255,
        preprocessing_function=preprocess_image
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=preprocess_image
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    return train_generator, val_generator

def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=1e-4
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            min_delta=1e-4
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]