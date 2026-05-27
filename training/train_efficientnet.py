import os
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

import matplotlib.pyplot as plt

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =========================
# DATASET PATHS
# =========================

train_dir = os.path.join(
    BASE_DIR,
    'dataset',
    'train'
)

test_dir = os.path.join(
    BASE_DIR,
    'dataset',
    'test'
)

# =========================
# MODEL SAVE PATH
# =========================

model_path = os.path.join(
    BASE_DIR,
    'model',
    'efficientnet_model.keras'
)

# =========================
# IMAGE PREPROCESSING
# =========================

IMG_SIZE = 96

train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=25,

    width_shift_range=0.15,

    height_shift_range=0.15,

    zoom_range=0.15,

    horizontal_flip=True,

    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================
# DATA GENERATORS
# =========================

train_generator = train_datagen.flow_from_directory(
    train_dir,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode='rgb',

    batch_size=32,

    class_mode='categorical',

    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode='rgb',

    batch_size=32,

    class_mode='categorical',

    shuffle=False
)

# =========================
# CLASS LABELS
# =========================

print("\nClass Indices:")
print(train_generator.class_indices)

# =========================
# LOAD EFFICIENTNET
# =========================

base_model = EfficientNetB0(
    weights='imagenet',

    include_top=False,

    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# =========================
# PARTIAL FINE TUNING
# =========================

base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-40]:

    layer.trainable = False

# =========================
# BUILD MODEL
# =========================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    BatchNormalization(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(128, activation='relu'),

    Dropout(0.3),

    Dense(7, activation='softmax')

])

# =========================
# MODEL SUMMARY
# =========================

model.summary()

# =========================
# COMPILE MODEL
# =========================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(

    monitor='val_accuracy',

    patience=5,

    restore_best_weights=True,

    verbose=1
)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=2,

    verbose=1,

    min_lr=1e-6
)

checkpoint = ModelCheckpoint(

    model_path,

    monitor='val_accuracy',

    save_best_only=True,

    verbose=1
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(

    train_generator,

    epochs=15,

    validation_data=test_generator,

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)

# =========================
# SAVE MODEL
# =========================

model.save(model_path)

print("\nEfficientNetB0 Model Saved!")

# =========================
# EVALUATE MODEL
# =========================

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

print(f"Test Loss: {test_loss:.4f}")

# =========================
# PLOT ACCURACY
# =========================

plt.figure(figsize=(10,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('EfficientNetB0 Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.grid(True)

accuracy_path = os.path.join(
    BASE_DIR,
    'evaluation',
    'efficientnet_accuracy.png'
)

plt.savefig(accuracy_path)

plt.show()

# =========================
# PLOT LOSS
# =========================

plt.figure(figsize=(10,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('EfficientNetB0 Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.grid(True)

loss_path = os.path.join(
    BASE_DIR,
    'evaluation',
    'efficientnet_loss.png'
)

plt.savefig(loss_path)

plt.show()