import os
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    SeparableConv2D,

    MaxPooling2D,

    BatchNormalization,

    Dropout,

    Dense,

    GlobalAveragePooling2D
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
    'custom_cnn_v2.keras'
)

# =========================
# IMAGE GENERATORS
# =========================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    width_shift_range=0.15,

    height_shift_range=0.15,

    zoom_range=0.15,

    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================
# DATA LOADERS
# =========================

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(48, 48),

    color_mode='grayscale',

    batch_size=64,

    class_mode='categorical',

    shuffle=True
)

test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(48, 48),

    color_mode='grayscale',

    batch_size=64,

    class_mode='categorical',

    shuffle=False
)

# =========================
# BUILD MODEL
# =========================

model = Sequential([

    # BLOCK 1
    SeparableConv2D(
        32,
        (3,3),
        activation='relu',
        padding='same',
        input_shape=(48,48,1)
    ),

    BatchNormalization(),

    SeparableConv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Dropout(0.25),

    # BLOCK 2
    SeparableConv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    SeparableConv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Dropout(0.25),

    # BLOCK 3
    SeparableConv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    SeparableConv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Dropout(0.3),

    # GLOBAL POOLING
    GlobalAveragePooling2D(),

    Dense(
        128,
        activation='relu'
    ),

    BatchNormalization(),

    Dropout(0.4),

    Dense(
        7,
        activation='softmax'
    )

])

# =========================
# MODEL SUMMARY
# =========================

model.summary()

# =========================
# COMPILE MODEL
# =========================

loss_fn = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=0.1
)

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=loss_fn,

    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(

    monitor='val_accuracy',

    patience=6,

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

    epochs=50,

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

print("\nCNN V2 Model Saved!")

# =========================
# EVALUATE
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

plt.title('CNN V2 Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.grid(True)

accuracy_path = os.path.join(
    BASE_DIR,
    'evaluation',
    'cnn_v2_accuracy.png'
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

plt.title('CNN V2 Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.grid(True)

loss_path = os.path.join(
    BASE_DIR,
    'evaluation',
    'cnn_v2_loss.png'
)

plt.savefig(loss_path)

plt.show()