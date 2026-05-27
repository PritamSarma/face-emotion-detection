import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
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
    'mobilenet_model.keras'
)

# =========================
# IMAGE PREPROCESSING
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
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
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# =========================
# DISPLAY CLASS INDICES
# =========================

print("\nClass Indices:")
print(train_generator.class_indices)

# =========================
# LOAD PRETRAINED MODEL
# =========================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

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
    optimizer='adam',
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
    epochs=20,
    validation_data=test_generator,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)

# =========================
# SAVE FINAL MODEL
# =========================

model.save(model_path)

print("\nMobileNetV2 Model Saved Successfully!")

# =========================
# EVALUATE MODEL
# =========================

test_loss, test_accuracy = model.evaluate(test_generator)

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

plt.title('MobileNetV2 Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.grid(True)

accuracy_graph_path = os.path.join(
    BASE_DIR,
    'evaluation',
    'mobilenet_accuracy.png'
)

plt.savefig(accuracy_graph_path)

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

plt.title('MobileNetV2 Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.grid(True)

loss_graph_path = os.path.join(
    BASE_DIR,
    'evaluation',
    'mobilenet_loss.png'
)

plt.savefig(loss_graph_path)

plt.show()