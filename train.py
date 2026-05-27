import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# =========================
# DATASET PATHS
# =========================

train_dir = 'dataset/train'
test_dir = 'dataset/test'

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
# DISPLAY CLASS INDICES
# =========================

print("\nClass Indices:")
print(train_generator.class_indices)

# =========================
# CNN MODEL
# =========================

model = Sequential()

# ---------- BLOCK 1 ----------
model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    padding='same',
    input_shape=(48,48,1)
))

model.add(BatchNormalization())

model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    padding='same'
))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

# ---------- BLOCK 2 ----------
model.add(Conv2D(
    64,
    (3,3),
    activation='relu',
    padding='same'
))

model.add(BatchNormalization())

model.add(Conv2D(
    64,
    (3,3),
    activation='relu',
    padding='same'
))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

# ---------- BLOCK 3 ----------
model.add(Conv2D(
    128,
    (3,3),
    activation='relu',
    padding='same'
))

model.add(BatchNormalization())

model.add(Conv2D(
    128,
    (3,3),
    activation='relu',
    padding='same'
))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

# ---------- FLATTEN ----------
model.add(Flatten())

# ---------- DENSE LAYERS ----------
model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

# OUTPUT LAYER
model.add(Dense(7, activation='softmax'))

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
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stop, reduce_lr]
)

# =========================
# CREATE MODEL DIRECTORY
# =========================

os.makedirs("model", exist_ok=True)

# =========================
# SAVE MODEL
# =========================

model.save("model/emotion_model.keras")

print("\nModel Saved Successfully!")

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

plt.title('Training vs Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.grid(True)

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

plt.title('Training vs Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.grid(True)

plt.show()