import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from sklearn.metrics import (

    confusion_matrix,

    classification_report,

    accuracy_score
)

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =========================
# MODEL PATH
# =========================

model_path = os.path.join(
    BASE_DIR,
    'model',
    'custom_cnn_v2.keras'
)

# =========================
# TEST DATASET PATH
# =========================

test_dir = os.path.join(
    BASE_DIR,
    'dataset',
    'test'
)

# =========================
# LOAD MODEL
# =========================

model = load_model(model_path)

print("\nCNN V2 Model Loaded Successfully!")

# =========================
# IMAGE PREPROCESSING
# =========================

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================
# TEST GENERATOR
# =========================

test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(48, 48),

    color_mode='grayscale',

    batch_size=64,

    class_mode='categorical',

    shuffle=False
)

# =========================
# CLASS LABELS
# =========================

class_labels = list(
    test_generator.class_indices.keys()
)

print("\nClass Labels:")
print(class_labels)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(
    test_generator
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_generator.classes

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report:\n")

report = classification_report(

    y_true,

    y_pred,

    target_names=class_labels
)

print(report)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10,8))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=class_labels,

    yticklabels=class_labels
)

plt.title('CNN V2 Confusion Matrix')

plt.xlabel('Predicted Label')

plt.ylabel('True Label')

# =========================
# SAVE CONFUSION MATRIX
# =========================

save_path = os.path.join(

    BASE_DIR,

    'evaluation',

    'confusion_matrix_v2.png'
)

plt.savefig(save_path)

plt.show()

print(f"\nConfusion Matrix Saved:\n{save_path}")