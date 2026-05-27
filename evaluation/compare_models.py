import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report
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
# MODEL PATHS
# =========================

v1_path = os.path.join(
    BASE_DIR,
    'model',
    'custom_cnn.keras'
)

v2_path = os.path.join(
    BASE_DIR,
    'model',
    'custom_cnn_v2.keras'
)

# =========================
# TEST DATASET
# =========================

test_dir = os.path.join(
    BASE_DIR,
    'dataset',
    'test'
)

# =========================
# LOAD MODELS
# =========================

print("\nLoading Models...")

model_v1 = load_model(v1_path)

model_v2 = load_model(v2_path)

print("Models Loaded Successfully!")

# =========================
# IMAGE GENERATOR
# =========================

test_datagen = ImageDataGenerator(
    rescale=1./255
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
# CLASS LABELS
# =========================

class_labels = list(
    test_generator.class_indices.keys()
)

# =========================
# TRUE LABELS
# =========================

y_true = test_generator.classes

# =========================
# EVALUATION FUNCTION
# =========================

def evaluate_model(model, model_name):

    print(f"\nEvaluating {model_name}...")

    predictions = model.predict(
        test_generator
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_true,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average='weighted'
    )

    print(f"\n{model_name} Results")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_labels
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return {

        "name": model_name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "confusion_matrix": cm
    }

# =========================
# RUN EVALUATIONS
# =========================

results_v1 = evaluate_model(
    model_v1,
    "CNN V1"
)

results_v2 = evaluate_model(
    model_v2,
    "CNN V2"
)

# =========================
# CREATE COMPARISON TABLE
# =========================

comparison_df = pd.DataFrame({

    "Model": [
        results_v1["name"],
        results_v2["name"]
    ],

    "Accuracy": [
        results_v1["accuracy"],
        results_v2["accuracy"]
    ],

    "Precision": [
        results_v1["precision"],
        results_v2["precision"]
    ],

    "Recall": [
        results_v1["recall"],
        results_v2["recall"]
    ],

    "F1 Score": [
        results_v1["f1"],
        results_v2["f1"]
    ]
})

print("\nModel Comparison:\n")

print(comparison_df)

# =========================
# SAVE CSV
# =========================

csv_path = os.path.join(

    BASE_DIR,

    'evaluation',

    'model_comparison.csv'
)

comparison_df.to_csv(
    csv_path,
    index=False
)

print(f"\nComparison CSV Saved:\n{csv_path}")

# =========================
# BAR CHART
# =========================

metrics = [

    'Accuracy',

    'Precision',

    'Recall',

    'F1 Score'
]

x = np.arange(len(metrics))

width = 0.35

fig, ax = plt.subplots(figsize=(10,6))

v1_scores = [

    results_v1["accuracy"],

    results_v1["precision"],

    results_v1["recall"],

    results_v1["f1"]
]

v2_scores = [

    results_v2["accuracy"],

    results_v2["precision"],

    results_v2["recall"],

    results_v2["f1"]
]

ax.bar(
    x - width/2,
    v1_scores,
    width,
    label='CNN V1'
)

ax.bar(
    x + width/2,
    v2_scores,
    width,
    label='CNN V2'
)

ax.set_ylabel('Score')

ax.set_title('Model Comparison')

ax.set_xticks(x)

ax.set_xticklabels(metrics)

ax.legend()

plt.grid(True)

comparison_plot_path = os.path.join(

    BASE_DIR,

    'evaluation',

    'model_comparison.png'
)

plt.savefig(comparison_plot_path)

plt.show()

print(f"\nComparison Plot Saved:\n{comparison_plot_path}")

# =========================
# CONFUSION MATRICES
# =========================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

sns.heatmap(

    results_v1["confusion_matrix"],

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=class_labels,

    yticklabels=class_labels,

    ax=axes[0]
)

axes[0].set_title("CNN V1")

axes[0].set_xlabel("Predicted")

axes[0].set_ylabel("True")

sns.heatmap(

    results_v2["confusion_matrix"],

    annot=True,

    fmt='d',

    cmap='Greens',

    xticklabels=class_labels,

    yticklabels=class_labels,

    ax=axes[1]
)

axes[1].set_title("CNN V2")

axes[1].set_xlabel("Predicted")

axes[1].set_ylabel("True")

confusion_compare_path = os.path.join(

    BASE_DIR,

    'evaluation',

    'confusion_matrix_comparison.png'
)

plt.savefig(confusion_compare_path)

plt.show()

print(f"\nConfusion Matrix Comparison Saved:\n{confusion_compare_path}")

# =========================
# FINAL OBSERVATIONS
# =========================

print("\n=========================")
print("REALTIME OBSERVATIONS")
print("=========================\n")

print("CNN V1:")
print("- Better realtime stability")
print("- Better confidence")
print("- Handles spectacles better")
print("- More reliable webcam inference\n")

print("CNN V2:")
print("- Better architecture complexity")
print("- More sensitive to eye-region")
print("- Overpredicts Surprise sometimes")
print("- Lower realtime robustness\n")

print("Conclusion:")
print("CNN V1 is the better production model.")