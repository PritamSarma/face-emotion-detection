import os

from tensorflow.keras.models import load_model

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

cnn_v1_path = os.path.join(
    BASE_DIR,
    'model',
    'custom_cnn.keras'
)

cnn_v2_path = os.path.join(
    BASE_DIR,
    'model',
    'custom_cnn_v2.keras'
)

# =========================
# LOAD MODELS
# =========================

cnn_v1_model = load_model(cnn_v1_path)

print("\nCNN V1 Loaded!")

cnn_v2_model = load_model(cnn_v2_path)

print("\nCNN V2 Loaded!")