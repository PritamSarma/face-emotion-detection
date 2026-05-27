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
# MODEL PATH
# =========================

MODEL_PATH = os.path.join(
    BASE_DIR,
    'model',
    'custom_cnn.keras'
)

# =========================
# LOAD MODEL
# =========================

model = load_model(MODEL_PATH)

print("\nCustom CNN Model Loaded Successfully!")