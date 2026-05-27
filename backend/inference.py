import cv2
import numpy as np

from model_loader import model

# =========================
# EMOTION LABELS
# =========================

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

# =========================
# PREPROCESS IMAGE
# =========================

def preprocess_face(face_img):

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    gray = cv2.resize(gray, (48, 48))

    gray = gray.astype("float32") / 255.0

    gray = np.expand_dims(gray, axis=0)

    gray = np.expand_dims(gray, axis=-1)

    return gray

# =========================
# PREDICT EMOTION
# =========================

def predict_emotion(face_img):

    processed = preprocess_face(face_img)

    prediction = model.predict(
        processed,
        verbose=0
    )

    max_index = np.argmax(prediction)

    emotion = emotion_labels[max_index]

    confidence = float(
        prediction[0][max_index] * 100
    )

    return {
        "emotion": emotion,
        "confidence": round(confidence, 2)
    }