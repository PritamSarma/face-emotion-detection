import os
import cv2
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

from inference import predict_emotion

# =========================
# CREATE APP
# =========================

app = Flask(__name__)

CORS(app)

# =========================
# FACE DETECTOR
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

cascade_path = os.path.join(
    BASE_DIR,
    'haarcascade',
    'haarcascade_frontalface_default.xml'
)

face_classifier = cv2.CascadeClassifier(
    cascade_path
)

# =========================
# HOME ROUTE
# =========================

@app.route('/')

def home():

    return jsonify({
        "message": "Emotion Detection API Running"
    })

# =========================
# PREDICTION ROUTE
# =========================

@app.route('/predict', methods=['POST'])

def predict():

    if 'image' not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files['image']

    npimg = np.frombuffer(
        file.read(),
        np.uint8
    )

    frame = cv2.imdecode(
        npimg,
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:

        return jsonify({
            "emotion": "No Face Detected",
            "confidence": 0
        })

    # Largest face
    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    face_img = frame[y:y+h, x:x+w]

    model_name = request.form.get(
        'model',
        'v1'
    )

    result = predict_emotion(
        face_img,
        model_name
    )

    # Add bounding box
    result["box"] = {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h)
    }

    return jsonify(result)

# =========================
# RUN SERVER
# =========================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )