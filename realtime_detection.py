import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# =========================
# LOAD MODEL
# =========================

model = load_model('model/emotion_model.keras')

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
# LOAD FACE DETECTOR
# =========================

face_classifier = cv2.CascadeClassifier(
    'haarcascade/haarcascade_frontalface_default.xml'
)

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

# Increase webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================
# PREDICTION SMOOTHING
# =========================

prediction_buffer = deque(maxlen=10)

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Extract face
        roi_gray = gray[y:y+h, x:x+w]

        # Improve contrast
        roi_gray = cv2.equalizeHist(roi_gray)

        # Resize
        roi_gray = cv2.resize(roi_gray, (48, 48))

        # Normalize
        roi = roi_gray.astype("float32") / 255.0

        # Expand dimensions
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Prediction
        prediction = model.predict(roi, verbose=0)

        max_index = np.argmax(prediction)

        confidence = prediction[0][max_index] * 100

        # Add prediction to buffer
        prediction_buffer.append(max_index)

        # Smooth predictions
        smoothed_prediction = max(
            set(prediction_buffer),
            key=prediction_buffer.count
        )

        label = emotion_labels[smoothed_prediction]

        # Display text
        text = f"{label} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Window title
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================

cap.release()

cv2.destroyAllWindows()