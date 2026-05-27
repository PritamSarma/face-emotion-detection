# Real-Time Facial Emotion Detection using Deep Learning

## Overview

This project is a real-time facial emotion recognition web application built using Deep Learning, Computer Vision, Flask, and JavaScript.
The system detects human emotions from webcam video in real time and displays live prediction confidence scores through an interactive browser interface.

The project uses custom Convolutional Neural Network (CNN) architectures trained on the FER2013 dataset and includes comparative evaluation between multiple models.

---

# Features

* Real-time emotion detection using webcam
* Browser-based frontend using HTML, CSS, and JavaScript
* Flask backend API
* Live emotion probability bars
* Model comparison system
* Multiple CNN architectures
* Real-time inference
* Evaluation metrics and confusion matrices
* Model toggle support
* Responsive UI
* AI-style webcam interface

---

# Detected Emotions

The system predicts the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

# Technologies Used

## Deep Learning

* TensorFlow
* Keras

## Computer Vision

* OpenCV

## Backend

* Flask
* Flask-CORS

## Frontend

* HTML
* CSS
* JavaScript

## Data Visualization

* Matplotlib
* Seaborn
* Pandas

---

# Dataset

This project uses the FER2013 dataset for facial emotion recognition.

Dataset characteristics:

* 48x48 grayscale facial images
* 7 emotion classes
* Real-world facial expressions
* Challenging low-resolution emotion dataset

---

# Project Structure

```text
face-emotion-detection/
│
├── backend/
│   ├── app.py
│   ├── inference.py
│   └── model_loader.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── training/
│   ├── train_custom.py
│   ├── train_custom_v2.py
│   ├── train_mobilenet.py
│   └── train_efficientnet.py
│
├── evaluation/
│   ├── evaluate_custom.py
│   ├── evaluate_custom_v2.py
│   ├── compare_models.py
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── confusion_matrix_comparison.png
│
├── model/
│   ├── custom_cnn.keras
│   └── custom_cnn_v2.keras
│
├── dataset/
│   ├── train/
│   └── test/
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Models Used

## CNN V1 (Baseline Model)

The baseline CNN model demonstrated the best real-time performance and stability.

### Characteristics

* Lightweight architecture
* Fast inference
* Stable webcam predictions
* Better handling of spectacles and lighting variations

### Performance

* Approximate validation accuracy: 65–70%
* Best realtime stability
* Highest confidence for correct predictions

---

## CNN V2 (Experimental Model)

An improved CNN architecture with additional optimization techniques.

### Improvements

* BatchNormalization
* SeparableConv2D
* GlobalAveragePooling
* Label smoothing
* Improved regularization

### Observations

* Better architecture complexity
* Better confusion matrix in some classes
* More sensitive to spectacles and eye-region occlusion
* Overpredicted Surprise during realtime testing

---

# Transfer Learning Experiments

The following transfer learning models were also explored:

* MobileNetV2
* EfficientNetB0

However, due to:

* FER2013 low-resolution grayscale images
* CPU-only training constraints
* realtime deployment requirements

The custom CNN architectures performed better in practical realtime inference.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/face-emotion-detection.git

cd face-emotion-detection
```

---

# Create Virtual Environment

## Windows

```bash
python -m venv venv

venv\Scripts\activate
```

## Linux/Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Start Flask Backend

```bash
python backend/app.py
```

Backend runs on:

```text
http://127.0.0.1:5000
```

---

## Start Frontend

Open a second terminal:

```bash
cd frontend

python -m http.server 5500
```

Open browser:

```text
http://127.0.0.1:5500
```

---

# Training Models

## Train CNN V1

```bash
python training/train_custom.py
```

## Train CNN V2

```bash
python training/train_custom_v2.py
```

---

# Evaluation

## Evaluate CNN V1

```bash
python evaluation/evaluate_custom.py
```

## Evaluate CNN V2

```bash
python evaluation/evaluate_custom_v2.py
```

## Compare Models

```bash
python evaluation/compare_models.py
```

---

# Evaluation Metrics

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Real-time inference observations

---

# Real-Time Observations

During realtime webcam testing:

* CNN V1 provided more stable predictions
* CNN V1 handled spectacles better
* CNN V2 was more sensitive to eye-region occlusion
* Both models showed reduced accuracy with spectacles
* CNN V2 occasionally overpredicted Surprise

These observations demonstrate the importance of practical realtime evaluation beyond standard validation accuracy.

---

# Future Improvements

Possible future enhancements:

* GPU acceleration
* Better face tracking
* Improved UI animations
* Emotion analytics dashboard
* Audio emotion detection
* Mobile deployment
* Better dataset augmentation
* Transformer-based emotion recognition

---

# Screenshots

Add screenshots here:

* Main UI
* Probability bars
* Realtime detection
* Confusion matrix
* Model comparison graphs

---

# Conclusion

This project demonstrates a complete end-to-end AI application pipeline involving:

* Deep learning model development
* Real-time computer vision
* Backend API integration
* Frontend development
* Model experimentation
* Comparative evaluation
* Practical realtime deployment analysis

The baseline custom CNN architecture achieved the best balance between:

* realtime stability
* inference speed
* practical usability
* deployment efficiency

making it the preferred production model for this application.

---

# Author

Pritam Sarma

---

# License

This project is for educational and research purposes.
