const video = document.getElementById('video');

const canvas = document.getElementById('canvas');

const emotionText = document.getElementById('emotion');

const confidenceText = document.getElementById('confidence');

const startBtn = document.getElementById('startBtn');

const stopBtn = document.getElementById('stopBtn');

let stream = null;

let predictionInterval = null;

// =========================
// START CAMERA
// =========================

startBtn.addEventListener('click', async () => {

    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;

        startPrediction();

    } catch (error) {

        console.error(error);

        alert("Could not access webcam.");
    }
});

// =========================
// STOP CAMERA
// =========================

stopBtn.addEventListener('click', () => {

    if (stream) {

        stream.getTracks().forEach(track => track.stop());

        video.srcObject = null;
    }

    clearInterval(predictionInterval);

    emotionText.innerText = "Emotion: -";

    confidenceText.innerText = "Confidence: -";
});

// =========================
// START PREDICTION LOOP
// =========================

function startPrediction() {

    predictionInterval = setInterval(async () => {

        if (!video.videoWidth) return;

        canvas.width = video.videoWidth;

        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');

        ctx.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        canvas.toBlob(async (blob) => {

            const formData = new FormData();

            formData.append(
                'image',
                blob,
                'frame.jpg'
            );

            try {

                const response = await fetch(
                    'http://127.0.0.1:5000/predict',
                    {
                        method: 'POST',
                        body: formData
                    }
                );

                const data = await response.json();

                emotionText.innerText =
                    `Emotion: ${data.emotion}`;

                confidenceText.innerText =
                    `Confidence: ${data.confidence}%`;

            } catch (error) {

                console.error(error);
            }

        }, 'image/jpeg');

    }, 1000);

}