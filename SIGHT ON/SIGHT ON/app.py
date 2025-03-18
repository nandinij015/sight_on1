from flask import Flask, render_template, Response, redirect, url_for
import cv2
import torch
import numpy as np
import sounddevice as sd  # Import sounddevice for audio playback

app = Flask(__name__)

KNOWN_WIDTH = 0.5  
FOCAL_LENGTH = 600  
ALERT_DISTANCE = 0.75  

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
import numpy as np
import sounddevice as sd  # Import sounddevice for audio playback

def play_alert():
    duration = 0.5  # Duration of the beep in seconds
    frequency = 440  # Frequency of the beep in Hz
    fs = 44100  # Sampling frequency

    # Generate a sine wave for the beep
    t = np.linspace(0, duration, int(fs * duration), False)
    beep = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Play the sound
    sd.play(beep, fs)
    sd.wait()  # Wait until the sound finishes playing


def calculate_distance(pixel_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  

        results = model(frame)
        detections = results.pandas().xyxy[0]  

        for _, row in detections.iterrows():
            if row['confidence'] >= 0.4:  
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                label = row['name']
                pixel_width = x2 - x1
                distance = calculate_distance(pixel_width)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {distance:.2f}m', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if distance < ALERT_DISTANCE:
                    play_alert()
                    print(f"Alert: {label} is {distance:.2f} meters away!")

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    return redirect(url_for('calculate'))

@app.route('/calculate')
def calculate():
    return render_template('calculate.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
