from flask import Flask, request, render_template, Response, jsonify
from camera import LocalCamera
import cv2
from model import Expressions
import numpy as np
import base64

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = Expressions("model_2.json", "model_weights.h5")
font = cv2.FONT_ITALIC

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input', methods=['POST'])
def process_input():
    image_b64 = request.json['image'].split(",")[1]
    binary = base64.b64decode(image_b64)
    image = np.asarray(bytearray(binary), dtype="uint8")
    fr = cv2.imdecode(image, cv2.COLOR_BGR2GRAY) # cv2.IMREAD_COLOR in OpenCV 3.1
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray_fr, 1.3, 5)

    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]

        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        padding = 35
        cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(fr,(x-padding,y+padding),(x+w+padding,y+h+padding),(255,0,0),2)

    _, jpeg = cv2.imencode('.jpg', fr)
    img_base64 = base64.b64encode(jpeg.tobytes())

    return jsonify({'status':str(img_base64)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)