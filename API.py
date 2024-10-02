from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('defect_detection_model.h5')

# Defect detection function
def detect_defect(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (64, 64))
    image_array = np.expand_dims(image_resized, axis=0)

    result = model.predict(image_array)
    if result[0][0] == 1:
        return "Defective"
    else:
        return "Non-defective"

# API endpoint for defect detection
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    image_path = './uploads/' + image_file.filename
    image_file.save(image_path)

    result = detect_defect(image_path)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
