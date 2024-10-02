import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('defect_detection_model.h5')

# Load an image for defect detection
def detect_defect(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (64, 64))
    image_array = np.expand_dims(image_resized, axis=0)

    # Predict whether the image contains a defect
    result = model.predict(image_array)
    if result[0][0] == 1:
        return "Defective"
    else:
        return "Non-defective"

# Example usage
print(detect_defect('test_images/sample_image.jpg'))
