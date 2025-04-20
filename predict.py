from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("papaya_disease_model.h5")
class_labels = ['Anthracnose', 'Black_Spot', 'Ring_Spot', 'Phytophthora', 'Powdery_Mildew', 'Good']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"Predicted: {predicted_class} ({confidence*100:.2f}%)")

# Example:
# predict_image("sample.jpg")
