from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = load_model("papaya_disease_model.h5")
img_size = 224

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory('data_split/test', target_size=(img_size, img_size), batch_size=32, class_mode='categorical', shuffle=False)

preds = model.predict(test_data)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_data.classes

print("Classification Report:\n", classification_report(true_classes, pred_classes, target_names=test_data.class_indices.keys()))
