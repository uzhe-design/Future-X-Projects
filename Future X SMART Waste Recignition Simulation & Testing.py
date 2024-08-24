from google.colab import drive
drive.mount('/content/drive')
from google.colab import files
from IPython.display import Image, display

import tensorflow as tf

# Load the pre-trained model
model_path = '/content/drive/MyDrive/MECHANICAL ENGINEERING/COMPETITION/ZTE NextGen 5G MMU Hackathon 2024/Trained Model/smart_waste_bin_model.h5'
model = tf.keras.models.load_model(model_path)

# Verify the model is loaded
model.summary()

labels_path = '/content/drive/MyDrive/Waste/Trained(Using Google Golab)/labels.txt'
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

print("Labels:", labels)

from tensorflow.keras.preprocessing import image
import numpy as np

def predict_waste(img_path):
    # Resize the input image to 150x150 pixels
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)

    # Expand dimensions to match the model's input requirements and normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Predict the class
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    predicted_label = labels[class_idx]

    return predicted_label

def classify_multiple_wastes():
    while True:
        uploaded = files.upload()
        for img_path in uploaded.keys():
            display(Image(img_path))
            predicted_label = predict_waste(img_path)
            print(f'Predicted waste category: {predicted_label}')

        cont = input("Do you want to classify another image? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Exiting the loop.")
            break

classify_multiple_wastes()
