import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16

from google.colab import drive
drive.mount('/content/drive')

# Step 1: Data Preprocessing
data_dir = '/content/drive/MyDrive/MECHANICAL ENGINEERING/COMPETITION/ZTE NextGen 5G MMU Hackathon 2024/Waste Classification/Train'

# Create ImageDataGenerator for training and validation with data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Added rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',  # Handle pixels outside the input boundaries
    validation_split=0.15  # 15% of the data for validation
)

# Training data generator
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),  # Standard size for image input
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

labels = train_data.class_indices
labels = dict((v, k) for k, v in labels.items())  # Reverse the key-value pairs

with open('labels.txt', 'w') as f:
    for idx in range(len(labels)):
        f.write(f"{idx}: {labels[idx]}\n")

print("Labels saved to labels.txt")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(4, activation='softmax')  # 4 output classes: paper, plastic, glass, general waste
])

# Step 3: Compile the Model with a Lower Learning Rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Training the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0001)

# Train the model
history = model.fit(
    train_data,
    epochs=100,  # Increased epochs
    validation_data=val_data,
    callbacks=[early_stopping, reduce_lr]
)

# Step 5: Evaluate the Model
val_acc = max(history.history['val_accuracy'])
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

import matplotlib.pyplot as plt

# Step 6: Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper left')

plt.show()

# Step 7: Save the Model
model.save('smart_waste_bin_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('waste_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)
