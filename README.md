# Excercise-Posture-Detection-CNN
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a CNN Model
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(3, activation='softmax')  # Assuming 3 classes: correct, incorrect deadlift, incorrect bench press
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Load data using ImageDataGenerator for preprocessing
def load_data(train_dir, val_dir, input_size):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_size,
        batch_size=32,
        class_mode='categorical')

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=input_size,
        batch_size=32,
        class_mode='categorical')

    return train_generator, val_generator

# Train the model
def train_model(model, train_data, val_data, epochs=10):
    model.fit(train_data, epochs=epochs, validation_data=val_data)

# Example usage
input_shape = (150, 150, 3)  # Adjust based on your image size
train_dir = 'path_to_train_data'  # Folder structure should follow ImageDataGenerator's convention
val_dir = 'path_to_val_data'

model = create_cnn_model(input_shape)
train_data, val_data = load_data(train_dir, val_dir, input_shape[:2])
train_model(model, train_data, val_data)
