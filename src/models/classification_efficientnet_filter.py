#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.model_selection import train_test_split


def load_all_batches(source_folder, num_batches):
    """
    Load filtered images and labels from .npy files.
    Expected file names:
      processed_images_batch_{i}.npy
      processed_labels_batch_{i}.npy
    """
    all_images = []
    all_labels = []
    for i in range(num_batches):
        image_path = os.path.join(source_folder, f'processed_images_batch_{i}.npy')
        label_path = os.path.join(source_folder, f'processed_labels_batch_{i}.npy')
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print(f"Batch {i} not found, skipping.")
            continue
        images = np.load(image_path)
        labels = np.load(label_path)
        all_images.append(images)
        all_labels.append(labels)
    X = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def preprocess_images(images, target_size=(224, 224)):
    """
    Preprocess images:
      - Convert grayscale images to 3-channel
      - Resize to target_size
      - Normalize to [0, 1]
      - Apply EfficientNet preprocessing
    """
    processed = []
    for img in images:
        # If image is 2D (grayscale) or has one channel, convert to 3 channels.
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        # Resize image
        img_resized = tf.image.resize(img, target_size).numpy()
        processed.append(img_resized)
    processed = np.array(processed)
    processed = processed.astype(np.float32) / 255.0
    processed = preprocess_input(processed)
    return processed


def build_model(num_classes):
    """
    Build the EfficientNetB0-based classifier with additional dense layers.
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    # Set the folder where your filtered .npy files are stored.
    source_folder = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/data/processed_images_by_filter'

    # Adjust num_batches to the number of batches you have.
    num_batches = 10  # For example, if you have batches 0 to 9.

    print("Loading data...")
    X, y = load_all_batches(source_folder, num_batches)
    print("Data loaded. X shape:", X.shape, "y shape:", y.shape)

    print("Preprocessing images...")
    X_processed = preprocess_images(X, target_size=(224, 224))
    print("Images preprocessed. New shape:", X_processed.shape)

    # Determine number of classes based on labels.
    num_classes = len(np.unique(y))
    print("Number of classes:", num_classes)

    # Split data into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    print("Training set:", X_train.shape, y_train.shape)
    print("Validation set:", X_val.shape, y_val.shape)

    # Build and compile the model.
    model = build_model(num_classes)
    model.summary()

    # Train the model.
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model.
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print("Validation Loss:", val_loss, "Validation Accuracy:", val_accuracy)

    # Save the trained model.
    save_folder = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/src/models h5 format'
    os.makedirs(save_folder, exist_ok=True)
    model_save_path = os.path.join(save_folder, 'efficientnet_classifier_filter.h5')
    model.save(model_save_path)
    print("Model saved to", model_save_path)


if __name__ == '__main__':
    main()
