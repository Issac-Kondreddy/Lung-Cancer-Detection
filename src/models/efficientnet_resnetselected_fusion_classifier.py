#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


def load_all_batches_images(source_folder, num_batches):
    """
    Load filtered images and labels from individual batch files.
    Files are expected to be named:
       processed_images_batch_{i}.npy
       processed_labels_batch_{i}.npy
    """
    all_images = []
    all_labels = []
    for i in range(num_batches):
        image_path = os.path.join(source_folder, f'images_batch_{i}.npy')
        label_path = os.path.join(source_folder, f'labels_batch_{i}.npy')
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
      - Convert grayscale (or single-channel) images to 3 channels.
      - Resize images to target_size.
      - Normalize pixel values to [0, 1] and apply EfficientNet preprocessing.
    """
    processed = []
    for img in images:
        if img.ndim == 2:  # Grayscale image
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        img_resized = tf.image.resize(img, target_size).numpy()
        processed.append(img_resized)
    processed = np.array(processed)
    processed = processed.astype(np.float32) / 255.0
    processed = efficientnet_preprocess(processed)
    return processed


def load_selected_features(features_file, labels_file):
    """
    Load preselected ResNet features and corresponding labels.
    """
    X_features = np.load(features_file)
    y_features = np.load(labels_file)
    return X_features, y_features


def build_fusion_model(num_features, num_classes):
    """
    Build a fusion model with two input branches:
      - Image branch: EfficientNetB0 (frozen) with GlobalAveragePooling2D.
      - Feature branch: Dense layers processing preselected features.
    The outputs are concatenated and fed into further Dense layers to produce the final classification.
    """
    # Image branch
    image_input = Input(shape=(224, 224, 3), name='image_input')
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=image_input)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)

    # Selected features branch
    selected_feature_input = Input(shape=(num_features,), name='selected_feature_input')
    y_branch = Dense(64, activation='relu')(selected_feature_input)
    y_branch = Dense(32, activation='relu')(y_branch)

    # Concatenate both branches
    combined = Concatenate()([x, y_branch])
    z = Dense(256, activation='relu')(combined)
    z = Dense(128, activation='relu')(combined)
    z = Dense(64, activation='relu')(combined)
    z = Dense(32, activation='relu')(z)
    z = Dense(16, activation='relu')(z)
    output = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[image_input, selected_feature_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # Set paths:
    # Folder containing processed image batch files.
    images_folder = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/data/processed'
    # Files for preselected features.
    features_file = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/data/features_selected_Resnet/selected_features.npy'
    labels_file = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/data/features_selected_Resnet/labels.npy'

    # Adjust the number of batches to match your data.
    num_batches = 181  # For example, batches 0 to 180.

    print("Loading filtered images...")
    X_images, y_images = load_all_batches_images(images_folder, num_batches)
    print("Loaded images shape:", X_images.shape, "Labels shape:", y_images.shape)

    print("Preprocessing images...")
    X_images_processed = preprocess_images(X_images, target_size=(224, 224))
    print("Preprocessed images shape:", X_images_processed.shape)

    print("Loading selected features...")
    X_features, y_features = load_selected_features(features_file, labels_file)
    print("Selected features shape:", X_features.shape, "Features labels shape:", y_features.shape)

    # Assuming the label order is consistent, choose one set of labels.
    y = y_images  # or y_features if they are identical.

    if X_images_processed.shape[0] != X_features.shape[0]:
        print("Mismatch in sample count between images and selected features!")
        return

    num_classes = len(np.unique(y))
    print("Number of classes:", num_classes)

    # Split data (both image and feature inputs) into training and validation sets.
    X_img_train, X_img_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
        X_images_processed, X_features, y, test_size=0.2, random_state=42)
    print("Training set shapes:", X_img_train.shape, X_feat_train.shape, y_train.shape)
    print("Validation set shapes:", X_img_val.shape, X_feat_val.shape, y_val.shape)

    # Build and compile the fusion model.
    num_features = X_features.shape[1]
    model = build_fusion_model(num_features, num_classes)
    model.summary()

    # Train the model.
    history = model.fit(
        {'image_input': X_img_train, 'selected_feature_input': X_feat_train},
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=({'image_input': X_img_val, 'selected_feature_input': X_feat_val}, y_val)
    )

    # Evaluate the model.
    val_loss, val_accuracy = model.evaluate(
        {'image_input': X_img_val, 'selected_feature_input': X_feat_val},
        y_val
    )
    print("Validation Loss:", val_loss, "Validation Accuracy:", val_accuracy)

    # Save the trained model.
    save_folder = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/src/models h5 format'
    os.makedirs(save_folder, exist_ok=True)
    model_save_path = os.path.join(save_folder, 'efficientnet_resnetselected_fusion_classifier.h5')
    model.save(model_save_path)
    print("Model saved to", model_save_path)


if __name__ == '__main__':
    main()
