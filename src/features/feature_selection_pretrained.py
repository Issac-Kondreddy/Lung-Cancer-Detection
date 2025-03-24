#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from sklearn.feature_selection import SelectKBest, f_classif

def load_batch_npy(source_folder, batch_index):
    image_path = os.path.join(source_folder, f'images_batch_{batch_index}.npy')
    label_path = os.path.join(source_folder, f'labels_batch_{batch_index}.npy')
    images = np.load(image_path)
    labels = np.load(label_path)
    return images, labels

def extract_features_batch(images, model, target_size=(224, 224)):
    features = []
    for image in images:
        # Convert grayscale image to 3-channel if necessary.
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        image_resized = tf.image.resize(image, target_size).numpy()
        image_preprocessed = preprocess_input(image_resized.astype(np.float32))
        image_expanded = np.expand_dims(image_preprocessed, axis=0)
        feature = model.predict(image_expanded)
        features.append(feature.flatten())
    return np.array(features)

def process_all_batches(source_folder, num_batches, model):
    all_features = []
    all_labels = []
    for i in range(num_batches):
        print(f"Processing batch {i}")
        images, labels = load_batch_npy(source_folder, i)
        features = extract_features_batch(images, model)
        all_features.append(features)
        all_labels.append(labels)
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels

def main():
    # Folder containing your preprocessed .npy files.
    source_folder = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/data/processed'
    # Total number of batches (adjust as needed)
    num_batches = 181  # e.g., batches 0 to 180

    # Load the pretrained ResNet50 model.
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Extract features from all batches.
    features, labels = process_all_batches(source_folder, num_batches, model)
    print("Feature extraction complete. Features shape:", features.shape)

    # Perform feature selection.
    selector = SelectKBest(score_func=f_classif, k=100)
    selected_features = selector.fit_transform(features, labels)
    print("Feature selection complete. Selected features shape:", selected_features.shape)

    # Save selected features and labels.
    save_folder = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/data/features_selected_Resnet'
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, 'selected_features.npy'), selected_features)
    np.save(os.path.join(save_folder, 'labels.npy'), labels)
    print("Saved selected features and labels to", save_folder)

if __name__ == "__main__":
    main()
