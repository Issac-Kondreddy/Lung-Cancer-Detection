import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
import os
import warnings

warnings.filterwarnings('ignore')

# Constants
batch_size = 16
image_height = 180
image_width = 180


# Load the dataset
def load_data(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )


# Data augmentation
def augment_data():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomHeight(0.2),
        tf.keras.layers.RandomWidth(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])


# Resize and normalize images
def resize_and_normalize(dataset):
    resize_layer = tf.keras.layers.Resizing(224, 224)
    normalization_layer = tf.keras.layers.Rescaling(1 /.255)
    return dataset.map(lambda x, y: (normalization_layer(resize_layer(x)), y))


# Save preprocessed images and labels
def save_preprocessed_data(dataset, directory, batch_size):
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    for i, (images, labels) in enumerate(dataset):
        # Save images and labels in batches
        np.save(os.path.join(directory, f'images_batch_{i}.npy'), images.numpy())
        np.save(os.path.join(directory, f'labels_batch_{i}.npy'), labels.numpy())


# Main execution
if __name__ == "__main__":
    data_directory = "data/Augmented IQ-OTHNCCD lung cancer dataset"
    save_directory = "data/processed"

    dataset = load_data(data_directory)
    class_names = dataset.class_names
    print("Class names:", class_names)

    data_augmentation = augment_data()
    augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x), y))
    processed_dataset = resize_and_normalize(augmented_dataset)

    # Save the processed dataset
    save_preprocessed_data(processed_dataset, save_directory, batch_size)

    # Example to visualize one batch of processed images
    plt.figure(figsize=(10, 10))
    for images, labels in processed_dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
            plt.axis('off')
        plt.show()