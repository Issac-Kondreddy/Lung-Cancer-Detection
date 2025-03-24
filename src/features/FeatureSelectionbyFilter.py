#!/usr/bin/env python3
import cv2
import numpy as np
import os


def gabor_filter(image, ksize=21, sigma=5, theta=0, lamda=1 * np.pi, gamma=0.5, psi=0):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image


def laplacian_of_gaussian(image, sigma=0.5):
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    # Use CV_32F to keep the source and destination depths consistent.
    laplacian = cv2.Laplacian(blur, cv2.CV_32F)
    return laplacian


def apply_hybrid_gabor_log(image):
    # If the image is colored, convert it to grayscale.
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert image to float32 and normalize to [0, 1] if not already.
    if image.dtype != np.float32:
        image = np.float32(image) / 255.0

    theta_values = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    gabor_features = np.zeros_like(image, dtype=float)
    for theta in theta_values:
        gabor_features += gabor_filter(image, theta=theta)

    # Normalize Gabor features to range [0, 1]
    gabor_features = cv2.normalize(gabor_features, None, alpha=0, beta=1,
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    log_features = laplacian_of_gaussian(image)

    # Combine features using a weighted sum.
    combined_features = cv2.addWeighted(gabor_features, 0.5, log_features, 0.5, 0)
    return combined_features


def apply_hybrid_filters_to_batch(images):
    processed_images = []
    for image in images:
        processed_image = apply_hybrid_gabor_log(image)
        processed_images.append(processed_image)
    return np.array(processed_images)


def process_all_batches(source_folder, num_batches):
    all_processed_images = []
    all_labels = []

    # Process batches from 0 to num_batches - 1.
    for i in range(num_batches):
        image_path = os.path.join(source_folder, f'images_batch_{i}.npy')
        label_path = os.path.join(source_folder, f'labels_batch_{i}.npy')

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print(f"Batch {i}: files not found. Skipping this batch.")
            continue

        images = np.load(image_path)
        labels = np.load(label_path)

        processed_images = apply_hybrid_filters_to_batch(images)

        all_processed_images.append(processed_images)
        all_labels.append(labels)
        print(f"Processed batch {i}")

    return all_processed_images, all_labels


def save_processed_batches(processed_images, processed_labels, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for i, (images, labels) in enumerate(zip(processed_images, processed_labels)):
        image_save_path = os.path.join(save_folder, f'processed_images_batch_{i}.npy')
        label_save_path = os.path.join(save_folder, f'processed_labels_batch_{i}.npy')
        np.save(image_save_path, images)
        np.save(label_save_path, labels)
        print(f"Saved processed batch {i} to {save_folder}")


if __name__ == "__main__":
    source_folder = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/data/processed'

    save_folder = '/Users/issackondreddy/Desktop/Programming/ML Project/Lung Cancer Detection/data/processed_images_by_filter'

    num_batches = 181
    processed_images, processed_labels = process_all_batches(source_folder, num_batches)
    save_processed_batches(processed_images, processed_labels, save_folder)

    print("Processing and saving complete.")
