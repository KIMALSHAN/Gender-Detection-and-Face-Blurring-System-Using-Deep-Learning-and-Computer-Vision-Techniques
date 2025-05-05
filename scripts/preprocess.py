import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder_path, label, img_size):
    """
    Loads images from a folder, resizes them, and assigns a label.
    Skips invalid or hidden files.

    Args:
        folder_path (str): Path to the folder containing images.
        label (int): Label to assign to the images (e.g., 0 for male, 1 for female).
        img_size (tuple): Target size for resizing images (width, height).

    Returns:
        tuple: List of processed images and corresponding labels.
    """
    images = []
    labels = []

    print(f"Checking folder: {folder_path}")
    for image_name in os.listdir(folder_path):
        # Skip hidden files like .DS_Store or files starting with "."
        if image_name.startswith('.'):
            print(f"Skipping hidden file: {image_name}")
            continue

        image_path = os.path.join(folder_path, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image at {image_path} could not be read.")

            # Resize the image to the target size
            image = cv2.resize(image, img_size)

            # Normalize pixel values to range [0, 1]
            image = image / 255.0

            images.append(image)
            labels.append(label)

        except Exception as e:
            print(f"Warning: Could not process image '{image_path}'. Skipping. Error: {e}")

    return images, labels


def preprocess_data(data_dir, img_size=(64, 64)):
    """
    Preprocesses the dataset by loading images, resizing them, and splitting them into training and testing sets.

    Args:
        data_dir (str): Path to the root data directory containing subfolders for each class (e.g., 'male' and 'female').
        img_size (tuple): Target size for resizing images (width, height).

    Returns:
        tuple: Training and testing datasets in the format ((X_train, y_train), (X_test, y_test)).
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    images = []
    labels = []

    # Load images for each class
    for label_name, label in zip(['male', 'female'], [0, 1]):
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory '{folder_path}' does not exist.")
        
        class_images, class_labels = load_images_from_folder(folder_path, label, img_size)
        images.extend(class_images)
        labels.extend(class_labels)

    # Convert lists to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    return (X_train, y_train), (X_test, y_test)


# Example usage
if __name__ == "__main__":
    data_dir = "/Users/pk/Documents/ai_system/data_dir/data"
    try:
        (X_train, y_train), (X_test, y_test) = preprocess_data(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
