import os
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from preprocess import preprocess_data

def evaluate_model(model_path, data_dir, img_size=(64, 64)):
    """
    Loads the trained model and evaluates its performance on the test dataset.

    Args:
        model_path (str): Path to the trained model file.
        data_dir (str): Path to the directory containing the dataset.
        img_size (tuple): Target image size for resizing (width, height).

    Raises:
        FileNotFoundError: If the model file or data directory is missing.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    # Load the trained model
    print("Loading the trained model...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Preprocess the dataset
    print("Preprocessing the evaluation data...")
    (X_train, y_train), (X_test, y_test) = preprocess_data(data_dir, img_size=img_size)

    # Evaluate the model on the test dataset
    print("Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X_test, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)

    # Generate classification report
    print("Classification Report:")
    print(classification_report(y_test, predicted_labels, target_names=['male', 'female']))

if __name__ == "__main__":
    # Paths and parameters
    model_path = "models/gender_model.keras"
    data_dir = "/Users/pk/Documents/ai_system/data_dir/data"
    img_size = (64, 64)

    try:
        evaluate_model(model_path, data_dir, img_size=img_size)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
