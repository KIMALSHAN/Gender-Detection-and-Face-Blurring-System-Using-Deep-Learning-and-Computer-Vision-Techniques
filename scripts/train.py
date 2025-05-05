import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from preprocess import preprocess_data

def build_model(input_shape):
    """
    Builds a Convolutional Neural Network (CNN) model for gender classification.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).

    Returns:
        tensorflow.keras.models.Model: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Binary classification: male, female
    ])

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir, model_save_path, img_size=(64, 64), epochs=10, batch_size=32):
    """
    Preprocesses the data, trains the CNN model, and saves the trained model.

    Args:
        data_dir (str): Path to the directory containing the dataset.
        model_save_path (str): Path to save the trained model.
        img_size (tuple): Target image size for resizing (width, height).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    # Preprocess the dataset
    print("Preprocessing the data...")
    (X_train, y_train), (X_test, y_test) = preprocess_data(data_dir, img_size=img_size)

    # Build the model
    print("Building the model...")
    model = build_model(input_shape=(img_size[0], img_size[1], 3))

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size)

    # Save the model in the modern `.keras` format
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved at '{model_save_path}'")

    # Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    # Paths and parameters
    data_dir = "/Users/pk/Documents/ai_system/data_dir/data"
    model_save_path = "models/gender_model.keras"
    img_size = (64, 64)
    epochs = 10
    batch_size = 32

    try:
        train_model(data_dir, model_save_path, img_size=img_size, epochs=epochs, batch_size=batch_size)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
