from typing import List
from util import plot_image
import tensorflow as tf
from PIL import Image
import numpy as np


def predict_fruit(images: List[np.ndarray], class_labels) -> List[str]:

    result = []
    classification_model = tf.keras.models.load_model("katty-3.keras")
    if not isinstance(classification_model, tf.keras.Model):
        print("wrong model type")
        return

    processed_images = np.vstack(
        [preprocess_image(image) for image in images]
    )  # Stack into batch
    predictions = classification_model.predict(processed_images)  # Process in batch

    results = [
        class_labels[np.argmax(pred)] for pred in predictions
    ]  # Get top predictions

    return results


def preprocess_image(image: np.ndarray):
    pil_image = Image.fromarray(image)
    processed = pil_image.resize((100, 100))  # Adjust based on your model input size
    processed = np.expand_dims(processed, axis=0)  # Add batch dimension

    return processed
