import os
from typing import List
from dotenv import load_dotenv
from gcp_util import predict_custom_trained_model
from util import plot_image
import tensorflow as tf
from PIL import Image
import numpy as np

load_dotenv()

PROJECT = os.getenv("PROJECT_ID")
ENDPOINT = os.getenv("ENDPOINT_ID")
LOCATION = "us-east4"


def predict_fruit(
    images: List[np.ndarray], class_labels, classification_model
) -> List[str]:

    if not isinstance(classification_model, tf.keras.Model):
        print("wrong model type")
        return

    processed_images = np.vstack(
        [preprocess_image(image) for image in images]
    )  # Stack into batch

    # Use the correct input key based on SavedModel signature
    INPUT_KEY = "input_layer"

    # Convert NumPy array to a list of dictionaries
    instances = [{INPUT_KEY: image.tolist()} for image in processed_images]

    # Convert to JSON
    payload = {"instances": instances}

    predictions = predict_custom_trained_model(
        project=PROJECT, endpoint_id=ENDPOINT, location=LOCATION, instances=payload
    )  # Process in batch

    results = [
        class_labels[np.argmax(pred)] for pred in predictions
    ]  # Get top predictions

    return results


def preprocess_image(image: np.ndarray):
    pil_image = Image.fromarray(image)
    processed = pil_image.resize((100, 100))  # Adjust based on your model input size
    processed = np.expand_dims(processed, axis=0)  # Add batch dimension

    return processed
