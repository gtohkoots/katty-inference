import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image: np.ndarray):
    """Plots an OpenCV image using matplotlib (converting BGR to RGB)."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)  # Convert to RGB
    plt.axis("off")  # Hide axis
    plt.show()


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    Ensures that the given image is in BGR format.

    If the image is in RGB format, it converts it to BGR.
    If it's already in BGR, it returns the image as-is.

    Parameters:
        image (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: Image in BGR format.
    """

    # If red is higher than blue, it's likely RGB and needs conversion
    if identify_color_format(image) == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image  # Already in BGR, return as-is


def identify_color_format(image: np.ndarray) -> str:
    """
    Identifies if an image (NumPy array) is in RGB or BGR format.

    Args:
        image (np.ndarray): Input image.

    Returns:
        str: "RGB" if the image is in RGB format, "BGR" if in BGR format, or "Unknown" if it has an invalid shape.
    """
    if image is None or not isinstance(image, np.ndarray):
        return "Invalid Image"

    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if image has 3 channels
        # Extract mean color values for the first few pixels
        b_mean, g_mean, r_mean = (
            np.mean(image[:, :, 0]),
            np.mean(image[:, :, 1]),
            np.mean(image[:, :, 2]),
        )

        # OpenCV loads in BGR, so blue > red indicates BGR
        if b_mean > r_mean:
            return "BGR"
        else:
            return "RGB"

    return "Unknown"
