import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image(image: np.ndarray):
    """Plots an OpenCV image using matplotlib (converting BGR to RGB)."""
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to RGB
    plt.axis("off")  # Hide axis
    plt.show()