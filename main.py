from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
from celery import Celery
from detect import detect_fruits
from predict import predict_fruit
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import requests
import json
import cv2

from util import identify_color_format

app = FastAPI()

classification_model = tf.keras.models.load_model("katty-3.keras")

# Celery setup (connect to Redis)
celery_app = Celery(
    "tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1"
)

# Load YOLOv8 pre-trained model (trained on COCO dataset)
detection_model = YOLO(
    "yolov8n.pt"
)  # 'yolov8n' is the smallest model, you can use 'yolov8s', 'yolov8m' for better accuracy

# Load class names from the JSON file
with open("labels.json", "r") as json_file:
    class_labels = json.load(json_file)


class ImageRequest(BaseModel):
    file_url: str


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/predict/file")
async def predict_file(file: UploadFile):

    raw_image = Image.open(BytesIO(await file.read()))
    raw_image_np = np.array(raw_image)
    raw_image_np = cv2.cvtColor(raw_image_np, cv2.COLOR_RGB2BGR)

    # pass to the detect service to extract fruits in the image
    fruit_images = detect_fruits(raw_image_np, detection_model)

    if len(fruit_images) == 0:
        print("no fruit detected from the image, exiting")
        return {"prediction": "no fruit detected from the image, exiting"}

    # pass the images to predict service
    output = predict_fruit(fruit_images, class_labels, classification_model)

    return {"prediction": output}


@app.post("/predict")
async def predict(request: ImageRequest):

    raw_image = read_image_from_url(request.file_url)

    print("color format %s" % (identify_color_format(raw_image)))

    # pass to the detect service to extract fruits in the image
    fruit_images = detect_fruits(raw_image, detection_model)

    if len(fruit_images) == 0:
        print("no fruit detected from the image, exiting")
        return {"prediction": "no fruit detected from the image, exiting"}

    # pass the images to predict service
    output = predict_fruit(fruit_images, class_labels, classification_model)

    return {"prediction": output}


def read_image_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch image: {response.status_code}")

    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image from URL")

    return image


@celery_app.task(name="tasks.process_image")
def predict(image_id: int, image_url: str):
    print("image id : %d, image url: %s" % (image_id, image_url))

    raw_image = read_image_from_url(image_url)

    print("color format %s" % (identify_color_format(raw_image)))

    # pass to the detect service to extract fruits in the image
    fruit_images = detect_fruits(raw_image, detection_model)

    print("Post Detection: %d" % (len(fruit_images)))

    if len(fruit_images) == 0:
        print("no fruit detected from the image, exiting")
        return {"prediction": "no fruit detected from the image, exiting"}

    # pass the images to predict service
    output = predict_fruit(fruit_images, class_labels, classification_model)

    print("Post Prediction: ", output)

    return {"prediction": output}
