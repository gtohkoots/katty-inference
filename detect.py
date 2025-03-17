import cv2


def detect_fruits(image, detection_model):
    fruit_images = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Run YOLOv8 inference
    results = detection_model(image_rgb, conf=0.5)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cropped_fruit = image_rgb[y1:y2, x1:x2]

            fruit_images.append(cropped_fruit)

    return fruit_images
