import cv2

from util import identify_color_format, plot_image


def detect_fruits(image, detection_model):
    fruit_images = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Run YOLOv8 inference
    results = detection_model(image, conf=0.5)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cropped_fruit = image_rgb[y1:y2, x1:x2]
            # plot_image(cropped_fruit)

            # print("color format %s" % (identify_color_format(cropped_fruit)))

            fruit_images.append(cropped_fruit)

    # plot_image(image_rgb)

    return fruit_images
