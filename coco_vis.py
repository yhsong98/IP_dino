import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load JSON data
with open('images/instruments/coco_vis/1.json') as f:
    data = json.load(f)

# Load the target image
image_path = 'images/instruments/coco_vis/imgs/1.png'  # Path to the target image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Iterate over shapes and draw them on the image
for shape in data['shapes']:
    points = np.array(shape['points'], dtype=np.int32)
    label = shape['label']

    # Set color for each shape
    color = tuple(shape['fill_color']) if shape['fill_color'] else (255, 255, 255)

    if shape['shape_type'] == 'polygon':
        cv2.fillPoly(image, [points], color=color)
    elif shape['shape_type'] == 'line':
        for i in range(len(points) - 1):
            cv2.line(image, tuple(points[i]), tuple(points[i + 1]), color=color, thickness=2)
    elif shape['shape_type'] == 'point':
        cv2.circle(image, tuple(points[0]), radius=5, color=color, thickness=-1)

    # Optional: Label the shapes
    cv2.putText(image, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()
