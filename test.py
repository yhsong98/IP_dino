# import cv2
#
# image = cv2.imread('images/cat_face/cat_5_mask.png')
# image = cv2.resize(image, (224, 224))
# cv2.imwrite('curl_test/cat_5_mask.png', image)
import numpy as np
def classify_vector_direction(vector):
    """
    Classify a 2D vector into one of four angular regions based on its angle with the positive x-axis.

    :param vector: Tuple (x, y) representing the vector.
    :return: One of four categories: 1, 2, 3, 4 corresponding to
             [-45, 45], [45, 135], [135, 225], [225, 315(-45)]
    """
    x, y = vector

    # Compute angle in degrees
    angle = np.degrees(np.arctan2(y, x))  # atan2(y, x) gives angle in [-180, 180]
    angle = (angle + 360) % 360  # Convert to [0, 360] range

    # Classify based on angle ranges
    if -45 <= angle < 45 or 315 <= angle < 360:
        return 0  # Region [-45, 45]
    elif 45 <= angle < 135:
        return 1  # Region [45, 135]
    elif 135 <= angle < 225:
        return 2  # Region [135, 225]
    elif 225 <= angle < 315:
        return 3  # Region [225, 315]

print(classify_vector_direction((-3,-4)))