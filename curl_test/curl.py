import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import random

def compute_vector_field(height, width, landmarks):
    """
    Computes the vector field for each landmark considering diagonal quadrants.

    Parameters:
        image (np.ndarray): Input image for visualization.
        landmarks (np.ndarray): Nx2 array of (x, y) coordinates of landmarks.
        radius (float): Radius of the circular neighborhood.

    Returns:
        vector_field (np.ndarray): Nx2 array of vector field (Fx, Fy).
        LRTB_counts (list): List of dictionaries storing L, R, T, B counts for each point.
    """
    vector_field = np.zeros((height, width, 2))
    vector_field = vector_field.reshape(-1,2)

    for index,_ in enumerate(vector_field):
        xi, yi = index%width, index//width
        count = [0, 0, 0, 0] #R,T,L,B

        for (x_i, y_i) in landmarks:
            id = classify_vector_direction(np.subtract([x_i,yi],[xi,y_i]))
            count[id]+=1

        vector_field[index]=np.array([count[2]-count[0],count[1]-count[3]])

    vector_field = vector_field.reshape(height,width,2)
    return vector_field

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

def compute_curl(vector_field, dx=50, dy=50):
    """
    Compute the curl of a 2D vector field (P, Q).

    :param P: 2D NumPy array representing the x-component of the vector field.
    :param Q: 2D NumPy array representing the y-component of the vector field.
    :param dx: Spacing in the x-direction (default: 1.0).
    :param dy: Spacing in the y-direction (default: 1.0).
    :return: 2D NumPy array representing the curl of the vector field.
    """
    Fx, Fy = vector_field[:,:, 0], vector_field[:,:, 1]
    dFydx = np.gradient(Fy, dx, axis=1)
    dFxdy = np.gradient(Fx, dy, axis=0)

    curl_values = dFydx - dFxdy
    # for i, (x_i, y_i) in enumerate(landmarks):
    #     magnitude = np.sqrt(x_i ** 2 + y_i ** 2) * np.sqrt(Fx[i] ** 2 + Fy[i] ** 2)
    #     theta = np.arctan2(Fy[i], Fx[i])  # Angle in radians
    #     curl_values[i] = 2 * np.pi * np.cos(theta) * magnitude

    return curl_values


def visualize_curl_on_image(image_path, landmarks):
    """
    Plots the image with landmarks, vector field, and color-coded Curl values.

    Parameters:
        image_path (str): Path to the image file.
        landmarks (np.ndarray): Nx2 array of (x, y) coordinates of landmarks.
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute vector field and Curl
    vector_field = compute_vector_field(image.shape[1], image.shape[0],landmarks)
    curl_values = compute_curl(vector_field)

    # Normalize Curl values for visualization
    curl_min, curl_max = np.min(curl_values), np.max(curl_values)
    normalized_curl = (curl_values - curl_min) / (curl_max - curl_min)  # Normalize to [0,1]
    normalized_curl = normalized_curl*2
    normalized_curl = normalized_curl-1

    # Plot the image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    x_indices, y_indices = random.sample(range(0,224),200), random.sample(range(0,224),200)
    picked_curl = normalized_curl[x_indices, y_indices]
    # Scatter plot of landmarks with color-coded Curl values
    scatter = ax.scatter(
        x_indices, y_indices,
        c=picked_curl, cmap='coolwarm', s=100, edgecolors='black'
    )

    #Quiver plot for the vector field
    # ax.quiver(
    #     landmarks[:, 0], landmarks[:, 1],
    #     vector_field[:, 0], vector_field[:, 1],
    #     color='gold', scale=40, width=0.005
    # )

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Curl Value (Red = High, Blue = Low)")

    # Labels and title
    ax.set_title("Curl Visualization on Image")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")

    plt.show()


def rotate_landmarks_90_ccw(landmarks, image_width):
    """
    Rotates landmark coordinates 90 degrees counterclockwise.

    Parameters:
        landmarks (np.ndarray): Nx2 array of (x, y) landmark coordinates.
        image_width (int): Width of the original image.
        image_height (int): Height of the original image.

    Returns:
        rotated_landmarks (np.ndarray): Nx2 array of rotated (x', y') coordinates.
    """
    # Decompose landmarks into x and y
    x, y = landmarks[:, 0], landmarks[:, 1]

    # Compute new coordinates
    x_new = y
    y_new = image_width - x

    # Stack the new coordinates
    rotated_landmarks = np.column_stack((x_new, y_new))
    return rotated_landmarks

def rotate_landmarks(image_shape, landmarks, angle):
    """
    Rotate landmark coordinates by a given angle around the image center.

    :param image_shape: Tuple (height, width) of the image.
    :param landmarks: List of (x, y) coordinates.
    :param angle: Rotation angle in degrees (counterclockwise).
    :return: List of rotated (x, y) coordinates.
    """
    width, height  = image_shape[:2]
    cx, cy = width / 2, height / 2  # Image center

    # Convert angle to radians
    theta = np.radians(angle)

    # Rotation matrix
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated_landmarks = []
    for x, y in landmarks:
        # Translate point to origin
        x_translated, y_translated = x - cx, y - cy

        # Apply rotation
        x_rotated, y_rotated = rotation_matrix @ np.array([x_translated, y_translated])

        # Translate back to image space
        x_new, y_new = x_rotated + cx, y_rotated + cy
        rotated_landmarks.append((int(x_new), int(y_new)))  # Round to nearest pixel

    return rotated_landmarks
# Example usage (replace with your image path and landmarks)
image_path = "cat_5.jpg"

with open('landmarks_A.csv', 'r') as f:
    landmark_file = csv.reader(f)
    landmarks = []
    for landmark in landmark_file:
        landmarks.append((float(landmark[0]), float(landmark[1])))

# landmarks = np.array(landmarks)
# landmarks = np.rot90(landmarks, 1)
image_width = 224
image_height = 224
rotated_landmarks = rotate_landmarks((image_width,image_height),np.asarray(landmarks), 0)

visualize_curl_on_image(image_path, np.asarray(rotated_landmarks))
