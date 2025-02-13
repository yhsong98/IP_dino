import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv


def compute_vector_field_diagonal(landmarks, radius=50):
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
    num_points = len(landmarks)
    Fx = np.zeros(num_points)
    Fy = np.zeros(num_points)
    LRTB_counts = []  # Store counts for visualization

    for i, (x_i, y_i) in enumerate(landmarks):
        # Get all neighboring points within the radius
        neighbors = landmarks[np.sqrt((landmarks[:, 0] - x_i) ** 2 + (landmarks[:, 1] - y_i) ** 2) <= radius]

        # Exclude the current point
        neighbors = neighbors[(neighbors[:, 0] != x_i) | (neighbors[:, 1] != y_i)]

        # Count points in diagonal quadrants
        L = np.sum((neighbors[:, 0] < x_i) & (neighbors[:, 1] > y_i))  # Top-left (↖)
        R = np.sum((neighbors[:, 0] > x_i) & (neighbors[:, 1] < y_i))  # Bottom-right (↘)
        T = np.sum((neighbors[:, 0] > x_i) & (neighbors[:, 1] > y_i))  # Top-right (↗)
        B = np.sum((neighbors[:, 0] < x_i) & (neighbors[:, 1] < y_i))  # Bottom-left (↙)

        # Compute vector components
        Fx[i] = L - R
        Fy[i] = T - B

        # Store counts for visualization
        LRTB_counts.append({"L": L, "R": R, "T": T, "B": B})

    return np.column_stack((Fx, Fy))


def compute_curl(landmarks, vector_field):
    """
    Computes the discrete Curl at each landmark.

    Parameters:
        landmarks (np.ndarray): Nx2 array of (x, y) coordinates.
        vector_field (np.ndarray): Nx2 array of (Fx, Fy).

    Returns:
        curl_values (np.ndarray): Nx1 array of Curl values.
    """
    Fx, Fy = vector_field[:, 0], vector_field[:, 1]
    curl_values = np.zeros(len(landmarks))

    for i, (x_i, y_i) in enumerate(landmarks):
        magnitude = np.sqrt(x_i ** 2 + y_i ** 2) * np.sqrt(Fx[i] ** 2 + Fy[i] ** 2)
        theta = np.arctan2(Fy[i], Fx[i])  # Angle in radians
        curl_values[i] = 2 * np.pi * np.cos(theta) * magnitude

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
    vector_field = compute_vector_field_diagonal(landmarks)
    curl_values = compute_curl(landmarks, vector_field)

    # Normalize Curl values for visualization
    curl_min, curl_max = np.min(curl_values), np.max(curl_values)
    normalized_curl = (curl_values - curl_min) / (curl_max - curl_min)  # Normalize to [0,1]
    normalized_curl = normalized_curl*2
    normalized_curl = normalized_curl-1

    # Plot the image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    # Scatter plot of landmarks with color-coded Curl values
    scatter = ax.scatter(
        landmarks[:, 0], landmarks[:, 1],
        c=normalized_curl, cmap='coolwarm', s=100, edgecolors='black'
    )

    #Quiver plot for the vector field
    ax.quiver(
        landmarks[:, 0], landmarks[:, 1],
        vector_field[:, 0], vector_field[:, 1],
        color='gold', scale=40, width=0.005
    )

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Curl Value (Red = High, Blue = Low)")

    # Labels and title
    ax.set_title("Curl Visualization on Image")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")

    plt.show()


def rotate_landmarks_90_ccw(landmarks, image_width, image_height):
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
# Example usage (replace with your image path and landmarks)
image_path = "cat_5_rotated90_224.jpg"

with open('auto_proposed_landmarks.csv', 'r') as f:
    landmark_file = csv.reader(f)
    landmarks = []
    for landmark in landmark_file:
        landmarks.append((float(landmark[0]), float(landmark[1])))

# landmarks = np.array(landmarks)
# landmarks = np.rot90(landmarks, 1)
image_width = 224
image_height = 224
rotated_landmarks = rotate_landmarks_90_ccw(np.asarray(landmarks), image_width, image_height)

visualize_curl_on_image(image_path, np.asarray(rotated_landmarks))
