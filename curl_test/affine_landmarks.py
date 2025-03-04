import numpy as np
import cv2
from scipy.spatial.distance import cdist
import csv
import matplotlib.pyplot as plt

def estimate_affine_transform(landmarks_A, landmarks_B):
    """
    Estimate an affine transformation matrix from landmarks in Image A to Image B.

    :param landmarks_A: List of (x, y) landmark coordinates in Image A.
    :param landmarks_B: List of (x, y) matched landmark coordinates in Image B.
    :return: 2x3 affine transformation matrix.
    """
    landmarks_A = np.array(landmarks_A, dtype=np.float32)
    landmarks_B = np.array(landmarks_B, dtype=np.float32)

    # Compute affine transformation using OpenCV
    transform_matrix = cv2.estimateAffinePartial2D(landmarks_A, landmarks_B)[0]  # 2x3 matrix
    return transform_matrix

def map_point_using_transform(point_A, transform_matrix):
    """
    Map a single point from Image A to Image B using the estimated transformation.

    :param point_A: Tuple (x, y) coordinate in Image A.
    :param transform_matrix: 2x3 affine transformation matrix.
    :return: Transformed point in Image B.
    """
    point_A_homo = np.array([[point_A[0], point_A[1], 1]], dtype=np.float32).T  # Convert to homogeneous
    point_B = np.dot(transform_matrix, point_A_homo).flatten()  # Apply transformation
    return tuple(point_B[:2])

def resolve_ambiguity(point_A, candidates_B, landmarks_A, landmarks_B):
    """
    Resolve ambiguity in matching candidates by using spatial correlation with landmarks.

    :param point_A: Selected (x, y) coordinate in Image A.
    :param candidates_B: List of (x, y) candidate coordinates in Image B.
    :param landmarks_A: List of landmark coordinates in Image A.
    :param landmarks_B: List of corresponding landmark coordinates in Image B.
    :return: Best-matched coordinate in Image B.
    """
    # Step 1: Compute transformation matrix from landmarks
    transform_matrix = estimate_affine_transform(landmarks_A, landmarks_B)

    # Step 2: Map point_A to Image B using the transformation
    predicted_point_B = map_point_using_transform(point_A, transform_matrix)

    if candidates_B:
        # Step 3: Find the closest candidate to the predicted point
        candidates_B = np.array(candidates_B)
        distances = cdist([predicted_point_B], candidates_B, metric='euclidean')
        best_match_idx = np.argmin(distances)
        return tuple(candidates_B[best_match_idx])  # Best-matched candidate
    else:
        # Step 4: If no candidate is found, return the predicted location
        return predicted_point_B

def visualize_matching(image_A, image_B, landmarks_A, landmarks_B, point_A, candidates_B, best_match_B):
    """
    Visualizes the original landmarks, selected point, and matched points.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Image A: Show landmarks and selected point
    axes[0].imshow(cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB))
    axes[0].scatter(*zip(*landmarks_A), color='blue', s=100, label="Landmarks")
    axes[0].scatter(point_A[0], point_A[1], color='red', s=150, label="Selected Point")
    axes[0].set_title("Image A (Reference)")
    axes[0].legend()

    # Image B: Show landmarks, candidates, and final matched point
    axes[1].imshow(cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB))
    axes[1].scatter(*zip(*landmarks_B), color='blue', s=100, label="Landmarks")
    if candidates_B:
        axes[1].scatter(*zip(*candidates_B), color='orange', s=100, label="Candidates")
    axes[1].scatter(best_match_B[0], best_match_B[1], color='red', s=150, label="Final Matched Point")
    axes[1].set_title("Image B (Target)")
    axes[1].legend()

    plt.show()

# Example Usage
if __name__ == "__main__":

    image_A = cv2.imread("cat_5.jpg")
    image_B = cv2.imread("cat1.jpg")


    with open('landmarks_A.csv', 'r') as f:
        landmark_file = csv.reader(f)
        landmarks_A = []
        for landmark in landmark_file:
            landmarks_A.append((float(landmark[0]), float(landmark[1])))

    with open('landmarks_B.csv', 'r') as f:
        landmark_file = csv.reader(f)
        landmarks_B = []
        for landmark in landmark_file:
            landmarks_B.append((float(landmark[0]), float(landmark[1])))


    # Example picked point in Image A
    point_A = (75, 115)

    # Example candidates in Image B (found by visual descriptor)
    #candidates_B = [(216.5, 216.5), (312, 216)]  # Ambiguous matches
    candidates_B=None

    # Resolve ambiguity and find best match
    best_match_B = resolve_ambiguity(point_A, candidates_B, landmarks_A, landmarks_B)

    visualize_matching(image_A, image_B, landmarks_A, landmarks_B, point_A, candidates_B, best_match_B)
    print(f"Selected Point in Image A: {point_A}")
    print(f"Best-matched Point in Image B: {best_match_B}")
