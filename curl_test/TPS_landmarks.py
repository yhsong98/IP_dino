import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
import csv


def estimate_tps_transform(landmarks_A, landmarks_B):
    """
    Estimate a Thin Plate Spline (TPS) transformation.

    :param landmarks_A: List of (x, y) landmark coordinates in Image A.
    :param landmarks_B: List of (x, y) corresponding landmarks in Image B.
    :return: RBF interpolator for mapping points from A to B.
    """
    landmarks_A = np.array(landmarks_A, dtype=np.float32)
    landmarks_B = np.array(landmarks_B, dtype=np.float32)

    # Train separate interpolators for X and Y coordinates
    tps_x = RBFInterpolator(landmarks_A, landmarks_B[:, 0], kernel="thin_plate_spline")
    tps_y = RBFInterpolator(landmarks_A, landmarks_B[:, 1], kernel="thin_plate_spline")

    return tps_x, tps_y


def map_point_tps(point_A, tps_x, tps_y):
    """
    Map a point using the TPS transformation.

    :param point_A: (x, y) coordinate in Image A.
    :param tps_x: RBF interpolator for X-coordinates.
    :param tps_y: RBF interpolator for Y-coordinates.
    :return: Transformed (x, y) point in Image B.
    """
    point_A = np.array(point_A).reshape(1, -1)  # Convert to proper shape
    return float(tps_x(point_A)), float(tps_y(point_A))


def resolve_ambiguity_tps(point_A, candidates_B, landmarks_A, landmarks_B):
    """
    Resolve ambiguity using TPS warping and landmark correlation.
    """
    # Compute TPS transformation
    tps_x, tps_y = estimate_tps_transform(landmarks_A, landmarks_B)

    # Predict where the point should be in Image B
    predicted_point_B = map_point_tps(point_A, tps_x, tps_y)

    if candidates_B:
        # Find the closest candidate to the predicted location
        candidates_B = np.array(candidates_B)
        distances = cdist([predicted_point_B], candidates_B, metric='euclidean')
        best_match_idx = np.argmin(distances)
        return tuple(candidates_B[best_match_idx])  # Best-matched candidate
    else:
        return predicted_point_B  # Return predicted location if no candidates are found


def visualize_tps_matching(image_A, image_B, landmarks_A, landmarks_B, point_A, candidates_B, best_match_B):
    """
    Visualizes the Thin Plate Spline (TPS) transformation results.

    :param image_A: Image A (reference).
    :param image_B: Image B (target).
    :param landmarks_A: List of landmark points in Image A.
    :param landmarks_B: List of corresponding landmark points in Image B.
    :param point_A: Selected point in Image A.
    :param candidates_B: List of candidate matches in Image B.
    :param best_match_B: Final selected point in Image B.
    :param predicted_point_B: TPS-predicted location in Image B (if no candidates are found).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # === IMAGE A (Reference) ===
    axes[0].imshow(cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB))
    axes[0].scatter(*zip(*landmarks_A), color='blue', s=100, label="Landmarks A")
    axes[0].scatter(point_A[0], point_A[1], color='red', s=150, label="Selected Point")
    axes[0].set_title("Image A (Reference)")
    axes[0].legend()

    # === IMAGE B (Target) ===
    axes[1].imshow(cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB))
    axes[1].scatter(*zip(*landmarks_B), color='blue', s=100, label="Landmarks B")

    # if candidates_B:
    #     axes[1].scatter(*zip(*candidates_B), color='orange', s=100, label="Candidates B")

    axes[1].scatter(best_match_B[0], best_match_B[1], color='red', s=150, label="Best Matched Point")

    # Show TPS-predicted location if no candidates exist
    # if not candidates_B:
    #     axes[1].scatter(predicted_point_B[0], predicted_point_B[1], color='green', s=150, marker='x',
    #                     label="TPS-Predicted Point")

    axes[1].set_title("Image B (Target)")
    axes[1].legend()

    plt.show()
# Example Usage
if __name__ == "__main__":
    # Example landmarks
    image_A = cv2.imread("cat_5.jpg")
    image_B = cv2.imread("wild_001702.jpg")

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
    candidates_B=None

    # Resolve ambiguity and get best match
    best_match_B = resolve_ambiguity_tps(point_A, candidates_B, landmarks_A, landmarks_B)
    #predicted_point_B = map_point_tps(point_A, *estimate_tps_transform(landmarks_A, landmarks_B))

    # Visualize results
    visualize_tps_matching(image_A, image_B, landmarks_A, landmarks_B, point_A, candidates_B, best_match_B)

    print(f"Selected Point in Image A: {point_A}")
    print(f"Best-matched Point in Image B: {best_match_B}")
