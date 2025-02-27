import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import csv
from PIL import Image

def find_nearest_landmarks(candidates, landmarks, k=5):
    """
    Find the k nearest landmarks for each candidate point.

    :param candidates: List of (x, y) candidate coordinates.
    :param landmarks: List of (x, y) landmark coordinates.
    :param k: Number of nearest landmarks to consider.
    :return: Dictionary {candidate_index: nearest_landmark_indices}.
    """
    candidates = np.array(candidates)
    landmarks = np.array(landmarks)

    # Compute pairwise distances
    distances = cdist(candidates, landmarks, metric='euclidean')

    # Get the indices of the k-nearest landmarks for each candidate
    nearest_indices = np.argsort(distances, axis=1)[:, :k]

    return {i: nearest_indices[i] for i in range(len(candidates))}


def classify_by_nearest_landmarks(candidates, landmarks, k=3):
    """
    Classify candidate points based on proximity to their nearest landmarks.

    :param candidates: List of (x, y) candidate coordinates.
    :param landmarks: List of (x, y) landmark coordinates.
    :param k: Number of nearest landmarks to consider.
    :return: Dictionary {candidate_index: assigned_landmark}.
    """
    nearest_landmarks = find_nearest_landmarks(candidates, landmarks, k)

    classifications = {}
    for candidate_idx, landmark_indices in nearest_landmarks.items():
        # Compute average location of k-nearest landmarks
        assigned_group_center = np.mean(np.array([landmarks[i] for i in landmark_indices]), axis=0)

        # Assign based on the closest landmark in the nearest list
        best_match = landmark_indices[0]  # Assign to the closest landmark
        classifications[candidate_idx] = best_match

    return classifications


def visualize_results(image, candidates, landmarks, classifications):
    """
    Visualize the candidate points and their nearest landmarks.

    :param candidates: List of (x, y) candidate coordinates.
    :param landmarks: List of (x, y) landmark coordinates.
    :param classifications: Dictionary {candidate_index: assigned_landmark}.
    """
    plt.figure(figsize=(8, 6))
    image = Image.open(image)
    plt.imshow(image)

    # Plot landmarks
    landmarks = np.array(landmarks)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', label="Landmarks", s=100, edgecolors='k')

    # Plot candidates and draw lines to assigned landmarks
    candidates = np.array(candidates)
    for i, (x, y) in enumerate(candidates):
        assigned_landmark = classifications[i]
        lx, ly = landmarks[assigned_landmark]

        plt.scatter(x, y, c='red', marker='*', s=200, label="Candidate" if i == 0 else None, edgecolors='k')
        plt.plot([x, lx], [y, ly], 'k--', alpha=0.6)  # Line to the assigned landmark

    plt.legend()
    plt.title("Nearest Landmark Assignment")
    # plt.gca().invert_yaxis()  # Match image coordinate system
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Example landmark points (not predefined groups)
    image_path = "cat_5.jpg"

    with open('landmarks_A.csv', 'r') as f:
        landmark_file = csv.reader(f)
        landmarks = []
        for landmark in landmark_file:
            landmarks.append((float(landmark[0]), float(landmark[1])))

    # Example candidate points (ambiguous matches)
    candidates = [
        (50, 100), (55, 110), (150, 100), (155, 110),  # Eye Candidates
        (105, 150), (110, 155)  # Nose Candidates
    ]

    # Classify candidates by nearest landmarks
    classifications = classify_by_nearest_landmarks(candidates, landmarks, k=3)

    # Visualize results
    image = 'cat_5.jpg'
    visualize_results(image, candidates, landmarks, classifications)

    # Print classification results
    for i, landmark_idx in classifications.items():
        print(f"Candidate {candidates[i]} → Assigned to Landmark {landmarks[landmark_idx]}")
