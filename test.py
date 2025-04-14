import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from scipy.interpolate import Rbf
import matplotlib
matplotlib.use('Qt5Agg')


def tps_warp_dynamic(source_points, target_points, grid_size=20, x_range=(0, 1), y_range=(0, 1), frames=10):
    """
    Dynamically visualizes Thin Plate Spline (TPS) transformation from source_points to target_points.
    """
    source_points = np.array(source_points)
    target_points = np.array(target_points)

    # Invert the y-coordinates to match the Matplotlib image coordinate system
    source_points[:, 1] = 1 - source_points[:, 1]
    target_points[:, 1] = 1 - target_points[:, 1]

    # Create grid for visualization
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    X_flat, Y_flat = X.ravel(), Y.ravel()

    # TPS Interpolation
    rbf_x = Rbf(source_points[:, 0], source_points[:, 1], target_points[:, 0], function='thin_plate')
    rbf_y = Rbf(source_points[:, 0], source_points[:, 1], target_points[:, 1], function='thin_plate')

    X_warped_full = rbf_x(X_flat, Y_flat).reshape(grid_size, grid_size)
    Y_warped_full = rbf_y(X_flat, Y_flat).reshape(grid_size, grid_size)

    fig, ax = plt.subplots()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title("TPS Transformation Animation")

    # Initialize plots
    scatter_source = ax.scatter(source_points[:, 0], source_points[:, 1], c='red', label='Source Landmarks')
    scatter_target = ax.scatter(target_points[:, 0], target_points[:, 1], c='blue', label='Target Landmarks')
    scatter_intermediate = ax.scatter(source_points[:, 0], source_points[:, 1], c='green',
                                      label='Interpolated Landmarks')
    grid_lines = []

    for i in range(grid_size):
        grid_lines.append(ax.plot([], [], 'k-', alpha=0.3)[0])  # Horizontal
        grid_lines.append(ax.plot([], [], 'k-', alpha=0.3)[0])  # Vertical

    def update(frame):
        t = frame / frames  # Interpolation factor
        intermediate_points = (1 - t) * source_points + t * target_points

        X_warped = ((1 - t) * X + t * X_warped_full).reshape(grid_size, grid_size)
        Y_warped = ((1 - t) * Y + t * Y_warped_full).reshape(grid_size, grid_size)

        for i in range(grid_size):
            grid_lines[2 * i].set_data(X_warped[i, :], Y_warped[i, :])  # Horizontal
            grid_lines[2 * i + 1].set_data(X_warped[:, i], Y_warped[:, i])  # Vertical

        scatter_intermediate.set_offsets(intermediate_points)
        return grid_lines + [scatter_intermediate, scatter_source, scatter_target]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    plt.legend()
    plt.show()


# Example input: Two sets of corresponding landmarks
source_landmarks = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]])
target_landmarks = np.array([[0.1, 0.1], [0.7, 0.3], [0.6, 0.9]])

# Run the dynamic TPS visualization
tps_warp_dynamic(source_landmarks, target_landmarks)

# Example input: Two sets of corresponding landmarks
with open('landmarks_A.csv', 'r') as f:
    landmark_file = csv.reader(f)
    source_landmarks = []
    for landmark in landmark_file:
        source_landmarks.append((float(landmark[0]), float(landmark[1])))

with open('landmarks_B.csv', 'r') as f:
    landmark_file = csv.reader(f)
    target_landmarks = []
    for landmark in landmark_file:
        target_landmarks.append((float(landmark[0]), float(landmark[1])))

# source_landmarks = np.array(source_landmarks)
# target_landmarks = np.array(target_landmarks)

# Run the TPS visualization
tps_warp_dynamic(source_landmarks, target_landmarks)
