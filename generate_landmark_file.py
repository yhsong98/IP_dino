import argparse
import os
import torch
from sklearn.cluster import DBSCAN
from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import cv2
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
import heapq
import random
matplotlib.use('Qt5Agg')

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def show_similarity_interactive(image_path_a: str, image_folder_path_b: str, landmark_file, num_ref_points: int, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8',
                                num_sim_patches: int = 1, sim_threshold: float = 0.65):
    """
     finding similarity between a descriptor in one image to the all descriptors in the other image.
     :param image_path_a: path to first image.
     :param image_path_b: path to second image.
     :param load_size: size of the smaller edge of loaded images. If None, does not resize.
     :param layer: layer to extract descriptors from.
     :param facet: facet to extract descriptors from.
     :param bin: if True use a log-binning descriptor.
     :param stride: stride of the model.
     :param model_type: type of model to extract descriptors from.
     :param num_sim_patches: number of most similar patches from image_b to plot.
    """
    # extract descriptors
    color_map = [
    "blue", "green", "red", "cyan", "magenta", "yellow", "black", "orange", "purple", "brown",
    "pink", "gray", "olive", "teal", "navy", "maroon", "lime", "gold", "indigo", "turquoise",
    "violet", "aqua", "coral", "orchid", "salmon", "khaki", "plum", "darkgreen", "darkblue", "crimson"
]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size


    mask = cv2.resize(cv2.imread(landmark_file, cv2.IMREAD_GRAYSCALE),(load_size_a[1],load_size_a[0]))
    coords = cv2.findNonZero(mask)
    if coords is not None:
        coords_list = coords.reshape(-1, 2).tolist()
        landmarks = random.sample(coords_list, num_ref_points)
    else:
        landmarks = []


    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    all_files = os.listdir(image_folder_path_b)
    images = [file for file in all_files if file.endswith(('.jpg', '.png', '.jpeg'))]
    for image_path in sorted(images)[::-1]:
        start=time.time()
        image_path_b = os.path.join(image_folder_path_b, image_path)
        batch_b_rotations = extractor.preprocess(image_path_b, load_size, rotate=True)

        descs_b_s = []
        num_patches_b_rotations, load_size_b_rotations = [], []
        for batch in batch_b_rotations:
            descs_b_s.append(extractor.extract_descriptors(batch[0].to(device), layer, facet, bin, include_cls=True))
            num_patches_b_rotations.append(extractor.num_patches)
            load_size_b_rotations.append(extractor.load_size)

        radius = patch_size // 2
        # plot image_a and the chosen patch. if nothing marked chosen patch is cls patch.
        axes[0][0].clear()
        axes[0][0].title.set_text('A (reference)')
        axes[0][0].set_axis_off()
        axes[0][0].imshow(image_pil_a)

        axes[0][1].clear()
        axes[0][1].title.set_text('B (original orientation)')
        axes[0][1].set_axis_off()
        axes[0][1].imshow(batch_b_rotations[0][1])
        ptses = np.asarray(landmarks)


        a_landmark_points_rotations = []
        b_landmark_points_rotations = []
        landmarks_ids_rotation = []
        multi_curr_similarities_rotations = []


        for id, descs_b_rot in enumerate(descs_b_s):
            a_landmark_points = []
            b_landmark_points = []
            landmark_ids = []
            similarities = chunk_cosine_sim(descs_a, descs_b_rot)
            multi_curr_similarities = []
            for idx, pts in enumerate(ptses):
                y_coor, x_coor = int(pts[1]), int(pts[0])
                new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
                new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
                y_descs_coor = int(new_H / load_size_a[0] * y_coor)
                x_descs_coor = int(new_W / load_size_a[1] * x_coor)

                # get and draw current similarities
                raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
                reveled_desc_idx_including_cls = raveled_desc_idx + 1

                curr_similarities = similarities[0, 0, reveled_desc_idx_including_cls, 1:]
                curr_similarities = curr_similarities.reshape(num_patches_b_rotations[id])

                multi_curr_similarities.append(curr_similarities)
            multi_curr_similarities_rotations.append(multi_curr_similarities)

            # get and draw most similar points

            for landmark_id, curr_similarities in enumerate(multi_curr_similarities):
                sim, idx = torch.topk(curr_similarities.flatten(), 1)
                if sim > sim_threshold:
                    center_a = ptses[landmark_id]
                    a_landmark_points.append(center_a)
                    b_y_descs_coor, b_x_descs_coor = torch.div(idx, num_patches_b_rotations[id][1], rounding_mode='floor'), idx % num_patches_b_rotations[id][1]
                    center_b = ((b_x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                                (b_y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                    b_landmark_points.append([center_b[0].cpu().numpy()[0],center_b[1].cpu().numpy()[0]])

                    landmark_ids.append(landmark_id)

            a_landmark_points_rotations.append(a_landmark_points)
            b_landmark_points_rotations.append(b_landmark_points)
            landmarks_ids_rotation.append(landmark_ids)

        scores_num = []
        scores_ds =[]
        for a_landmark_points, b_landmark_points in zip(a_landmark_points_rotations,b_landmark_points_rotations):
            num_landmark_points = len(b_landmark_points)
            directional_sim = compare_directions(a_landmark_points, b_landmark_points)
            directional_sim = directional_sim if not np.isnan(directional_sim) else 0
            print("[", num_landmark_points,",",directional_sim,"] ",end="")
            scores_num.append(len(b_landmark_points))
            scores_ds.append(directional_sim)

        curr_max = np.max(scores_num)
        candidates = {}
        for id, (num, ds) in enumerate(zip(scores_num, scores_ds)):
            if curr_max < num*1.1:
                candidates[id]=ds
        try:
            fittest_index = max(candidates, key=candidates.get)
        except ValueError:
            fittest_index = 0

        print()
        rotations={0:'origin',1:'90°',2:'180°',3:'270°'}
        print('rotation_degree:',rotations[fittest_index])
        print(image_path)


        #descs_b = descs_b_s[fittest_index]
        image_pil_b = batch_b_rotations[fittest_index][1]
        axes[1][0].clear()
        axes[1][0].set_axis_off()
        axes[1][0].title.set_text('B (rotated)')
        axes[1][0].imshow(image_pil_b)

        real_landmark_points=[]
        multi_curr_similarities = multi_curr_similarities_rotations[fittest_index]

        for landmark_id, curr_similarities in enumerate(multi_curr_similarities):
            center_b_candidates = []
            sims, idxes = torch.topk(curr_similarities.flatten(), num_sim_patches)
            for sim,idx in zip(sims, idxes):
                if sim > sim_threshold:
                    b_y_descs_coor, b_x_descs_coor = torch.div(idx, num_patches_b_rotations[id][1], rounding_mode='floor'), idx % \
                                                     num_patches_b_rotations[id][1]
                    center_b = ((b_x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                                (b_y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                    center_b_candidates.append([center_b[0].cpu().numpy(), center_b[1].cpu().numpy()])

            if len(center_b_candidates) > 1 and classify_landmark(center_b_candidates)['label']:
                real_landmark_points.append(landmark_id)

        a_landmark_points = a_landmark_points_rotations[fittest_index]
        b_landmark_points = b_landmark_points_rotations[fittest_index]
        landmark_ids = landmarks_ids_rotation[fittest_index]

        print('num_landmark_points:',len(real_landmark_points))
        print('num_confident_points:',len(a_landmark_points))

        landmark_coords = []
        for id, pt in enumerate(zip(a_landmark_points,b_landmark_points)):
            if landmark_ids[id] in real_landmark_points:

                landmark_coords.append(pt[0])

                patch_a= plt.Circle(pt[0], radius, color=color_map[id%len(color_map)])
                axes[0][0].add_patch(patch_a)
                label = axes[0][0].annotate(str(id), xy=pt[0], fontsize=6, ha="center")

                patch_b = plt.Circle(pt[1], radius, color=color_map[id%len(color_map)])
                axes[1][0].add_patch(patch_b)
                label = axes[1][0].annotate(str(id), xy=pt[1], fontsize=6, ha="center")

        a=np.asarray(landmark_coords)
        landmark_file_path = 'auto_proposed_landmarks.csv'
        if os.path.exists(landmark_file_path):
            os.remove(landmark_file_path)
        np.savetxt('auto_proposed_landmarks.csv',a,delimiter=',')

        axes[1][1].clear()
        axes[1][1].title.set_text('Placeholder')
        axes[1][1].set_axis_off()
        axes[1][1].imshow(np.asarray(Image.open('images/placeholder.jpg')))
        # if len(b_landmark_points) > 2:
        #     b_landmark_points = np.asarray(b_landmark_points)
        #     axes[1][1].clear()
        #     try:
        #         axes[1][1].imshow(robust_fit_and_draw_line(np.asarray(image_pil_b), b_landmark_points))
        #     except:
        #         axes[1][1].imshow(image_pil_b)
        # else:
        #     axes[1][1].imshow(image_pil_b)

        plt.draw()
        print("-----------")
        print(image_path_b)
        print('time:', time.time() - start)
        ptses = plt.ginput(num_ref_points, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None)


def max_two_values_and_indexes_heapq(arr):
    """
    Find the two maximum values and their indexes using heapq.
    :param arr: List of numbers.
    :return: Two maximum values and their indexes.
    """
    if len(arr) < 2:
        raise ValueError("The list must contain at least two elements.")

    # Use heapq.nlargest to get the largest two elements along with their indices
    largest = heapq.nlargest(2, enumerate(arr), key=lambda x: x[1])  # (Index, Value)
    return (largest[0][1], largest[0][0]), (largest[1][1], largest[1][0])

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fit_and_draw_line(image, points, line_color=(0, 255, 0), line_thickness=2, point_color=(0, 0, 255), point_size=5):
    """
    Fit a line that minimizes the perpendicular distance to the given points and draw it on the image.

    :param image: Input image (H x W x C).
    :param points: Array of 2D points (N x 2).
    :param line_color: Color of the line (default: green).
    :param line_thickness: Thickness of the line (default: 2).
    :param point_color: Color of the points (default: red).
    :param point_size: Size of the points (default: 5).
    :return: Image with the best-fit orthogonal regression line and points drawn.
    """
    # Compute the centroid of the points
    image = np.asarray(image)
    centroid = np.mean(points, axis=0)

    # Center the points by subtracting the centroid
    centered_points = points - centroid

    # Perform Singular Value Decomposition (SVD) to find the direction of the best-fit line
    _, _, vh = np.linalg.svd(centered_points)
    direction_vector = vh[0]  # The direction vector of the line (first right singular vector)

    # Define the line equation in terms of the direction vector and centroid
    # Line: x = centroid + t * direction_vector
    t = np.linspace(-500, 500, 1000)  # Generate points along the line
    line_points = centroid + t[:, None] * direction_vector

    # Draw the line on the image
    output_image = image.copy()
    x1, y1 = line_points[0].astype(int)
    x2, y2 = line_points[-1].astype(int)
    cv2.line(output_image, (x1, y1), (x2, y2), line_color, line_thickness)

    # Draw the original points
    for px, py in points:
        cv2.circle(output_image, (int(px), int(py)), point_size, point_color, -1)

    output_image = Image.fromarray(output_image)
    return output_image

def robust_fit_and_draw_line(image, points, threshold=15, line_color=(0, 255, 0), line_thickness=2, point_color=(0, 0, 255), point_size=5):
    """
    Fit a robust line using SVD while ignoring outliers, and draw it on the image.

    :param image: Input image (H x W x C).
    :param points: Array of (x, y) coordinates.
    :param threshold: Distance threshold to filter outliers.
    :param line_color: Color of the line (default: green).
    :param line_thickness: Thickness of the line (default: 2).
    :param point_color: Color of the points (default: red).
    :param point_size: Size of the points (default: 5).
    :return: Image with the robust best-fit line and points drawn.
    """
    # Iteratively remove outliers and fit the line
    points = np.array(points)
    max_iterations = 5
    inlier_mask = np.ones(len(points), dtype=bool)

    for _ in range(max_iterations):
        # Use only inlier points
        inlier_points = points[inlier_mask]

        # Compute the centroid and center the points
        centroid = np.mean(inlier_points, axis=0)
        centered_points = inlier_points - centroid

        # Perform SVD
        _, _, vh = np.linalg.svd(centered_points)
        direction_vector = vh[0]  # Direction of the best-fit line

        # Calculate distances of all points to the line
        line_normal = np.array([-direction_vector[1], direction_vector[0]])  # Perpendicular to the line
        distances = np.abs(np.dot(points - centroid, line_normal))

        # Update inliers
        inlier_mask = distances < threshold

    # Final line calculation using inliers
    inlier_points = points[inlier_mask]
    centroid = np.mean(inlier_points, axis=0)
    centered_points = inlier_points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    direction_vector = vh[0]

    # Define the line equation in terms of the direction vector and centroid
    t = np.linspace(-500, 500, 1000)
    line_points = centroid + t[:, None] * direction_vector

    # Draw the line on the image
    output_image = image.copy()
    x1, y1 = line_points[0].astype(int)
    x2, y2 = line_points[-1].astype(int)
    cv2.line(output_image, (x1, y1), (x2, y2), line_color, line_thickness)

    # Draw all points (inliers in green, outliers in red)
    for i, (px, py) in enumerate(points):
        color = (0, 255, 0) if inlier_mask[i] else (255, 0, 0)
        cv2.circle(output_image, (int(px), int(py)), point_size, color, -1)

    return output_image

def compute_hausdorff_distance(sequence1, sequence2):
    """
    Compute the Hausdorff distance between two sequences of points.

    :param sequence1: Array of (x, y) coordinates for sequence 1.
    :param sequence2: Array of (x, y) coordinates for sequence 2.
    :return: Hausdorff distance (lower means more similar).
    """
    sequence1 = np.array(sequence1)
    sequence2 = np.array(sequence2)
    forward_distance = directed_hausdorff(sequence1, sequence2)[0]
    backward_distance = directed_hausdorff(sequence2, sequence1)[0]
    return max(forward_distance, backward_distance)

def draw_grid(image,save_path):
    """
    Draw a grid on the image to separate it into four areas: NW, NE, SW, SE.

    :param image: Input image (H x W x C).
    :return: Image with grid and labeled areas.
    """
    # Get image dimensions
    image = np.asarray(image)
    height, width = image.shape[:2]

    # Compute the center of the image
    center_x, center_y = width // 2, height // 2

    # Draw vertical and horizontal lines
    grid_image = image.copy()
    cv2.line(grid_image, (center_x, 0), (center_x, height), (0, 255, 0), 2)  # Vertical line
    cv2.line(grid_image, (0, center_y), (width, center_y), (0, 255, 0), 2)  # Horizontal line

    # Add labels for areas
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 0, 0)  # Blue for text

    cv2.putText(grid_image, "NW", (center_x // 2, center_y // 2), font, font_scale, color, thickness)
    cv2.putText(grid_image, "NE", (3 * center_x // 2, center_y // 2), font, font_scale, color, thickness)
    cv2.putText(grid_image, "SW", (center_x // 2, 3 * center_y // 2), font, font_scale, color, thickness)
    cv2.putText(grid_image, "SE", (3 * center_x // 2, 3 * center_y // 2), font, font_scale, color, thickness)

    cv2.imwrite(save_path, grid_image)


def calculate_overall_direction(sequence):
    """
    Calculate the overall direction of a sequence of points.

    :param sequence: List of (x, y) coordinates.
    :return: Normalized mean direction vector (overall direction).
    """
    sequence = np.array(sequence)
    displacements = sequence[1:] - sequence[:-1]  # Pairwise displacements
    mean_direction = np.mean(displacements, axis=0)  # Mean displacement vector
    normalized_direction = mean_direction / np.linalg.norm(mean_direction)  # Normalize to unit vector
    return normalized_direction


def compare_directions(seq_a, seq_b):
    """
    Compare the overall directions of two sequences using cosine similarity.

    :param seq_a: List of (x, y) coordinates for sequence A.
    :param seq_b: List of (x, y) coordinates for sequence B.
    :return: Cosine similarity between the overall directions of the sequences.
    """
    dir_a = calculate_overall_direction(seq_a)
    dir_b = calculate_overall_direction(seq_b)
    cosine_similarity = np.dot(dir_a, dir_b)
    return cosine_similarity

def classify_landmark(candidate_points, eps=20, min_samples=2):
    """
    Classify a query point based on candidate matches and also return the number of clusters.

    Parameters:
        candidate_points (np.ndarray): An array of shape (N, 2) containing the (x, y)
                                       coordinates of N candidate points on image B.
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other (adjust based on your image scale).
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point.

    Returns:
        result (dict): A dictionary containing:
            - 'label': "landmark" if the candidate points form a single cluster,
                       "non-landmark" if they form multiple clusters,
                       or "uncertain" if no clear cluster is formed.
            - 'num_clusters': The number of clusters detected (ignoring noise points).
    """
    # Run DBSCAN on the candidate points
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clustering.fit(candidate_points)
    labels = clustering.labels_  # DBSCAN labels: -1 means noise

    # Count the number of clusters (ignoring noise)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    num_clusters = len(unique_labels)

    # Decide based on the number of clusters
    if num_clusters == 1:
        label = True  # e.g., nose-like area
    elif num_clusters > 1:
        label = False  # e.g., eyes-like area
    else:
        # If no clusters found, you may decide to return a default label or "uncertain"
        label = False

    return {"label": label, "num_clusters": num_clusters}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate similarity inspection between two images.')
    parser.add_argument('--image_a', type=str, default="images/cat_face/cat_5_rotated90.jpg", help='Path to the first image')
    parser.add_argument('--image_b_folder', type=str, default="images/test/", help='Path to the second image.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=8, type=int, help="""stride of first convolution layer. 
                                                                    small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                              Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                              vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                       options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--num_sim_patches', default=30, type=int, help="number of closest patches to show.")
    parser.add_argument('--num_ref_points', default=300, type=int, help="number of reference points to show.")
    parser.add_argument('--landmark_file', default='curl_test/cat_5_rotated90_mask.png', type=str, help="landmarks file.")

    args = parser.parse_args()

    with torch.no_grad():

        show_similarity_interactive(args.image_a, args.image_b_folder, args.landmark_file, args.num_ref_points, args.load_size, args.layer,
                                    args.facet, args.bin,
                                    args.stride, args.model_type, args.num_sim_patches)