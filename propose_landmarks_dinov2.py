import argparse
import os
import torch
from sklearn.cluster import DBSCAN
from extractor_dinov2 import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import cv2
from PIL import Image
from scipy.spatial import procrustes
import random
from termcolor import colored
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


def show_similarity_interactive(image_path_a: str, image_folder_path_b: str, mask_file, num_ref_points: int, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 14, model_type: str = 'dinov2_vits14',
                                num_sim_patches: int = 1, sim_threshold: float = 0.95, num_rotations: int = 4):
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
    patch_size = extractor.model.patch_embed.patch_size[0] if isinstance(extractor.model.patch_embed.patch_size,tuple) else extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size

    if mask_file:
        mask = cv2.resize(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE),(load_size_a[1],load_size_a[0]))
        coords = cv2.findNonZero(mask)
        if coords is not None:
            coords_list = coords.reshape(-1, 2).tolist()
            landmarks = random.sample(coords_list, num_ref_points)
        else:
            landmarks = []
    else:
        coords = cv2.findNonZero(np.ones((load_size_a[0],load_size_a[1])))
        coords_list = coords.reshape(-1, 2).tolist()
        landmarks = random.sample(coords_list, num_ref_points)


    fig, axes = plt.subplots(2, 2, figsize=(30, 30))

    # axes[1][1].title.set_text('Placeholder')
    # axes[1][1].set_axis_off()
    # axes[1][1].imshow(Image.open('../data/images/placeholder.jpg'))

    all_files = os.listdir(image_folder_path_b)
    images = [file for file in all_files if file.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort()
    #random.shuffle(images)
    count_all=0
    count_correct=0
    count_mistake=0
    failure_case=[]
    for image_path in images:
        start=time.time()
        image_path_b = os.path.join(image_folder_path_b, image_path)
        batch_b_rotations = extractor.preprocess(image_path_b, load_size, rotate=num_rotations)

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

        axes[1][1].clear()
        axes[1][1].set_axis_off()
        axes[1][1].title.set_text('Marked Points on Original')
        axes[1][1].imshow(batch_b_rotations[0][1])


        ptses = np.asarray(landmarks)
        a_landmark_points_rotations = []
        b_landmark_points_rotations = []
        landmarks_ids_rotation = []
        multi_curr_similarities_rotations = []

        #test for remote control
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
            try:
                directional_sim = procrustes_analysis(a_landmark_points, b_landmark_points)
            except ValueError:
                directional_sim = 1.2
            #directional_sim = directional_sim if not np.isnan(directional_sim) else 0
            print("[", num_landmark_points,",",directional_sim,"] ",end="")
            scores_num.append(len(b_landmark_points))
            scores_ds.append(directional_sim)
        print()
        fittest_index = scores_num.index(max(scores_num))

        #Adjust result by shape similarity?
        # curr_max = np.max(scores_num)
        # candidates = {}
        # for id, (num, ds) in enumerate(zip(scores_num, scores_ds)):
        #     if curr_max < num*1.1:
        #         candidates[id]=ds
        # try:
        #     fittest_index = min(candidates, key=candidates.get)
        # except ValueError:
        #     fittest_index = 0

        # if fittest_index != scores_num.index(curr_max):
        #
        #     print("Adjusted by Shapesim!")
        #end



        rotation_degrees = [angle for angle in np.linspace(0, 360, num_rotations, endpoint=False)]
        rotations = {}
        for i,rotation_degree in enumerate(rotation_degrees):
            rotations[i]=rotation_degree

        print('rotation_degree:',rotations[fittest_index])
        print(image_path)

        # Testing skull images
        name = image_path.replace('image_', '')
        name = name.replace('.png', '')
        name = int(name)
        if 1 <= name <= 35:
            gt = 3
        elif 36 <= name <= 96:
            gt = 2
        elif 97 <= name <= 145:
            gt = 1
        else:
            gt = 0
        # end

        print('Ground_truth:',rotations[gt])

        if gt==fittest_index:
            # if fittest_index != scores_num.index(curr_max):
            #     count_correct+=1
            #     print(colored('Corrected by Shapesim!', 'yellow'))
            count_all+=1
            print(colored('Correct', 'green'))
        else:
            # if scores_num.index(curr_max) == gt:
            #     count_mistake+=1
            #     print(colored('Mistaken by Shapesim!', 'yellow'))
            #     failure_case.append(image_path+': Mistaken by Shapesim')
            # else:
            failure_case.append(image_path+': Inherent Failure')
            print(colored('Incorrect', 'red'))

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



        output_reference = []
        rotated_coords = []
        count = 0
        for id, pt in enumerate(zip(a_landmark_points,b_landmark_points)):
            if landmark_ids[id] in real_landmark_points:

                patch_a= plt.Circle(pt[0], radius, color=color_map[count%len(color_map)])
                axes[0][0].add_patch(patch_a)
                output_reference.append(pt[0])
                label = axes[0][0].annotate(str(count), xy=pt[0], fontsize=6, ha="center")

                patch_b = plt.Circle(pt[1], radius, color=color_map[count%len(color_map)])
                axes[1][0].add_patch(patch_b)
                rotated_coords.append(pt[1])
                label = axes[1][0].annotate(str(count), xy=pt[1], fontsize=6, ha="center")

                count+=1

        output_target = []
        landmarks_on_original = rotate_landmarks(image_pil_b.size,rotated_coords,rotations[fittest_index])
        for id,pt in enumerate(landmarks_on_original):
            patch_d =plt.Circle(pt,radius,color=color_map[id%len(color_map)])
            axes[1][1].add_patch(patch_d)
            output_target.append(pt)

            label = axes[1][1].annotate(str(id), xy=pt, fontsize=6, ha="center")

        # np.savetxt('landmarks_A.csv',output_reference,delimiter=',')
        # np.savetxt('landmarks_B.csv',output_target,delimiter=',')
        plt.draw()

        print('time:', time.time() - start)
        print("-----------")
        ptses = plt.ginput(num_ref_points, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None)
    print("Accuracy:",count_all/len(images))
    print("Correct:",count_correct)
    print("Mistake:",count_mistake)
    np.savetxt('failure_cases.txt',failure_case,delimiter=',',fmt='%s')
def procrustes_analysis(points_A, points_B):
    """
    Perform Procrustes analysis to measure shape similarity.

    :param points_A: Array of (x, y) coordinates for reference shape.
    :param points_B: Array of (x, y) coordinates for detected shape.
    :return: Procrustes distance (lower means more similar).
    """
    # Ensure points are NumPy arrays
    points_A = np.array(points_A)
    points_B = np.array(points_B)

    # Perform Procrustes alignment
    mtx1, mtx2, disparity = procrustes(points_A, points_B)
    return disparity


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
    parser.add_argument('--image_a', type=str, default="../data/images/landmark_files/skull.png",
                        help='Path to the first image')
    parser.add_argument('--mask_file', default='../data/images/landmark_files/skull_mask.png', type=str,
                        help="landmarks file.")
    parser.add_argument('--image_b_folder', type=str, default="../data/images/skull/",
                        help='Path to the second image.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=14, type=int, help="""stride of first convolution layer. 
                                                                    small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dinov2_vits14', type=str,
                        help="""type of model to extract. 
                              Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                              vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                       options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--num_sim_patches', default=20, type=int, help="number of closest patches to show.")
    parser.add_argument('--num_ref_points', default=300, type=int, help="number of reference points to show.")
    parser.add_argument('--sim_threshold', default=0.95, type=float, help="similarity threshold.")
    parser.add_argument('--num_rotation', default=4, type=int, help="number of test rotations, 4 or 8")
    parser.add_argument('--landmark_save', default='output_landmarks.csv', type=str,
                        help="CSV file to save landmark points.")
    args = parser.parse_args()

    with torch.no_grad():
        landmarks = show_similarity_interactive(args.image_a, args.image_b_folder, args.mask_file, args.num_ref_points,
                                                args.load_size,
                                                args.layer, args.facet, args.bin,
                                                args.stride, args.model_type, args.num_sim_patches,
                                                args.sim_threshold, args.num_rotation)

