import argparse
import os
import torch
from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import csv
import cv2
from PIL import Image
from sklearn.decomposition import PCA

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


def show_similarity_interactive(image_path_a: str, image_folder_path_b: str, landmarks, num_ref_points: int,
                                load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8',
                                num_sim_patches: int = 1, sim_threshold: float = 0.50):
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
    color_map = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes[0][0].title.set_text('A (reference)')
    axes[0][1].title.set_text('B (Original)')
    axes[1][0].title.set_text('B (Rotated)')
    axes[1][1].title.set_text('B (Reflection Axis)')

    for image_path in sorted(os.listdir(image_folder_path_b))[::-1]:
        start = time.time()
        image_path_b = os.path.join(image_folder_path_b, image_path)

        batch_b_rotations = extractor.preprocess(image_path_b, load_size, rotate=True)

        descs_b_s = []
        for batch in batch_b_rotations:
            descs_b_s.append(extractor.extract_descriptors(batch[0].to(device), layer, facet, bin, include_cls=True))

        num_patches_b, load_size_b = extractor.num_patches, extractor.load_size

        # plot
        # woops, it seems [1][0] and [1][1] are exchanged. But for the least effort, let's just remain as it is.
        [axi.set_axis_off() for axi in axes.ravel()]
        visible_patches = []
        radius = patch_size // 2
        # plot image_a and the chosen patch. if nothing marked chosen patch is cls patch.
        axes[0][0].imshow(image_pil_a)

        # Song_implemented: For visualizing the similarity map in image_a as well.
        # a_to_a_similarities = chunk_cosine_sim(descs_a, descs_a)
        # a_to_a_curr_similarities = a_to_a_similarities[0, 0, 0, 1:]
        # a_to_a_curr_similarities = a_to_a_curr_similarities.reshape(num_patches_a)

        # end

        # calculate and plot similarity between image1 and image2 descriptors

        ptses = np.asarray(landmarks)
        b_num_landmark_points = []

        for descs_b_rot in descs_b_s:
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
                curr_similarities = curr_similarities.reshape(num_patches_b)

                multi_curr_similarities.append(curr_similarities)

            # get and draw most similar points
            num_landmark_points_b = 0
            for color_id, curr_similarities in enumerate(multi_curr_similarities):
                sims, idxs = torch.topk(curr_similarities.flatten(), num_sim_patches)
                for idx, sim in zip(idxs, sims):
                    if sim > sim_threshold:
                        num_landmark_points_b += 1
            b_num_landmark_points.append(num_landmark_points_b)
            print(num_landmark_points_b)

        fittest_index = b_num_landmark_points.index(max(b_num_landmark_points))
        descs_b = descs_b_s[fittest_index]
        image_pil_b = batch_b_rotations[fittest_index][1]

        axes[0][1].imshow(batch_b_rotations[0][1])

        similarities = chunk_cosine_sim(descs_a, descs_b)
        curr_similarities = similarities[0, 0, 0, 1:]  # similarity to all spatial descriptors, without cls token
        curr_similarities = curr_similarities.reshape(num_patches_b)
        # axes[1][1].imshow(curr_similarities.cpu().numpy(), cmap='jet')

        # plot image_b and the closest patch in it to the chosen patch in image_a
        axes[1][0].clear()
        axes[1][0].set_axis_off()
        axes[1][0].title.set_text('B (rotated)')
        axes[1][0].imshow(image_pil_b)
        sims, idxs = torch.topk(curr_similarities.flatten(), num_sim_patches)
        for idx, sim in zip(idxs, sims):
            y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
            center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                      (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
            patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
            axes[1][0].add_patch(patch)
            visible_patches.append(patch)
        plt.draw()

        # start interactive loop
        # get input point from user
        # fig.suptitle('Select a point on the left image. \n Right click to stop.', fontsize=16)

        ptses = np.asarray(landmarks)

        while len(ptses) == len(landmarks):
            # reset previous marks
            for patch in visible_patches:
                patch.remove()
                visible_patches = []

            multi_curr_similarities = []
            for idx, pts in enumerate(ptses):
                y_coor, x_coor = int(pts[1]), int(pts[0])
                new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
                new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
                y_descs_coor = int(new_H / load_size_a[0] * y_coor)
                x_descs_coor = int(new_W / load_size_a[1] * x_coor)

                raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
                reveled_desc_idx_including_cls = raveled_desc_idx + 1

                curr_similarities = similarities[0, 0, reveled_desc_idx_including_cls, 1:]
                (sim, idx) = torch.topk(curr_similarities, num_sim_patches)

                if sim > sim_threshold:
                    # draw chosen point
                    center_a = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                                (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                    patch_a = plt.Circle(center, radius, color=color_map[idx % len(color_map)])

                    axes[0][0].add_patch(patch_a)
                    visible_patches.append(patch_a)

                curr_similarities = curr_similarities.reshape(num_patches_b)
                multi_curr_similarities.append(curr_similarities)

            b_landmark_points = np.empty((0, 2))
            # get and draw most similar points
            for color_id, curr_similarities in enumerate(multi_curr_similarities):
                sims, idxs = torch.topk(curr_similarities.flatten(), num_sim_patches)
                color = color_map[color_id % len(color_map)]
                for idx, sim in zip(idxs, sims):
                    if sim > sim_threshold:
                        y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
                        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                                  (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                        b_landmark_points = np.vstack(
                            (b_landmark_points, np.asarray([center[0].cpu().numpy(), center[1].cpu().numpy()])))
                        patch = plt.Circle(center, radius, color=color)
                        axes[1][0].add_patch(patch)
                        visible_patches.append(patch)
            print(len(b_landmark_points))

            print(check_sequence_direction(ptses,b_landmark_points))

            # returned =
            if len(b_landmark_points) > 0:
                axes[1][1].imshow(fit_and_draw_line(image_pil_b, b_landmark_points))
                # get input point from use
                print('time:', time.time() - start)
            else:
                axes[1][1].imshow(image_pil_b)

            ptses = np.asarray(plt.ginput(num_ref_points, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))


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
    # plt.imshow(output_image)
    # plt.waitforbuttonpress()
    output_image = Image.fromarray(output_image)
    return output_image


def check_sequence_direction(sequence1, sequence2, threshold=0.8):
    """
    Check if two sequences of points follow the same general direction, allowing for noise or swapped coordinates.

    :param sequence1: Array of (x, y) coordinates for sequence 1.
    :param sequence2: Array of (x, y) coordinates for sequence 2.
    :param threshold: Cosine similarity threshold for "same direction" (default: 0.9).
    :return: Boolean indicating if the sequences follow the same direction and the cosine similarity value.
    """
    # Convert sequences to numpy arrays
    sequence1 = np.array(sequence1)
    sequence2 = np.array(sequence2)

    # Perform PCA to extract the principal direction of each sequence
    pca1 = PCA(n_components=2).fit(sequence1)
    pca2 = PCA(n_components=2).fit(sequence2)

    # Get the first principal component (direction vector)
    direction1 = pca1.components_[0]
    direction2 = pca2.components_[0]

    # Normalize the direction vectors
    direction1 /= np.linalg.norm(direction1)
    direction2 /= np.linalg.norm(direction2)

    # Compute cosine similarity in both forward and reverse directions
    cosine_similarity = np.dot(direction1, direction2)
    cosine_similarity_reversed = np.dot(direction1, -direction2)

    # Take the maximum similarity to account for reversed sequences
    max_similarity = max(cosine_similarity, cosine_similarity_reversed)

    # Check if the direction is similar
    is_similar = max_similarity >= threshold

    return is_similar, max_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate similarity inspection between two images.')
    parser.add_argument('--image_a', type=str,
                        default="LabData/3D dataset/semantic segmentation dataset/images/image_00001.png",
                        help='Path to the first image')
    parser.add_argument('--image_b_folder', type=str,
                        default="LabData/3D dataset/semantic segmentation dataset/images/",
                        help='Path to the second image.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=8, type=int, help="""stride of first convolution layer. 
                                                                    small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                              Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                              vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                       options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=9, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--num_sim_patches', default=1, type=int, help="number of closest patches to show.")
    parser.add_argument('--num_ref_points', default=30, type=int, help="number of reference points to show.")
    parser.add_argument('--landmark_file', default='landmarks.csv', type=str, help="landmarks file.")

    args = parser.parse_args()

    with torch.no_grad():
        with open(args.landmark_file, 'r') as f:
            landmark_file = csv.reader(f)
            landmarks = []
            for landmark in landmark_file:
                landmarks.append((float(landmark[0]), float(landmark[1])))

        show_similarity_interactive(args.image_a, args.image_b_folder, landmarks, args.num_ref_points, args.load_size,
                                    args.layer,
                                    args.facet, args.bin,
                                    args.stride, args.model_type, args.num_sim_patches)