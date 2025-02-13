import argparse
import os
import torch

from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import cv2
from matplotlib.patches import ConnectionPatch
#from scipy.spatial import distance

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


def show_similarity_interactive(image_path_a: str, image_folder_path_b: str, num_ref_point_pairs: int, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8',
                                num_sim_patches: int = 1, ):
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
    color_map = np.random.rand(num_ref_point_pairs, 3)
    cos = torch.nn.CosineSimilarity(dim=0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes[0][0].title.set_text('Obj A and Query')
    axes[0][1].title.set_text('Similarity Heat Map A')
    axes[1][1].title.set_text('Similarity Heat Map B')
    axes[1][0].title.set_text('Obj B and Result')

    for image_path in sorted(os.listdir(image_folder_path_b)):
        image_path_b = os.path.join(image_folder_path_b, image_path)
        image_batch_b, image_pil_b = extractor.preprocess(image_path_b, load_size)
        descs_b = extractor.extract_descriptors(image_batch_b.to(device), layer, facet, bin, include_cls=True)
        num_patches_b, load_size_b = extractor.num_patches, extractor.load_size

        # plot
        #woops, it seems [1][0] and [1][1] are exchanged. But for the least effort, let's just remain as it is.
        axes[0][0].clear()
        [axi.set_axis_off() for axi in axes.ravel()]
        visible_patches = []
        radius = patch_size // 2
        # plot image_a and the chosen patch. if nothing marked chosen patch is cls patch.
        axes[0][0].imshow(image_pil_a)

        # Song_implemented: For visualizing the similarity map in image_a as well.
        a_to_a_similarities = chunk_cosine_sim(descs_a, descs_a)
        a_to_a_curr_similarities = a_to_a_similarities[0, 0, 0, 1:]
        a_to_a_curr_similarities = a_to_a_curr_similarities.reshape(num_patches_a)
        axes[0][1].imshow(a_to_a_curr_similarities.cpu().numpy(), cmap='jet')
        #end

        # calculate and plot similarity between image1 and image2 descriptors
        similarities = chunk_cosine_sim(descs_a, descs_b)
        curr_similarities = similarities[0, 0, 0, 1:]  # similarity to all spatial descriptors, without cls token
        curr_similarities = curr_similarities.reshape(num_patches_b)
        axes[1][1].imshow(curr_similarities.cpu().numpy(), cmap='jet')



        # plot image_b and the closest patch in it to the chosen patch in image_a
        axes[1][0].clear()
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
        #fig.suptitle('Select a point on the left image. \n Right click to stop.', fontsize=16)
        pairs = []
        for _ in range(num_ref_point_pairs):
            pts_pair = np.asarray(plt.ginput(2, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
            pairs.append(pts_pair)

        while len(pairs) == num_ref_point_pairs:
            # reset previous marks
            for patch in visible_patches:
                patch.remove()
                visible_patches = []

            multi_vector_similarities = []
            multi_current_pair_similarities = []
            #visualize point paris on reference image

            for idx, point_pair in enumerate(pairs):
                coords = []
                pair_similarities = []
                descs_points = []
                for pts in point_pair:
                    y_coor, x_coor = int(pts[1]), int(pts[0])
                    new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
                    new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
                    y_descs_coor = int(new_H / load_size_a[0] * y_coor)
                    x_descs_coor = int(new_W / load_size_a[1] * x_coor)

                    #find the most similar point on target image
                    raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
                    reveled_desc_idx_including_cls = raveled_desc_idx + 1
                    curr_similarities = similarities[0, 0, reveled_desc_idx_including_cls, 1:]
                    descs_points.append(descs_a[0,0,reveled_desc_idx_including_cls,:])
                    #curr_similarities = curr_similarities.reshape(num_patches_b)
                    pair_similarities.append(curr_similarities)

                    # draw chosen point
                    center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                              (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                    coords.append(center)
                    patch = plt.Circle(center, radius, color=color_map[idx % len(color_map)])
                    axes[0][0].add_patch(patch)
                    visible_patches.append(patch)

                multi_current_pair_similarities.append(pair_similarities)
                multi_vector_similarities.append(descs_points[1] - descs_points[0])

                #Draw segments between a pair of points
                connection = ConnectionPatch(coords[0],coords[1],coordsA='data')
                connection.set_color(color_map[idx % len(color_map)])
                connection.set_linewidth(2)
                connection.set_linestyle(':')
                axes[0][0].add_patch(connection)
                visible_patches.append(connection)
            plt.draw()

            point_pairs_b = []
            # get and draw most similar point pairs
            for id, current_pair_similarities in enumerate(multi_current_pair_similarities):
                candidates=[]

                for point_similarities in current_pair_similarities:
                    sims, idxs = torch.topk(point_similarities, num_sim_patches)
                    candidates.append(idxs)

                baseline_vector = multi_vector_similarities[id]
                max = cos(descs_b[0, 0, candidates[1][0], :]-descs_b[0, 0, candidates[0][0],:],baseline_vector)
                a_idx, b_idx = candidates[0][0], candidates[1][0]
                for point_a_idx in candidates[0]:
                    for point_b_idx in candidates[1]:
                        curr = cos(descs_b[0,0,point_b_idx,:]-descs_b[0,0,point_a_idx,:],baseline_vector)
                        if curr>max:
                            max = curr
                            a_idx, b_idx = point_a_idx, point_b_idx
                point_pairs_b.append((a_idx, b_idx))


            for id, point_pair in enumerate(point_pairs_b):
                color = color_map[id % len(color_map)]
                coords = []
                for point in point_pair:
                    idx = point
                    y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
                    x = (x_descs_coor - 1) * stride + stride + patch_size // 2 - .5
                    y = (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5

                    coords.append((x.detach().item(),y.detach().item()))
                    patch = plt.Circle((x,y), radius, color=color)
                    axes[1][0].add_patch(patch)
                    visible_patches.append(patch)

                connection = ConnectionPatch(coords[0], coords[1], coordsA='data')
                connection.set_color(color)
                connection.set_linewidth(2)
                connection.set_linestyle(':')
                axes[1][0].add_patch(connection)
                visible_patches.append(connection)

            plt.draw()

            pairs = []
            for _ in range(num_ref_point_pairs):
                pts_pair = np.asarray(plt.ginput(2, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
                pairs.append(pts_pair)

def apply_geometric_constraints(pair_a, pair_b, target_points):
    """Apply geometric constraints based on the distance and angle between points."""
    # Compute the vector and distance between points in the reference pair
    ref_vector = pair_a - pair_b
    ref_distance = np.linalg.norm(ref_vector)

    valid_pairs = []

    for i, p1 in enumerate(target_points):
        for j, p2 in enumerate(target_points):
            if i == j:
                continue
            tgt_vector = p1 - p2
            tgt_distance = np.linalg.norm(tgt_vector)

            # Check if distances match within a threshold
            if abs(tgt_distance - ref_distance) < 0.1 * ref_distance:
                valid_pairs.append((i, j))

    return valid_pairs

def validate_geometric_constraint(pair1, pair2, threshold=0.1):
    """
    Validates the geometric constraint between two pairs of points.
    Ensures relative distances/angles are consistent.
    """
    distance1 = np.linalg.norm(pair1[1] - pair1[0])
    distance2 = np.linalg.norm(pair2[1] - pair2[0])
    return abs(distance1 - distance2) < threshold

def project_points(source_points, target_points):
    """
    Projects points from the source image to the target image using a homography matrix.
    :param source_points: Array of points in the source image.
    :param target_points: Array of corresponding points in the target image.
    :return: Projected points in the target image.
    """
    matrix, _ = cv2.findHomography(source_points, target_points, cv2.RANSAC)
    projected_points = cv2.perspectiveTransform(np.float32(source_points).reshape(-1, 1, 2), matrix)
    return projected_points

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate similarity inspection between two images.')
    parser.add_argument('--image_a', type=str, default="images/birds/ramune.png", help='Path to the first image')
    parser.add_argument('--image_b_folder', type=str, default="images/birds/", help='Path to the second image.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                                    small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vitb8', type=str,
                        help="""type of model to extract. 
                              Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                              vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                       options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--num_sim_patches', default=11, type=int, help="number of closest patches to show.")
    parser.add_argument('--num_ref_point_pairs', default=2, type=int, help="number of reference points to show.")

    args = parser.parse_args()

    with torch.no_grad():

        show_similarity_interactive(args.image_a, args.image_b_folder, args.num_ref_point_pairs, args.load_size, args.layer, args.facet, args.bin,
                                    args.stride, args.model_type, args.num_sim_patches)
