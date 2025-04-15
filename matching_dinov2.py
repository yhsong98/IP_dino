import argparse
import torch
from sklearn.cluster import DBSCAN
from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
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


def show_similarity_interactive(image_path_a: str, image_path_b: str, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 4, model_type: str = 'dinov2_vits14',
                                num_sim_patches: int = 1, check_unique: bool=False):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.p
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    image_batch_b, image_pil_b = extractor.preprocess(image_path_b, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size
    descs_b = extractor.extract_descriptors(image_batch_b.to(device), layer, facet, bin, include_cls=True)
    num_patches_b, load_size_b = extractor.num_patches, extractor.load_size

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(30,30))

    #woops, it seems [1][0] and [1][1] are exchanged. But for the least effort, let's just remain as it is.
    axes[0][0].title.set_text('Obj A and Query')
    axes[0][1].title.set_text('Similarity Heat Map A')
    axes[1][1].title.set_text('Similarity Heat Map B')
    axes[1][0].title.set_text('Obj B and Result')

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
    axes[1][0].imshow(image_pil_b)
    sims, idxs = torch.topk(curr_similarities.flatten(), num_sim_patches)
    for idx, sim in zip(idxs, sims):
        y_descs_coor, x_descs_coor = torch.div(idx, num_patches_b[1], rounding_mode='floor'), idx % num_patches_b[1]
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                  (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        patch = plt.Circle((center[0].cpu().numpy(),center[1].cpu().numpy()), radius, color=(1, 0, 0, 0.75))
        axes[1][0].add_patch(patch)
        visible_patches.append(patch)
    plt.draw()

    # start interactive loop
    # get input point from user
    #fig.suptitle('Select a point on the left image. \n Right click to stop.', fontsize=16)
    plt.draw()
    pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
    while len(pts) == 1:
        print('pts:', pts)
        y_coor, x_coor = int(pts[0, 1]), int(pts[0, 0])
        new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
        new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
        y_descs_coor = int(new_H / load_size_a[0] * y_coor)
        x_descs_coor = int(new_W / load_size_a[1] * x_coor)

        # reset previous marks
        for patch in visible_patches:
            patch.remove()
            visible_patches = []

        # draw chosen point
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                  (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
        axes[0][0].add_patch(patch)
        visible_patches.append(patch)

        # get and draw current similarities
        raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
        reveled_desc_idx_including_cls = raveled_desc_idx + 1

        # Song_implemented: For drawing heatmap on image_a
        a_to_a_curr_similarities = a_to_a_similarities[0, 0, reveled_desc_idx_including_cls, 1:]
        a_to_a_curr_similarities = a_to_a_curr_similarities.reshape(num_patches_a)
        axes[0][1].imshow(a_to_a_curr_similarities.cpu().numpy(), cmap='jet')
        # end

        curr_similarities = similarities[0, 0, reveled_desc_idx_including_cls, 1:]
        curr_similarities = curr_similarities.reshape(num_patches_b)
        axes[1][1].imshow(curr_similarities.cpu().numpy(), cmap='jet')



        # get and draw most similar points
        sims, idxs = torch.topk(curr_similarities.flatten(), num_sim_patches)
        b_center = []
        for idx, sim in zip(idxs, sims):
            y_descs_coor, x_descs_coor = torch.div(idx, num_patches_b[1], rounding_mode='floor'), idx % num_patches_b[1]
            center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                      (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
            #patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
            b_center.append([center[0].cpu().numpy(), center[1].cpu().numpy()])

        plt.draw()

        if check_unique:
            result = classify_landmark(b_center)
            print('landmark?',result)
            if result['label']:
                color = (0, 1, 0, 0.75)
            else:
                color = (1, 0, 0, 0.75)
            for center in b_center:
                patch = plt.Circle(center, radius, color=color)
                axes[1][0].add_patch(patch)
                visible_patches.append(patch)
        else:
            patch = plt.Circle(b_center[0], radius, color=(1, 1, 0, 0.75))
            axes[1][0].add_patch(patch)
            visible_patches.append(patch)

            # print(sim)
            # print(b_center)
        # get input point from user
        fig.suptitle('Select a point on the left image', fontsize=16)
        plt.draw()
        pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))


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


def classify_landmark(candidate_points, eps=20, min_samples=1):
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
    parser.add_argument('--image_a', type=str, default="../data/images/landmark_files/cat_5.jpg", help='Path to the first image')
    parser.add_argument('--image_b', type=str, default="../data/images/wild/flickr_wild_001702.jpg", help='Path to the second image.')
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
    parser.add_argument('--check_unique', default=False, type=str2bool, help="check if there are multiple possible areas")
    parser.add_argument('--num_sim_patches', default=10, type=int, help="number of closest patches to show.")

    args = parser.parse_args()

    with torch.no_grad():

        show_similarity_interactive(args.image_a, args.image_b, args.load_size, args.layer, args.facet, args.bin,
                                    args.stride, args.model_type, args.num_sim_patches, args.check_unique)
