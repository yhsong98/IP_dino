import cv2  # For video handling
import torch
import os
from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: a tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
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


def process_video(video_path, extractor, load_size):
    """Reads the video and preprocesses frames."""
    frames = extractor.v_preprocess(video_path)
    return frames


def show_similarity_with_video(image_path_a, video_path_b, num_ref_points, load_size=224, layer=11, facet='key',
                               bin=False, stride=4, model_type='dino_vits8', num_sim_patches=1):
    """
    Finds similarity between a descriptor in one image to all descriptors in a video.
    :param image_path_a: Path to the first image.
    :param video_path_b: Path to the video.
    :param load_size: Size of the smaller edge of loaded images. If None, does not resize.
    :param layer: Layer to extract descriptors from.
    :param facet: Facet to extract descriptors from.
    :param bin: If True, use a log-binning descriptor.
    :param stride: Stride of the model.
    :param model_type: Type of model to extract descriptors from.
    :param num_sim_patches: Number of most similar patches to plot.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)

    # Process Image A
    patch_size = extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size

    # Process Video B
    video_frames = process_video(video_path_b, extractor, load_size)

    # Prepare Interactive Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes[0][0].title.set_text('Obj A and Query')
    axes[0][1].title.set_text('Similarity Heat Map A')
    axes[1][1].title.set_text('Similarity Heat Map Frame')
    axes[1][0].title.set_text('Video Frame and Result')

    visible_patches = []
    radius = patch_size // 2

    axes[0][0].imshow(image_pil_a)
    a_to_a_similarities = chunk_cosine_sim(descs_a, descs_a)
    a_to_a_curr_similarities = a_to_a_similarities[0, 0, 0, 1:]
    a_to_a_curr_similarities = a_to_a_curr_similarities.reshape(num_patches_a)
    axes[0][1].imshow(a_to_a_curr_similarities.cpu().numpy(), cmap='jet')

    ptses = np.asarray(plt.ginput(num_ref_points, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
    while len(ptses) == num_ref_points:
        multi_curr_similarities = []

        for idx, pts in enumerate(ptses):
            y_coor, x_coor = int(pts[1]), int(pts[0])
            new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
            new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
            y_descs_coor = int(new_H / load_size_a[0] * y_coor)
            x_descs_coor = int(new_W / load_size_a[1] * x_coor)
            center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                      (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
            patch = plt.Circle(center, radius, color='r')
            axes[0][0].add_patch(patch)
            visible_patches.append(patch)

            raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
            reveled_desc_idx_including_cls = raveled_desc_idx + 1

            a_to_a_curr_similarities = a_to_a_similarities[0, 0, reveled_desc_idx_including_cls, 1:]
            a_to_a_curr_similarities = a_to_a_curr_similarities.reshape(num_patches_a)
            axes[0][1].imshow(a_to_a_curr_similarities.cpu().numpy(), cmap='jet')

            for frame_tensor, frame in video_frames:
                descs_b = extractor.extract_descriptors(frame_tensor.to(device), layer, facet, bin, include_cls=True)
                num_patches_b, load_size_b = extractor.num_patches, extractor.load_size
                curr_similarities = chunk_cosine_sim(descs_a, descs_b)[0, 0, reveled_desc_idx_including_cls, 1:]
                curr_similarities = curr_similarities.reshape(num_patches_b)

                axes[1][1].imshow(curr_similarities.cpu().numpy(), cmap='jet')
                sims, idxs = torch.topk(curr_similarities.flatten(), num_sim_patches)
                for idx, sim in zip(idxs, sims):
                    y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
                    center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                              (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                    patch = plt.Circle(center, radius, color='b')
                    axes[1][0].imshow(frame)
                    axes[1][0].add_patch(patch)
                    plt.draw()

        ptses = np.asarray(plt.ginput(num_ref_points, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Facilitate similarity inspection between an image and a video.')
    parser.add_argument('--image_a', type=str, default="images/0.png", help='Path to the first image')
    parser.add_argument('--video_b', type=str, default="images/JAXA_tool.mp4", help='Path to the video.')
    parser.add_argument('--load_size', default=224, type=int, help='Load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""Stride of the first convolution layer. 
                                                                Small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""Type of model to extract. 
                              Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                              vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""Facet to create descriptors from. 
                                                                    Options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="Layer to create descriptors from.")
    parser.add_argument('--bin', default=False, type=bool, help="Create a binned descriptor if True.")
    parser.add_argument('--num_sim_patches', default=1, type=int, help="Number of closest patches to show.")
    parser.add_argument('--num_ref_points', default=1, type=int, help="Number of reference points to show.")

    args = parser.parse_args()

    with torch.no_grad():
        show_similarity_with_video(args.image_a, args.video_b, args.num_ref_points, args.load_size, args.layer, args.facet,
                                   args.bin, args.stride, args.model_type, args.num_sim_patches)
