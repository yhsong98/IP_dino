import gradio as gr
import os
from PIL import Image, ImageDraw
from app_fns import show_similarity_interactive, find_correspondence  # Import the function
from extractor_app import ViTExtractor

# Define local workspace
temp_workspace = "gradio"
os.makedirs(temp_workspace, exist_ok=True)

stride=14
model_type='dinov2_vits14'
extractor = ViTExtractor(stride=stride, model_type=model_type, device= 'cuda')

def save_original_image(img):
    max_size = 400
    width, height = img.size
    scale = min(max_size / max(width, height), 1.0)  # Scale factor (only shrink, no enlargement)
    new_size = (int(width * scale), int(height * scale))
    img=img.resize(new_size)
    copy=img.copy()
    return img, copy


def on_click(original_img, evt: gr.SelectData ):
    # Reset to original image before marking new selection
    print('clicked')
    img = original_img.copy()
    draw = ImageDraw.Draw(img)
    draw.ellipse((evt.index[0] - 5, evt.index[1] - 5, evt.index[0] + 5, evt.index[1] + 5), fill="red", outline="red")

    return img, evt.index[0], evt.index[1]

def on_click_(img_path, evt: gr.SelectData):
    # Load the original image from Gradio workspace
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((evt.index[0] - 5, evt.index[1] - 5, evt.index[0] + 5, evt.index[1] + 5), fill="red", outline="red")
    return img, evt.index[0], evt.index[1]

def init_extractor(stride: int = 14, model_type: str = 'dinov2_vits14',device: str = 'cuda') -> ViTExtractor:
    extractor = ViTExtractor(model_type, stride, device=device)
    return extractor

def interactive_matching(image_a, image_b, similarity_threshold=0.95, num_ref_points=200):
    print(f"Reference Image Path: {image_a}")
    print(f"Target Image Path: {image_b}")

    # Call show_similarity_interactive with the necessary parameters
    result_a, result_b,  patch_size, stride, load_size_a, image_b_size, num_patches_a, num_patches_b, output_reference, output_rotated_coords, similarities, rotation = show_similarity_interactive(image_a, image_b, extractor, sim_threshold=similarity_threshold,num_ref_points=num_ref_points)
    # result_a.save(os.path.join(temp_workspace, "result_a.png"))
    # result_b.save(os.path.join(temp_workspace, "result_b.png"))
    #return os.path.join(temp_workspace, "result_a.png"), os.path.join(temp_workspace, "result_b.png")  # Update this if show_similarity_interactive saves an image
    return result_a, result_b ,result_a, result_b, patch_size, stride, load_size_a, image_b_size, num_patches_a, num_patches_b, output_reference, output_rotated_coords, similarities, rotation

def mark_correspondence(image_a, evt: gr.SelectData, image_b, original_image_a, original_image_b, patch_size, stride, load_size_a,
                        image_b_size, num_patches_a, num_patches_b, landmarks_a, landmarks_b, similarities, rotation, similarity_threshold,distance_threshold):
    marked_a, marked_b, marked_orig_a, marked_orig_b = find_correspondence(image_a, image_b, original_image_a,original_image_b, [evt.index], patch_size, stride, load_size_a,
                        image_b_size, num_patches_a, num_patches_b, landmarks_a, landmarks_b, similarities,rotation, num_candidates=10, sim_threshold=similarity_threshold,distance_threshold=distance_threshold)
    return marked_a, marked_b, marked_orig_a, marked_orig_b



demo = gr.Blocks()
with demo:
    gr.Markdown("## Image Correspondence Finder")
    gr.Markdown("### Instruction")
    gr.Markdown("""
    1. Upload two images, left one as reference and right as target;
    2. After images are uploaded, click 'Find landmarks' button;
    3. After landmarks are extracted, click on either of the images on the left to find the corresponding point on right image.
    """)
    gr.Markdown("### Adjustable parameters")
    gr.Markdown("""
    1. Similarity Threshold: the confidence of identifying correponding points;
    2. Distance Threshold: the distance shreshold between corresponding point found by ViT and the point location estimated by landmarks
    3. Number of Reference Points: how many random points on the reference image wil be classified to be a landmark or not.
    """)

    if extractor:
        gr.Markdown(model_type+' has been initialized with stride '+str(stride)+'.')

    gr.Markdown("<span style='color:red; font-size:18px;'>Red:selected via ViT;</span>"+
                "     <span style='color:green; font-size:18px;'>Green:selected via ViT, coordinates fixed by landmarks;</span>"+
               "     <span style='color:blue; font-size:18px;'>Blue:no confident result from ViT, selected purely by landmarks.</span>")


    with gr.Row():
        image_a = gr.Image(type='pil',interactive=True)
        image_b = gr.Image(type='pil')

    original_image_a = gr.Image(type='pil', visible=False)
    image_a.upload(save_original_image, inputs=[image_a], outputs=[image_a, original_image_a])
    #image_a.select(on_click, inputs=[original_image_a], outputs=[image_a, x_coord, y_coord])


    original_image_b = gr.Image(type='pil', visible=False)
    image_b.upload(save_original_image, inputs=[image_b], outputs=[image_b, original_image_b])

    with gr.Row():
        similarity_threshold = gr.Slider(minimum=0, maximum=1, step=0.001, value=0.95, label="Similarity Threshold", interactive=True)
        distance_threshold = gr.Slider(minimum=0, maximum=50, step=5, value=10, label="Distance Threshold",
                                         interactive=True)
        num_ref_points = gr.Slider(minimum=0, maximum=1500, step=100, value=200, label="Number of Reference Points",
                                         interactive=True)
        btn = gr.Button("Find landmarks")
    with gr.Row():
        marked_a = gr.Image(type="pil")
        marked_b = gr.Image(type="pil")

    marked_a_origin = gr.Image(type="pil", visible=False,interactive=True)
    marked_b_origin = gr.Image(type="pil", visible=False)

    patch_size, stride, load_size_a, image_b_size, num_patches_a, num_patches_b, output_reference, output_rotated_coords, similarities, rotation  = gr.State(),gr.State(),gr.State(),gr.State(),gr.State(),gr.State(),gr.State(),gr.State(),gr.State(),gr.State()

    btn.click(interactive_matching, inputs=[original_image_a, original_image_b, similarity_threshold, num_ref_points], outputs=[marked_a, marked_b, marked_a_origin, marked_b_origin, patch_size, stride, load_size_a, image_b_size, num_patches_a, num_patches_b, output_reference, output_rotated_coords, similarities, rotation])

    marked_a.select(mark_correspondence,
                    inputs=[marked_a_origin, marked_b_origin, original_image_a, original_image_b,patch_size, stride, load_size_a,
                                                 image_b_size, num_patches_a, num_patches_b, output_reference,
                            output_rotated_coords, similarities, rotation, similarity_threshold, distance_threshold],
                    outputs=[marked_a, marked_b, image_a, image_b])
    image_a.select(mark_correspondence,
                   inputs=[marked_a_origin, marked_b_origin, original_image_a, original_image_b, patch_size, stride,
                           load_size_a, image_b_size, num_patches_a, num_patches_b, output_reference,
                           output_rotated_coords, similarities, rotation, similarity_threshold, distance_threshold],
                   outputs=[marked_a, marked_b,image_a, image_b])


demo.launch(share=True)#)
