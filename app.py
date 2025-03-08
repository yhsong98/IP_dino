import gradio as gr
import numpy as np
import os
from PIL import Image, ImageDraw
from app_fns import show_similarity_interactive  # Import the function

# Define local workspace
temp_workspace = "gradio"
os.makedirs(temp_workspace, exist_ok=True)

def save_original_image(img, filename):
    print(img)
    img = Image.open(img).convert("RGB")
    original_path = os.path.join(temp_workspace,filename)
    img.save(original_path)
    return original_path


def on_click(evt: gr.SelectData, original_img):
    # Reset to original image before marking new selection
    original_img = Image.open(original_img).convert("RGB")
    img = original_img.copy()
    draw = ImageDraw.Draw(img)
    draw.ellipse((evt.index[0] - 5, evt.index[1] - 5, evt.index[0] + 5, evt.index[1] + 5), fill="red", outline="red")

    return img, evt.index[0], evt.index[1]


def interactive_matching(image_a, image_b, x, y):
    print(f"Reference Image Path: {image_a}")
    print(f"Target Image Path: {image_b}")

    # Call show_similarity_interactive with the necessary parameters
    result_a, result_b = show_similarity_interactive(image_a, image_b)
    result_a.save(os.path.join(temp_workspace, "result_a.png"))
    result_b.save(os.path.join(temp_workspace, "result_b.png"))
    return os.path.join(temp_workspace, "result_a.png"), os.path.join(temp_workspace, "result_b.png")  # Update this if show_similarity_interactive saves an image


demo = gr.Blocks()
with demo:
    gr.Markdown("## Image Correspondence Finder")
    gr.Markdown("Upload two images and click on Image A to find the corresponding location in Image B.")

    with gr.Row():
        image_a = gr.Image(type="filepath")
        image_b = gr.Image(type="filepath")

    x_coord = gr.Number(label="X Coordinate", visible=False)
    y_coord = gr.Number(label="Y Coordinate", visible=False)

    original_image_a = gr.State()
    image_a.upload(lambda img: save_original_image(img, "original_image_a.png"), inputs=[image_a], outputs=[original_image_a])
    image_a.select(on_click, inputs=[original_image_a], outputs=[image_a, x_coord, y_coord])

    original_image_b = gr.State()
    image_b.upload(lambda img: save_original_image(img, "original_image_b.png"), inputs=[image_b], outputs=[original_image_b])

    btn = gr.Button("Find landmarks")
    with gr.Row():
        marked_a = gr.Image(type="filepath", interactive=True)
        marked_b = gr.Image(type="filepath")

    btn.click(interactive_matching, inputs=[original_image_a, original_image_b, x_coord, y_coord], outputs=[marked_a, marked_b])
    #marked_a.select(on_click, inputs=[marked_a], outputs=[marked_a, x_coord, y_coord])
demo.launch()
