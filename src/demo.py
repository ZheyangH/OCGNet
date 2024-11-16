import gradio as gr
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import logging
import os
import time


from model.DetGeo import DetGeo
from OCGNet import OCGNet
from data_loader import visualize_bbox
from utils.checkpoint import load_pretrain

BOX_COLOR_DETGEO = (255, 0, 0)  # Red
BOX_COLOR_OCGNet = (0, 255, 0)  # Green

DetGeo_MODEL_PATH = 'saved_models/DetGeo.pth.tar'
OCGNet_MODEL_PATH = 'saved_models/OCGNet.pth.tar'

anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
anchors_full = np.array([float(x.strip()) for x in anchors.split(',')])
anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

def load_pretrain(model, model_path, logging):
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        modified_pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in modified_pretrained_dict.items() if k in model_dict}
        if len(pretrained_dict) == 0:
            print("Error: No matching keys found between the pretrained model and the current model.")
            return model

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded pretrain model at {}".format(model_path))
        logging.info("=> loaded pretrain model at {}".format(model_path))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no pretrained file found at '{}'".format(model_path))
        logging.info("=> no pretrained file found at '{}'".format(model_path))
    return model

def load_model(model_path, model_class):
    model = model_class().cuda()
    model = load_pretrain(model, model_path, logging)
    model.eval()
    return model


def calculate_distance_matrix_detgeo(coords):
    points = coords[0]
    click_hw = (int(points[1]), int(points[0]))
    mat_clickhw = np.zeros((256, 256), dtype=np.float32)
    click_h = [pow(one - click_hw[0], 2) for one in range(256)]
    click_w = [pow(one - click_hw[1], 2) for one in range(256)]
    norm_hw = pow(256 * 256 + 256 * 256, 0.5)
    for i in range(256):
        for j in range(256):
            tmp_val = 1 - (pow(click_h[i] + click_w[j], 0.5) / norm_hw)
            mat_clickhw[i, j] = tmp_val * tmp_val
    return mat_clickhw


def calculate_distance_matrix_OCGNet(coords, sigma=0.1):
    points = coords[0]
    click_hw = (int(points[1]), int(points[0]))
    mat_clickhw = np.zeros((256, 256), dtype=np.float32)
    click_h = [pow(one - click_hw[0], 2) for one in range(256)]
    click_w = [pow(one - click_hw[1], 2) for one in range(256)]
    norm_hw = pow(256 * 256 + 256 * 256, 0.5)
    sigma = sigma * norm_hw
    for i in range(256):
        for j in range(256):
            distance = pow(click_h[i] + click_w[j], 0.5)
            mat_clickhw[i, j] = np.exp(-distance**2 / (2 * sigma**2)) 
    return mat_clickhw



def xywh2xyxy(x):
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

def get_box(pred_anchor, anchors_full=anchors_full, image_wh=1024):
    batch_size, grid_stride = pred_anchor.shape[0], image_wh // pred_anchor.shape[3]
    assert (len(pred_anchor.shape) == 5)
    assert (pred_anchor.shape[3] == pred_anchor.shape[4])

    pred_confidence = pred_anchor[:, :, 4, :, :]
    scaled_anchors = anchors_full / grid_stride

    pred_bbox = torch.zeros(batch_size, 4)
    for batch_idx in range(batch_size):
        best_n, gj, gi = torch.where(pred_confidence[batch_idx].max() == pred_confidence[batch_idx])
        best_n, gj, gi = best_n[0], gj[0], gi[0]
        pred_bbox[batch_idx, 0] = pred_anchor[batch_idx, best_n, 0, gj, gi].sigmoid() + gi
        pred_bbox[batch_idx, 1] = pred_anchor[batch_idx, best_n, 1, gj, gi].sigmoid() + gj
        pred_bbox[batch_idx, 2] = torch.exp(pred_anchor[batch_idx, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[batch_idx, 3] = torch.exp(pred_anchor[batch_idx, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
    pred_bbox = pred_bbox * grid_stride
    pred_bbox = xywh2xyxy(pred_bbox)

    return pred_bbox

def draw_click_point(image, point):
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.circle(img, point, 5, (0, 0, 255), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def handle_click(image, evt: gr.SelectData, click_coords):
    if isinstance(image, str):  # filepath case
        image = Image.open(image)
    click_coords.clear()
    click_coords.append((evt.index[0], evt.index[1]))
    return draw_click_point(image, click_coords[0]), click_coords

def visualize_attention(image, attn_score):
    attn_score = cv2.resize(attn_score, (image.shape[1], image.shape[0]))
    attn_heatmap = cv2.applyColorMap(np.uint8(255 * attn_score), cv2.COLORMAP_JET)
    attn_overlay = cv2.addWeighted(image, 0.6, attn_heatmap, 0.4, 0)
    return attn_overlay


def demo(query_img, satellite_img, click_coords, enable_detgeo):
    distance_matrix_detgeo = calculate_distance_matrix_detgeo(click_coords)
    distance_matrix_detgeo = torch.from_numpy(distance_matrix_detgeo).unsqueeze(0).cuda()

    distance_matrix_OCGNet = calculate_distance_matrix_OCGNet(click_coords)
    distance_matrix_OCGNet = torch.from_numpy(distance_matrix_OCGNet).unsqueeze(0).cuda()

    DetGeo_model = load_model(DetGeo_MODEL_PATH, DetGeo)
    OCGNet_model = load_model(OCGNet_MODEL_PATH, OCGNet)
    print("Success for load models!!")

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    queryimg = cv2.imread(query_img)
    queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)

    rsimg = cv2.imread(satellite_img)
    rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)
    satellite_img = rsimg

    if not click_coords:
        return (cv2.cvtColor(np.array(satellite_img), cv2.COLOR_RGB2BGR), 
                cv2.cvtColor(np.array(satellite_img), cv2.COLOR_RGB2BGR), 
                cv2.cvtColor(np.array(satellite_img), cv2.COLOR_RGB2BGR))

    rsimg = input_transform(rsimg.copy())
    queryimg = input_transform(queryimg.copy())
    query_imgs, rs_imgs = queryimg.cuda(), rsimg.cuda()

    query_imgs = query_imgs.unsqueeze(0)
    rs_imgs = rs_imgs.unsqueeze(0)

    with torch.no_grad():
        start_OCGNet = time.time()
        pred_anchor_OCGNet, attn_score_OCGNet = OCGNet_model(query_imgs, rs_imgs, distance_matrix_OCGNet)
        end_OCGNet = time.time()
        print("OCGNet time:", end_OCGNet-start_OCGNet)
        pred_anchor_OCGNet = pred_anchor_OCGNet.view(pred_anchor_OCGNet.shape[0], 9, 5, pred_anchor_OCGNet.shape[2], pred_anchor_OCGNet.shape[3])
        bbox_OCGNet = get_box(pred_anchor_OCGNet)[0]
        bbox_OCGNet = bbox_OCGNet.squeeze().numpy().astype(int)
        attn_score_OCGNet = attn_score_OCGNet.squeeze().cpu().numpy()

        if enable_detgeo:
            start_detgeo = time.time()
            pred_anchor_detgeo, attn_score_detgeo = DetGeo_model(query_imgs, rs_imgs, distance_matrix_detgeo)
            end_detgeo = time.time()
            print("degeo time:", end_detgeo-start_detgeo)

            pred_anchor_detgeo = pred_anchor_detgeo.view(pred_anchor_detgeo.shape[0], 9, 5, pred_anchor_detgeo.shape[2], pred_anchor_detgeo.shape[3])
            bbox_detgeo = get_box(pred_anchor_detgeo)[0]
            bbox_detgeo = bbox_detgeo.squeeze().numpy().astype(int)
            attn_score_detgeo = attn_score_detgeo.squeeze().cpu().numpy()
        else:
            bbox_detgeo = None  
            attn_score_detgeo = None

    satellite_img_bbox = cv2.cvtColor(np.array(satellite_img), cv2.COLOR_RGB2BGR)
    if enable_detgeo and bbox_detgeo is not None:
        visualize_bbox(satellite_img_bbox, bbox_detgeo, color=BOX_COLOR_DETGEO)

    visualize_bbox(satellite_img_bbox, bbox_OCGNet, color=BOX_COLOR_OCGNet)

    attn_vis_detgeo = visualize_attention(satellite_img, attn_score_detgeo) if attn_score_detgeo is not None else satellite_img
    attn_vis_OCGNet = visualize_attention(satellite_img, attn_score_OCGNet)

    return (Image.fromarray(cv2.cvtColor(satellite_img_bbox, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(attn_vis_detgeo, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(attn_vis_OCGNet, cv2.COLOR_BGR2RGB)))


# Create Gradio interface
click_coords = []

with gr.Blocks() as demo_app:
    gr.Markdown("# OCGNet Demo")
    click_coords_state = gr.State(click_coords)

    with gr.Row():
        input_query_img = gr.Image(label="UAV Image", interactive=True, type="filepath")
        input_satellite_img = gr.Image(label="Satellite Image", type="filepath")
        click_visualization = gr.Image(label="Click Visualization")

    # Add a checkbox to control whether to output DetGeo results
    enable_detgeo = gr.Checkbox(label="Enable DetGeo Output", value=True)

    output_result_image_bbox = gr.Image(label="Bbox Result Image")
    output_result_image_detgeo = gr.Image(label="DetGeo Prediction Image")  # DetGeo predicted bounding box output image
    output_result_image_OCGNet = gr.Image(label="OCGNet Attention Image")

    # Handle click event to update click coordinates
    input_query_img.select(handle_click, inputs=[input_query_img, click_coords_state], outputs=[click_visualization, click_coords_state])

    demo_button = gr.Button("Detect the object")

    def process_images(query_img, satellite_img, click_coords, enable_detgeo):
        # Pass enable_detgeo parameter when calling the demo function
        result_bbox, result_detgeo, result_OCGNet = demo(query_img, satellite_img, click_coords, enable_detgeo)
        return result_bbox, result_detgeo, result_OCGNet, click_coords

    demo_button.click(
        process_images, 
        inputs=[input_query_img, input_satellite_img, click_coords_state, enable_detgeo], 
        outputs=[output_result_image_bbox, output_result_image_detgeo, output_result_image_OCGNet, click_coords_state]
    )

# Launch the demo interface and share it
demo_app.launch(share=True)
