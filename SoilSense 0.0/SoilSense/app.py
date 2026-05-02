import os
import sys
import subprocess

try:
    import gradio as gr
except ImportError:
    print("[*] Automatically installing missing module: gradio")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import qrcode
from io import BytesIO
import base64

from MainMicrobiome import AI_COUNT, CLASS_NAMES, load_ensemble, inference_transform, generate_cam

print("Loading Ensemble Models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_ensemble(device)
print("Models loaded successfully.")

SPACE_URL = "https://huggingface.co/spaces/DragonfireCoder/microbiome-ai-classifier"

def generate_qr_code():
    """Generate QR code as base64 image for embedding in HTML"""
    qr = qrcode.QRCode(version=1, box_size=4, border=2)
    qr.add_data(SPACE_URL)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    buffered = BytesIO()
    qr_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width: 100px; height: 100px; display: inline-block;">'

def predict(img):
    if img is None:
        return "No image provided", {}, None

    img = img.convert("RGB")
    tensor = inference_transform(img).unsqueeze(0).to(device, memory_format=torch.channels_last)

    with torch.inference_mode():
        outputs = [F.softmax(model(tensor), dim=1) for model in models]
        avg_probs = torch.mean(torch.stack(outputs), dim=0)[0]

    max_prob = torch.max(avg_probs).item()
    max_idx = torch.argmax(avg_probs).item()
    
    prediction_text = "UNSURE ⚠️" if max_prob < 0.5 else f"{CLASS_NAMES[max_idx]} ✅"

    confidences = {CLASS_NAMES[i]: float(avg_probs[i]) for i in range(len(CLASS_NAMES))}

    cam_img = None
    cam, _ = generate_cam(models[0], tensor, max_idx)
    if cam is not None:
        cam_resized = cv2.resize(cam, (img.size[0], img.size[1]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        img_np = np.array(img)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        cam_img = Image.fromarray(overlay)

    return prediction_text, confidences, cam_img

qr_html = generate_qr_code()

theme = gr.themes.Monochrome(
    primary_hue="slate",
    secondary_hue="zinc",
    neutral_hue="zinc",
    radius_size=gr.themes.sizes.radius_lg,
)

with gr.Blocks(theme=theme, title="Microbiome AI") as demo:
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # 🔬 Microbiome AI Classification Dashboard
                Upload a microscope image to let the **Ensemble of 5 AIs** classify it. 
                It will identify whether it is DIRT, GRASS, LEAF, or a MIX, and show you exactly what it was looking at using a Class Activation Map (CAM).
                """
            )
        with gr.Column(scale=1):
            gr.HTML(f'<div style="text-align: right; padding-top: 10px;"><p style="font-size: 12px; margin-bottom: 5px;">Scan to share</p>{qr_html}</div>')
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Microbiome Image")
            submit_btn = gr.Button("Analyze Sample", variant="primary")
            
        with gr.Column():
            prediction_output = gr.Textbox(label="Final Consensus")
            label_output = gr.Label(label="AI Confidence Scores", num_top_classes=4)
            cam_output = gr.Image(label="AI Focus Map (CAM)", type="pil")

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[prediction_output, label_output, cam_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)[9]