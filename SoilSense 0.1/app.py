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
import glob
import random

from MainMicrobiome import AI_COUNT, CLASS_NAMES, load_ensemble, inference_transform, generate_cam

# --- INITIALIZATION ---
print("[*] Awakening the Council of Seven (Loading Models)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_ensemble(device)
print("[*] Ensemble ready.")

# --- QR CODE LOGIC ---
SPACE_URL = "https://huggingface.co/spaces/DragonfireCoder/microbiome-ai-classifier"

def generate_qr_code():
    qr = qrcode.QRCode(version=1, box_size=4, border=2)
    qr.add_data(SPACE_URL)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    qr_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width: 80px; height: 80px; border-radius: 8px; border: 1px solid rgba(0,212,255,0.3);">'

def load_random_image():
    """Robust image discovery for Hugging Face or local environments"""
    images = []
    # os.walk is the 'gold standard' for finding files on any server
    for root, dirs, files in os.walk("."):
        # Skip system and hidden folders to keep it lightning fast
        if any(x in root for x in ["venv", ".git", "__pycache__", ".gemini", "node_modules"]):
            continue
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                images.append(full_path)
    
    if images:
        choice = random.choice(images)
        print(f"[*] Randomly selected: {choice}")
        return choice
    
    print("[!] No images found in the entire project directory!")
    return None

# --- PREDICTION ENGINE ---
def predict(img):
    if img is None:
        return "No Data", {}, None, ""

    img = img.convert("RGB")
    tensor = inference_transform(img).unsqueeze(0).to(device, memory_format=torch.channels_last)

    votes = []
    with torch.inference_mode():
        outputs = []
        for i, model in enumerate(models):
            raw_out = model(tensor)
            prob = F.softmax(raw_out, dim=1)
            outputs.append(prob)
            
            vote_idx = torch.argmax(prob, dim=1).item()
            votes.append(f"AI {i+1}: {CLASS_NAMES[vote_idx]}")
        
        avg_probs = torch.mean(torch.stack(outputs), dim=0)[0]

    max_prob = torch.max(avg_probs).item()
    max_idx = torch.argmax(avg_probs).item()
    
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

    brain_html = "<div style='display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;'>"
    for i, v in enumerate(votes):
        color = "#00ff9d" if CLASS_NAMES[max_idx] in v else "#ff4b4b"
        brain_html += f"<div style='padding: 5px 10px; border: 1px solid {color}; border-radius: 5px; font-size: 10px; color: {color}; background: rgba(0,0,0,0.3);'>{v}</div>"
    brain_html += "</div>"

    return f"{CLASS_NAMES[max_idx]} ({max_prob*100:.1f}%)", confidences, cam_img, brain_html

# --- UI DESIGN ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');
body, .gradio-container {
    background: #05070A !important;
    font-family: 'Outfit', sans-serif !important;
}
.main-container {
    background: rgba(20, 26, 35, 0.8) !important;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 24px;
    padding: 20px;
    box-shadow: 0 0 50px rgba(0, 212, 255, 0.1);
}
h1 {
    background: linear-gradient(90deg, #00d4ff, #00ff9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 2.5rem !important;
}
.brain-box {
    margin-top: 15px;
    padding: 15px;
    border-radius: 12px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.05);
}
button.primary {
    background: linear-gradient(135deg, #00d4ff 0%, #00ff9d 100%) !important;
    border: none !important;
    color: #05070A !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}
"""

with gr.Blocks(css=custom_css, title="Microbiome 7-AI Ensemble") as demo:
    with gr.Column(elem_classes="main-container"):
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("# 🔬 Microbiome 7-AI Ensemble")
                gr.Markdown("### Advanced Soil Classification for Sustainable Agriculture")
            with gr.Column(scale=1):
                gr.HTML(f"<div style='float: right;'>{generate_qr_code()}</div>")

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Microscope Sample")
                with gr.Row():
                    random_btn = gr.Button("🎲 Random Sample")
                    run_btn = gr.Button("ANALYZE ENSEMBLE", variant="primary")
            
            with gr.Column(scale=1):
                with gr.Group():
                    final_label = gr.Label(num_top_classes=4, label="Consensus Confidence")
                    brain_status = gr.HTML("<div style='text-align:center; color: #666;'>Individual AI Brain Status Waiting...</div>", label="Brain Votes")
                
                cam_view = gr.Image(type="pil", label="Visual Reasoning (Heatmap)")

    random_btn.click(load_random_image, outputs=img_input)
    run_btn.click(predict, inputs=img_input, outputs=[gr.Textbox(visible=False), final_label, cam_view, brain_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)