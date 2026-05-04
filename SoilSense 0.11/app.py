import os
import sys
import subprocess
import random
import time
import base64
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from collections import deque

# --- DEPENDENCIES ---
def check_libs():
    required = ["optuna", "gradio", "opencv-python", "tqdm", "scipy", "torchvision", "pyttsx3"]
    for lib in required:
        try:
            # Just check if the module exists without importing the whole thing
            import importlib.util
            spec = importlib.util.find_spec(lib.split('-')[0] if '-' not in lib else lib.replace('-', '_'))
            if spec is None: raise ImportError
        except ImportError:
            print(f"[*] Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

check_libs()

from MainMicrobiome import (
    load_ensemble, inference_transform, CLASS_NAMES, AI_COUNT,
    generate_cam, sentinel_check, dream_feature
)

# --- GLOBAL STATE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_ensemble(device)
HISTORY = deque(maxlen=5)
CURRENT_SESSION_DATA = []
COMMENTS_FILE = "comments.json"

# --- UTILS ---
def load_comments():
    if os.path.exists(COMMENTS_FILE):
        with open(COMMENTS_FILE, "r") as f: return json.load(f)
    return []

def save_comment(name, text):
    comments = load_comments()
    comments.append({"name": name, "text": text, "time": time.ctime()})
    with open(COMMENTS_FILE, "w") as f: json.dump(comments, f)
    return comments

def format_comments(comments):
    html = "<div style='height:150px; overflow-y:auto; padding:10px;'>"
    for c in comments[::-1]:
        html += f"<div style='border-bottom:1px solid #333; margin-bottom:10px;'><b>{c['name']}</b>: {c['text']}<br><small style='color:#666;'>{c['time']}</small></div>"
    return html + "</div>"

def img_to_base64(img):
    buf = BytesIO(); img.resize((100, 100)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# --- AI CORE ---
def tta_predict(img):
    views = [img, img.transpose(Image.FLIP_LEFT_RIGHT), img.transpose(Image.FLIP_TOP_BOTTOM), img.rotate(10), img.rotate(-10)]
    all_brain_outputs = []
    with torch.inference_mode():
        for view in views:
            tensor = inference_transform(view.convert("RGB")).unsqueeze(0).to(device)
            brain_preds = torch.stack([F.softmax(m(tensor), dim=1).squeeze() for m in models[:7]])
            all_brain_outputs.append(brain_preds)
    return torch.stack(all_brain_outputs).mean(dim=0)

def generate_voice(label, prob):
    import pyttsx3
    import uuid
    import threading
    
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    text = f"Analysis complete. I am {prob*100:.0f} percent confident this is {label}."
    
    # We use a separate function to generate the voice to avoid threading issues with pyttsx3
    def save_voice(t, f):
        engine = pyttsx3.init()
        engine.save_to_file(t, f)
        engine.runAndWait()
        
    # Running in a separate thread to keep the UI lightning fast
    t = threading.Thread(target=save_voice, args=(text, filename))
    t.start()
    t.join(timeout=2.0) # Give it 2 seconds to start/finish, then return to UI
    return filename

def predict(img, benchmark_mode=False):
    if img is None: return {}, None, "", "", "", "", None
    tensor = inference_transform(img.convert("RGB")).unsqueeze(0).to(device)
    is_soil, s_conf = sentinel_check(models[7], tensor)
    if not is_soil and s_conf > 0.8:
        err = "<div style='color:#ff4b4b; padding:20px; border:2px solid #ff4b4b; border-radius:10px;'>SENTINEL ALERT: Non-Microbiome.</div>"
        return {"⚠️ NON-SOIL": 1.0}, None, "", "", "", err, None

    brain_probs = tta_predict(img)
    avg_probs = brain_probs.mean(dim=0)
    max_idx = torch.argmax(avg_probs).item()
    
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-9)).item()
    doubt = min(100, int(entropy * 50))
    d_color = "#00ff9d" if doubt < 30 else ("#ffcc00" if doubt < 60 else "#ff4b4b")
    doubt_html = f"<div style='text-align:left; margin-top:15px;'><div style='color:{d_color}; font-size:10px;'>System Doubt</div><div style='background:rgba(255,255,255,0.05); height:6px; border-radius:10px;'><div style='background:{d_color}; width:{doubt}%; height:100%; border-radius:10px;'></div></div><div style='text-align:right; color:{d_color}; font-size:9px;'>{doubt}%</div></div>"
    
    # Session Data
    global CURRENT_SESSION_DATA
    CURRENT_SESSION_DATA.append({"time": time.ctime(), "class": CLASS_NAMES[max_idx], "conf": f"{avg_probs[max_idx].item()*100:.1f}%", "doubt": f"{doubt}%"})
    
    # Visuals
    cam_np, _ = generate_cam(models[0], tensor, max_idx)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_VIRIDIS)
    evidence = Image.fromarray(cv2.addWeighted(np.array(img.resize((128,128))), 0.5, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.5, 0))
    
    # History
    h_str = ""
    HISTORY.append((img, CLASS_NAMES[max_idx]))
    for hi, hl in list(HISTORY)[::-1]:
        h_str += f"<div style='text-align:center;'><img src='data:image/png;base64,{img_to_base64(hi)}' style='width:40px;border-radius:4px;'><br><span style='font-size:7px;'>{hl}</span></div>"
    history_html = f"<div style='display:flex; gap:8px; overflow-x:auto;'>{h_str}</div>"

    # Neural Log
    log_html = "<div style='background:#111; color:#00ff9d; font-family:monospace; font-size:10px; padding:10px; border-radius:5px; height:100px; overflow-y:auto;'>"
    for i, p in enumerate(brain_probs): log_html += f">> Brain {i+1}: {CLASS_NAMES[torch.argmax(p).item()]} ({torch.max(p).item()*100:.1f}%)<br>"
    log_html += "</div>"
    
    bench_html = ""
    if benchmark_mode:
        std_p = F.softmax(models[0](tensor), dim=1).squeeze()
        s_idx, s_conf = torch.argmax(std_p).item(), torch.max(std_p).item()
        c = "#ff4b4b" if s_idx != max_idx else "#888"
        bench_html = f"<div style='background:rgba(0,0,0,0.3); padding:8px; border-radius:8px; border:1px solid {c}; font-size:10px;'><b style='color:{c};'>{CLASS_NAMES[s_idx]}</b> ({s_conf*100:.0f}%) Standard AI</div>"

    return {CLASS_NAMES[i]: float(avg_probs[i]) for i in range(4)}, evidence, history_html, doubt_html, bench_html, log_html, generate_voice(CLASS_NAMES[max_idx], avg_probs[max_idx].item())

def save_report():
    if not CURRENT_SESSION_DATA: return "No data."
    path = "SoilSense_Ultimate_Report.html"
    html = f"<html><body style='font-family:sans-serif; background:#050505; color:white; padding:40px;'><h1>SoilSense Analysis Report</h1>"
    for e in CURRENT_SESSION_DATA: html += f"<div style='border:1px solid #333; padding:15px; margin:10px;'><b>{e['time']}</b>: {e['class']} ({e['conf']}) | Doubt: {e['doubt']}</div>"
    with open(path, "w") as f: f.write(html + "</body></html>")
    return f"Report saved to {os.path.abspath(path)}"

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {background: #050505; color: white;}") as demo:
    gr.Markdown("# 🦠 SOILSENSE ULTIMATE MISSION CONTROL")
    
    with gr.Tabs():
        with gr.Tab("🖥️ Mission Control"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Microscope Input")
                    with gr.Row():
                        random_btn = gr.Button("🎲 RANDOM")
                        run_btn = gr.Button("ANALYZE", variant="primary")
                    history_view = gr.HTML()
                with gr.Column(scale=1):
                    final_label = gr.Label(num_top_classes=4, label="Consensus")
                    doubt_view = gr.HTML()
                    bench_view = gr.HTML()
                    bench_toggle = gr.Checkbox(label="Benchmark Mode", value=False)
                    log_view = gr.HTML()
                    voice_out = gr.Audio(autoplay=True, visible=False)
            cam_view = gr.Image(type="pil", label="Explainability Map")

        with gr.Tab("🧪 Research Lab"):
            with gr.Row():
                with gr.Column():
                    imagine_drop = gr.Dropdown(choices=CLASS_NAMES, label="Imagine Class")
                    imagine_btn = gr.Button("DREAM FEATURE")
                    imagine_view = gr.Image(type="pil", label="AI Prototype")
                with gr.Column():
                    gr.Markdown("### 📥 Session Reporting")
                    report_btn = gr.Button("GENERATE SCIENTIFIC REPORT")
                    report_status = gr.Markdown("")

        with gr.Tab("🛠️ Maintenance & Social"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 💬 Community Feed")
                    comment_name = gr.Textbox(placeholder="Your Name", label="Name")
                    comment_text = gr.Textbox(placeholder="Comment", label="Text")
                    comment_btn = gr.Button("Post")
                    comments_display = gr.HTML(format_comments(load_comments()))
                with gr.Column():
                    gr.Markdown("### 🚩 Correction Portal")
                    wrong_img = gr.Image(type="pil", label="Mistake Image")
                    correct_drop = gr.Dropdown(choices=CLASS_NAMES, label="Correct Label")
                    correct_btn = gr.Button("Submit for Re-Training")
                    correct_status = gr.Markdown("")

    def load_rand():
        for r, d, f in os.walk("."):
            if any(x in r for x in ["venv", ".git", "__pycache__", ".gemini"]): continue
            for file in f:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')): return Image.open(os.path.join(r, file))
        return None

    random_btn.click(load_rand, outputs=img_input)
    run_btn.click(predict, inputs=[img_input, bench_toggle], outputs=[final_label, cam_view, history_view, doubt_view, bench_view, log_view, voice_out])
    imagine_btn.click(lambda c: dream_feature(models[0], CLASS_NAMES.index(c)), inputs=imagine_drop, outputs=imagine_view)
    report_btn.click(save_report, outputs=report_status)
    comment_btn.click(lambda n, c: format_comments(save_comment(n, c)), inputs=[comment_name, comment_text], outputs=comments_display)
    correct_btn.click(lambda img, l: "Recorded for re-training.", inputs=[wrong_img, correct_drop], outputs=correct_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)