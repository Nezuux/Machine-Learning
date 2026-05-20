# -*- coding: utf-8 -*-
import streamlit as st
import torch
import numpy as np
import os
import time
import io
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from datetime import datetime
import matplotlib.cm as cm

st.set_page_config(
    page_title='SDIS 44 - Vigie IA Feux de For\u00eat',
    page_icon='\U0001f6a8',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --- Session state for history ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- PREMIUM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
        --border-glass: rgba(255, 255, 255, 0.08);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-red: #ef4444;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--bg-primary) !important;
        color: var(--text-primary);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
        border-right: 1px solid var(--border-glass);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] p {
        color: var(--text-secondary) !important;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #7f1d1d 100%);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.05) 0%, transparent 50%);
        animation: rotate 20s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .hero-content {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }

    .hero-icon {
        width: 70px;
        height: 70px;
        background: linear-gradient(135deg, var(--accent-red), #dc2626);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
        animation: glow 3s ease-in-out infinite;
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 8px 48px rgba(239, 68, 68, 0.5); }
    }

    .hero-text h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 800;
        color: white !important;
        letter-spacing: -0.02em;
    }

    .hero-text p {
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    .hero-badge {
        position: absolute;
        top: 1.5rem;
        right: 2rem;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        animation: statusPulse 2s ease-in-out infinite;
    }

    @keyframes statusPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Glass cards */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
        transform: translateY(-2px);
    }

    /* Alert panels */
    .alert-fire-premium {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.8), rgba(153, 27, 27, 0.6));
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: fireAlert 1.5s ease-in-out infinite;
    }

    .alert-fire-premium::before {
        content: '';
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 10px,
            rgba(239, 68, 68, 0.03) 10px,
            rgba(239, 68, 68, 0.03) 20px
        );
    }

    @keyframes fireAlert {
        0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.2), inset 0 0 20px rgba(239, 68, 68, 0.05); }
        50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.4), inset 0 0 30px rgba(239, 68, 68, 0.1); }
    }

    .alert-fire-premium h2 {
        position: relative;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 800;
        color: #fca5a5 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .alert-fire-premium p {
        position: relative;
        margin: 0.5rem 0 0 0;
        color: #fecaca;
        font-size: 0.9rem;
    }

    .alert-safe-premium {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.6), rgba(4, 120, 87, 0.4));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }

    .alert-safe-premium h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 800;
        color: #6ee7b7 !important;
        letter-spacing: 0.02em;
    }

    .alert-safe-premium p {
        margin: 0.5rem 0 0 0;
        color: #a7f3d0;
        font-size: 0.9rem;
    }

    /* Circular gauge */
    .gauge-container {
        position: relative;
        width: 130px;
        height: 130px;
        margin: 0 auto;
    }

    .gauge-svg {
        transform: rotate(-90deg);
        width: 130px;
        height: 130px;
    }

    .gauge-bg {
        fill: none;
        stroke: rgba(255,255,255,0.06);
        stroke-width: 8;
    }

    .gauge-fill {
        fill: none;
        stroke-width: 8;
        stroke-linecap: round;
        transition: stroke-dashoffset 1.5s ease;
    }

    .gauge-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
    }

    .gauge-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .gauge-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* Metric cards */
    .metric-gauge {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-gauge:hover {
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }

    .metric-gauge .label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    .metric-gauge .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-gauge .value.fire {
        background: linear-gradient(135deg, #f87171, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-gauge .value.safe {
        background: linear-gradient(135deg, #34d399, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Confidence bar */
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(255,255,255,0.1);
        margin-top: 0.5rem;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 1s ease;
    }

    .confidence-bar-fill.fire {
        background: linear-gradient(90deg, #f87171, #ef4444);
    }

    .confidence-bar-fill.safe {
        background: linear-gradient(90deg, #34d399, #10b981);
    }

    /* Risk level badge */
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 0.8rem;
    }

    .risk-critical {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
        animation: riskPulse 1s ease-in-out infinite;
    }

    .risk-high {
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid rgba(245, 158, 11, 0.5);
        color: #fcd34d;
    }

    .risk-moderate {
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.5);
        color: #93c5fd;
    }

    .risk-low {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #6ee7b7;
    }

    @keyframes riskPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Image container */
    .image-container {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1rem;
        overflow: hidden;
    }

    .image-container h4 {
        margin: 0 0 0.8rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .image-container img {
        border-radius: 8px;
    }

    /* Feature cards */
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }

    .feature-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }

    .feature-icon {
        width: 50px;
        height: 50px;
        margin: 0 auto 1rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }

    .feature-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .feature-card p {
        margin: 0;
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* Dropzone */
    .dropzone {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(99, 102, 241, 0.05));
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .dropzone:hover {
        border-color: rgba(99, 102, 241, 0.6);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(99, 102, 241, 0.08));
    }

    .dropzone-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    /* Disclaimer */
    .disclaimer-premium {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.5rem;
        font-size: 0.8rem;
        color: #fbbf24;
    }

    .disclaimer-premium strong {
        color: #f59e0b;
    }

    /* Footer */
    .footer-premium {
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border-glass);
        color: var(--text-secondary);
        font-size: 0.75rem;
    }

    /* History log */
    .history-item {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
        margin-bottom: 0.5rem;
        font-size: 0.78rem;
    }

    .history-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .history-dot.fire { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.5); }
    .history-dot.safe { background: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.5); }

    .history-time {
        font-family: 'JetBrains Mono', monospace;
        color: rgba(255,255,255,0.4);
        font-size: 0.7rem;
    }

    .history-result {
        color: rgba(255,255,255,0.8);
        flex: 1;
    }

    .history-conf {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.75rem;
    }

    /* Analysis time */
    .analysis-time {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #93c5fd;
    }

    /* Fire particles */
    .fire-particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    .particle {
        position: absolute;
        bottom: -10px;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        animation: rise linear infinite;
        opacity: 0;
    }

    @keyframes rise {
        0% { transform: translateY(0) scale(1); opacity: 0; }
        10% { opacity: 0.8; }
        90% { opacity: 0.3; }
        100% { transform: translateY(-100vh) scale(0.3); opacity: 0; }
    }

    /* Sidebar section */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
    }

    .sidebar-section h4 {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(255,255,255,0.4) !important;
        margin: 0 0 0.8rem 0;
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.82rem;
    }

    .stat-row:last-child { border-bottom: none; }
    .stat-label { color: rgba(255,255,255,0.5); }
    .stat-value { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: rgba(255,255,255,0.9); }

    /* Timestamp */
    .timestamp {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
        background: rgba(255,255,255,0.03);
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        border: 1px solid var(--border-glass);
    }

    /* Interpretation text */
    .interpretation {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border-glass);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.82rem;
        line-height: 1.6;
        color: var(--text-secondary);
    }

    .interpretation strong {
        color: var(--text-primary);
    }
</style>
""", unsafe_allow_html=True)

# --- GradCAM implementation ---
def compute_gradcam(model, input_tensor, target_layer):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_idx = output.argmax(dim=-1).item()
    model.zero_grad()
    output[0, pred_idx].backward()

    handle_f.remove()
    handle_b.remove()

    act = activations[0][0]
    grad = gradients[0][0]

    weights = grad.mean(dim=(1, 2))
    cam_map = (weights[:, None, None] * act).sum(dim=0)
    cam_map = torch.relu(cam_map)
    cam_map = cam_map - cam_map.min()
    if cam_map.max() > 0:
        cam_map = cam_map / cam_map.max()

    cam_map = cam_map.unsqueeze(0).unsqueeze(0)
    cam_map = torch.nn.functional.interpolate(cam_map, size=(224, 224), mode='bilinear', align_corners=False)
    return cam_map[0, 0].numpy()


def overlay_cam_on_image(img_array, cam_map, alpha=0.5):
    colormap = cm.get_cmap('inferno')
    heatmap = colormap(cam_map)[:, :, :3]
    overlay = (1 - alpha) * img_array + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def get_risk_level(pred_class, confidence):
    """Return risk level, class, and interpretation."""
    if pred_class == 'fire':
        if confidence >= 0.9:
            return 'CRITIQUE', 'risk-critical', 'Forte probabilit\u00e9 de d\u00e9part de feu. V\u00e9rification imm\u00e9diate requise. Le mod\u00e8le est tr\u00e8s confiant dans sa d\u00e9tection.'
        elif confidence >= 0.7:
            return '\u00c9LEV\u00c9', 'risk-high', 'Suspicion de feu avec confiance mod\u00e9r\u00e9e. Confirmation visuelle recommand\u00e9e avant d\u00e9clenchement du protocole.'
        else:
            return 'MOD\u00c9R\u00c9', 'risk-moderate', 'Signal faible d\u00e9tect\u00e9. Possibilit\u00e9 de faux positif (fum\u00e9e, reflets, coucher de soleil). V\u00e9rification conseill\u00e9e.'
    else:
        if confidence >= 0.9:
            return 'FAIBLE', 'risk-low', 'Zone s\u00e9curis\u00e9e. Le mod\u00e8le n\'identifie aucun signe de d\u00e9part de feu avec une tr\u00e8s haute confiance.'
        else:
            return 'MOD\u00c9R\u00c9', 'risk-moderate', 'Pas de feu d\u00e9tect\u00e9 mais confiance limit\u00e9e. Conditions m\u00e9t\u00e9o ou qualit\u00e9 d\'image possiblement d\u00e9grad\u00e9es.'


def svg_gauge(value, color_start, color_end, label):
    """Generate SVG circular gauge."""
    circumference = 2 * 3.14159 * 52
    offset = circumference * (1 - value)
    return f"""
    <div class="gauge-container">
        <svg class="gauge-svg" viewBox="0 0 120 120">
            <circle class="gauge-bg" cx="60" cy="60" r="52"/>
            <circle class="gauge-fill" cx="60" cy="60" r="52"
                stroke="url(#grad-{label})"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"/>
            <defs>
                <linearGradient id="grad-{label}" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:{color_start}"/>
                    <stop offset="100%" style="stop-color:{color_end}"/>
                </linearGradient>
            </defs>
        </svg>
        <div class="gauge-text">
            <div class="gauge-value">{value:.0%}</div>
            <div class="gauge-label">{label}</div>
        </div>
    </div>
    """


def generate_particles_html():
    """Generate fire particles CSS animation."""
    particles = ""
    import random
    for i in range(20):
        left = random.randint(0, 100)
        delay = random.uniform(0, 5)
        duration = random.uniform(4, 8)
        size = random.uniform(2, 5)
        color = random.choice(['#ef4444', '#f97316', '#fbbf24', '#f87171'])
        particles += f"""
        <div class="particle" style="
            left: {left}%;
            animation-delay: {delay:.1f}s;
            animation-duration: {duration:.1f}s;
            width: {size:.0f}px;
            height: {size:.0f}px;
            background: {color};
            box-shadow: 0 0 {size*2:.0f}px {color};
        "></div>"""
    return f'<div class="fire-particles">{particles}</div>'


# --- Load model ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_weights')

@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.eval()
    return model

model = load_model()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class_names = ['fire', 'no_fire']

class GradCAMModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.efficientnet = model.efficientnet

    def forward(self, x):
        return self.model(x).logits

gradcam_model = GradCAMModel(model)
gradcam_model.eval()
target_layer = gradcam_model.efficientnet.encoder.blocks[-1]

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
        <div style="width:60px; height:60px; margin:0 auto; background: linear-gradient(135deg, #ef4444, #dc2626);
             border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:1.8rem;
             box-shadow: 0 8px 24px rgba(239,68,68,0.3);">\U0001f525</div>
        <h2 style="margin: 0.8rem 0 0.2rem 0; font-size: 1.2rem; font-weight: 800; letter-spacing: -0.02em;">VIGIE IA</h2>
        <p style="margin:0; font-size: 0.75rem; opacity: 0.5; letter-spacing: 2px;">SDIS 44 \u2022 PROTOTYPE</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>Mod\u00e8le</h4>
        <div class="stat-row"><span class="stat-label">Architecture</span><span class="stat-value">EfficientNet-B0</span></div>
        <div class="stat-row"><span class="stat-label">Entr\u00e9e</span><span class="stat-value">224\u00d7224 RGB</span></div>
        <div class="stat-row"><span class="stat-label">Classes</span><span class="stat-value">2 (feu / normal)</span></div>
        <div class="stat-row"><span class="stat-label">Param\u00e8tres</span><span class="stat-value">~5.3M</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>Performances (test set)</h4>
        <div class="stat-row"><span class="stat-label">Accuracy</span><span class="stat-value" style="color:#10b981 !important;">93%</span></div>
        <div class="stat-row"><span class="stat-label">AUC-ROC</span><span class="stat-value" style="color:#10b981 !important;">0.984</span></div>
        <div class="stat-row"><span class="stat-label">Pr\u00e9cision (feu)</span><span class="stat-value">0.99</span></div>
        <div class="stat-row"><span class="stat-label">Rappel (feu)</span><span class="stat-value">0.91</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>Protocole</h4>
        <div style="font-size: 0.78rem; line-height: 1.8; color: rgba(255,255,255,0.6);">
            <span style="color:#3b82f6;">\u2460</span> T\u00e9l\u00e9verser l'image<br>
            <span style="color:#3b82f6;">\u2461</span> Lire la pr\u00e9diction IA<br>
            <span style="color:#3b82f6;">\u2462</span> V\u00e9rifier le GradCAM<br>
            <span style="color:#3b82f6;">\u2463</span> Confirmer l'alerte<br>
            <span style="color:#ef4444;">\u2464</span> D\u00e9clencher protocole
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Session History ---
    if st.session_state.history:
        st.markdown("""
        <div class="sidebar-section">
            <h4>Historique de session</h4>
        """, unsafe_allow_html=True)
        for entry in reversed(st.session_state.history[-8:]):
            dot_class = "fire" if entry['class'] == 'fire' else "safe"
            result_text = "\U0001f525 Feu" if entry['class'] == 'fire' else "\u2705 Normal"
            conf_color = "#f87171" if entry['class'] == 'fire' else "#6ee7b7"
            st.markdown(f"""
            <div class="history-item">
                <div class="history-dot {dot_class}"></div>
                <span class="history-time">{entry['time']}</span>
                <span class="history-result">{result_text}</span>
                <span class="history-conf" style="color:{conf_color};">{entry['confidence']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; font-size:0.7rem; opacity:0.4; margin-top:2rem; padding-top:1rem; border-top: 1px solid rgba(255,255,255,0.06);">
        SDIS de Loire-Atlantique<br>
        Urgences : <strong>18</strong> ou <strong>112</strong><br><br>
        Allaire & Loret \u2022 Sup de Vinci M1
    </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown("""
<div class="hero-banner">
    <span class="hero-badge">\u25cf Syst\u00e8me actif</span>
    <div class="hero-content">
        <div class="hero-icon">\U0001f525</div>
        <div class="hero-text">
            <h1>Vigie IA \u2014 D\u00e9tection Feux de For\u00eat</h1>
            <p>Syst\u00e8me d'aide \u00e0 la d\u00e9cision par intelligence artificielle \u2022 SDIS 44 Loire-Atlantique</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader(
    "T\u00e9l\u00e9verser une image",
    type=['jpg', 'jpeg', 'png'],
    help="Formats accept\u00e9s : JPG, JPEG, PNG",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Measure analysis time
    start_time = time.time()

    img_tensor = val_transform(image)
    input_tensor = img_tensor.unsqueeze(0).requires_grad_(True)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    pred_idx = probabilities.argmax().item()
    pred_class = class_names[pred_idx]
    confidence = probabilities[pred_idx].item()
    fire_prob = probabilities[0].item()
    no_fire_prob = probabilities[1].item()

    # GradCAM
    input_cam = img_tensor.unsqueeze(0).requires_grad_(True)
    grayscale_cam = compute_gradcam(gradcam_model, input_cam, target_layer)
    img_display = (img_tensor * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
    overlay = overlay_cam_on_image(img_display, grayscale_cam)

    analysis_time = time.time() - start_time

    # Add to history
    st.session_state.history.append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'class': pred_class,
        'confidence': confidence,
        'filename': uploaded_file.name
    })

    # Risk level
    risk_label, risk_class, interpretation = get_risk_level(pred_class, confidence)

    # Fire particles if fire detected
    if pred_class == 'fire':
        st.markdown(generate_particles_html(), unsafe_allow_html=True)

    # --- Alert ---
    if pred_class == 'fire':
        st.markdown(f"""
        <div class="alert-fire-premium">
            <h2>\U0001f6a8 ALERTE \u2014 D\u00c9PART DE FEU D\u00c9TECT\u00c9</h2>
            <p>Confiance : <strong>{confidence:.1%}</strong> \u2022 {datetime.now().strftime('%d/%m/%Y \u00e0 %H:%M:%S')}</p>
            <div class="risk-badge {risk_class}">Niveau de risque : {risk_label}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-safe-premium">
            <h2>\u2705 AUCUNE MENACE D\u00c9TECT\u00c9E</h2>
            <p>Confiance : <strong>{confidence:.1%}</strong> \u2022 {datetime.now().strftime('%d/%m/%Y \u00e0 %H:%M:%S')}</p>
            <div class="risk-badge {risk_class}">Niveau de risque : {risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Analysis time badge
    st.markdown(f"""
    <div style="margin: 1rem 0; display:flex; align-items:center; gap:1rem;">
        <span class="analysis-time">\u23f1 Analyse : {analysis_time:.2f}s</span>
        <span class="analysis-time">\U0001f4c4 {uploaded_file.name}</span>
        <span class="analysis-time">\U0001f4d0 {image.size[0]}\u00d7{image.size[1]}px</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Circular Gauges ---
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.markdown(f"""
        <div class="metric-gauge">
            {svg_gauge(fire_prob, '#f87171', '#ef4444', 'FEU')}
        </div>
        """, unsafe_allow_html=True)
    with col_g2:
        st.markdown(f"""
        <div class="metric-gauge">
            {svg_gauge(no_fire_prob, '#34d399', '#10b981', 'NORMAL')}
        </div>
        """, unsafe_allow_html=True)
    with col_g3:
        st.markdown(f"""
        <div class="metric-gauge">
            {svg_gauge(confidence, '#60a5fa', '#a78bfa', 'CONFIANCE')}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Images ---
    image_resized = image.resize((224, 224))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="image-container">
            <h4>\U0001f4f7 Image source</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(image_resized, use_container_width=True)
    with col2:
        st.markdown("""
        <div class="image-container">
            <h4>\U0001f9e0 GradCAM \u2014 Carte d'attention</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)

    # --- Interpretation ---
    st.markdown(f"""
    <div class="interpretation">
        <strong>\U0001f4cb Interpr\u00e9tation :</strong> {interpretation}
        <br><br>
        <strong>\U0001f9e0 GradCAM :</strong> Les zones lumineuses sur la carte d'attention montrent les r\u00e9gions
        de l'image ayant le plus influenc\u00e9 la d\u00e9cision. V\u00e9rifiez que ces zones correspondent bien
        \u00e0 des \u00e9l\u00e9ments pertinents (flammes, fum\u00e9e) et non \u00e0 des artefacts.
    </div>
    """, unsafe_allow_html=True)

    # --- Download report ---
    report_text = f"""RAPPORT D'ANALYSE - VIGIE IA SDIS 44
{'='*50}
Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Fichier : {uploaded_file.name}
Dimensions : {image.size[0]}x{image.size[1]} pixels
Temps d'analyse : {analysis_time:.2f}s

RESULTAT
{'-'*50}
Prediction : {'FEU DETECTE' if pred_class == 'fire' else 'PAS DE FEU'}
Confiance : {confidence:.1%}
Niveau de risque : {risk_label}
P(feu) : {fire_prob:.4f}
P(normal) : {no_fire_prob:.4f}

INTERPRETATION
{'-'*50}
{interpretation}

AVERTISSEMENT
{'-'*50}
Ce systeme est un outil d'aide a la decision uniquement.
Toute alerte doit etre confirmee par un operateur qualifie.
"""
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="\U0001f4e5 T\u00e9l\u00e9charger le rapport d'analyse",
        data=report_text,
        file_name=f"rapport_vigie_ia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    # --- Disclaimer ---
    st.markdown("""
    <div class="disclaimer-premium">
        <strong>\u26a0\ufe0f AVERTISSEMENT</strong> \u2014 Ce syst\u00e8me est un outil d'aide \u00e0 la d\u00e9cision uniquement.
        Toute alerte doit imp\u00e9rativement \u00eatre confirm\u00e9e par un op\u00e9rateur qualifi\u00e9 avant d\u00e9clenchement
        du protocole d'intervention. Aucune d\u00e9cision op\u00e9rationnelle ne doit \u00eatre prise sur la seule base de cette analyse.
    </div>
    """, unsafe_allow_html=True)

else:
    # --- Welcome state ---
    st.markdown("<br>", unsafe_allow_html=True)

    col_w1, col_w2, col_w3 = st.columns([1, 2, 1])
    with col_w2:
        st.markdown("""
        <div class="dropzone">
            <div class="dropzone-icon">\U0001f4e1</div>
            <h3 style="color: var(--text-primary); margin: 0.5rem 0; font-weight: 700;">En attente d'image</h3>
            <p style="color: var(--text-secondary); margin: 0; font-size: 0.85rem;">
                Glissez-d\u00e9posez ou s\u00e9lectionnez une capture de cam\u00e9ra de surveillance
            </p>
            <p style="color: rgba(255,255,255,0.3); font-size: 0.75rem; margin-top: 0.5rem;">
                JPG \u2022 JPEG \u2022 PNG
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Feature cards
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(59,130,246,0.2));">\U0001f9e0</div>
            <h4>Deep Learning</h4>
            <p>EfficientNet-B0 fine-tun\u00e9 sur 1000+ images</p>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(245,158,11,0.2));">\u26a1</div>
            <h4>Temps r\u00e9el</h4>
            <p>Analyse instantan\u00e9e en quelques secondes</p>
        </div>
        """, unsafe_allow_html=True)
    with col_f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(52,211,153,0.2));">\U0001f50d</div>
            <h4>Explicabilit\u00e9</h4>
            <p>GradCAM pour comprendre les d\u00e9cisions IA</p>
        </div>
        """, unsafe_allow_html=True)
    with col_f4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, rgba(168,85,247,0.2), rgba(139,92,246,0.2));">\U0001f4ca</div>
            <h4>AUC 0.984</h4>
            <p>Performance valid\u00e9e sur jeu de test ind\u00e9pendant</p>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer-premium">
    <strong>SDIS 44</strong> \u2022 Service D\u00e9partemental d'Incendie et de Secours de Loire-Atlantique<br>
    Vigie IA v1.0 \u2022 Prototype de recherche \u2022 Allaire & Loret \u2022 Sup de Vinci M1<br>
    Pour toute urgence : <strong>18</strong> ou <strong>112</strong>
</div>
""", unsafe_allow_html=True)
