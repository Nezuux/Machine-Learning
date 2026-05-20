# -*- coding: utf-8 -*-
import streamlit as st
import torch
import numpy as np
import os
import time
import io
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoModelForImageClassification
from datetime import datetime
import matplotlib.cm as cm

st.set_page_config(
    page_title='SDIS 44 - Vigie IA',
    page_icon='\U0001f6a8',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --- Session state ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_fires' not in st.session_state:
    st.session_state.total_fires = 0
if 'total_safe' not in st.session_state:
    st.session_state.total_safe = 0

# --- ULTIMATE CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #050810;
        --bg-secondary: #0c1122;
        --bg-card: rgba(12, 17, 34, 0.85);
        --border-glass: rgba(255, 255, 255, 0.06);
        --border-hover: rgba(99, 102, 241, 0.4);
        --text-primary: #f1f5f9;
        --text-secondary: #64748b;
        --text-muted: #475569;
        --accent-indigo: #6366f1;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-red: #ef4444;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
        --accent-purple: #a855f7;
        --glow-red: rgba(239, 68, 68, 0.4);
        --glow-green: rgba(16, 185, 129, 0.3);
        --glow-blue: rgba(59, 130, 246, 0.3);
    }

    .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
        background: var(--bg-primary) !important;
        color: var(--text-primary);
    }

    /* Animated mesh gradient background */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 80% 50% at 20% 40%, rgba(239, 68, 68, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse 60% 80% at 80% 20%, rgba(99, 102, 241, 0.07) 0%, transparent 50%),
            radial-gradient(ellipse 50% 60% at 60% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 50%),
            radial-gradient(ellipse 40% 40% at 10% 90%, rgba(168, 85, 247, 0.04) 0%, transparent 50%);
        animation: meshMove 20s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    .stApp::after {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(circle at 25% 25%, rgba(255,255,255,0.015) 1px, transparent 1px),
            radial-gradient(circle at 75% 75%, rgba(255,255,255,0.01) 1px, transparent 1px);
        background-size: 60px 60px, 90px 90px;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes meshMove {
        0% { transform: scale(1) rotate(0deg); opacity: 0.8; }
        50% { transform: scale(1.05) rotate(1deg); opacity: 1; }
        100% { transform: scale(1) rotate(-1deg); opacity: 0.9; }
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.6rem 1.2rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-indigo), var(--accent-blue)) !important;
        color: white !important;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b14 0%, #0f0a2e 100%) !important;
        border-right: 1px solid var(--border-glass);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f0a2e 0%, var(--bg-primary) 40%, #1a0a0a 100%);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .hero::before {
        content: '';
        position: absolute;
        top: -100px;
        right: -100px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(239,68,68,0.08) 0%, transparent 70%);
        animation: breathe 6s ease-in-out infinite;
    }

    .hero::after {
        content: '';
        position: absolute;
        bottom: -100px;
        left: -100px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%);
        animation: breathe 8s ease-in-out infinite reverse;
    }

    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 1; }
    }

    .hero-inner {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .hero-left {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }

    .hero-logo {
        width: 64px;
        height: 64px;
        background: linear-gradient(135deg, #ef4444, #b91c1c);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        box-shadow: 0 0 30px rgba(239,68,68,0.3), 0 0 60px rgba(239,68,68,0.1);
        animation: logoGlow 4s ease-in-out infinite;
    }

    @keyframes logoGlow {
        0%, 100% { box-shadow: 0 0 30px rgba(239,68,68,0.3); }
        50% { box-shadow: 0 0 50px rgba(239,68,68,0.5), 0 0 80px rgba(239,68,68,0.2); }
    }

    .hero-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
        letter-spacing: -0.03em;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0.2rem 0 0 0;
    }

    .hero-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.25);
        padding: 0.5rem 1rem;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #6ee7b7;
    }

    .hero-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        animation: blink 2s ease-in-out infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* Alert panels */
    .alert-panel {
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }

    .alert-fire {
        background: linear-gradient(135deg, rgba(127,29,29,0.7), rgba(153,27,27,0.5));
        border: 1px solid rgba(239,68,68,0.4);
        animation: alertPulse 2s ease-in-out infinite;
    }

    .alert-safe {
        background: linear-gradient(135deg, rgba(6,78,59,0.5), rgba(4,120,87,0.3));
        border: 1px solid rgba(16,185,129,0.3);
    }

    @keyframes alertPulse {
        0%, 100% { box-shadow: 0 0 30px rgba(239,68,68,0.15); }
        50% { box-shadow: 0 0 60px rgba(239,68,68,0.3); }
    }

    .alert-panel h2 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.03em;
    }

    .alert-fire h2 { color: #fca5a5 !important; }
    .alert-safe h2 { color: #6ee7b7 !important; }

    .alert-panel .alert-sub {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }

    .alert-fire .alert-sub { color: #fecaca; }
    .alert-safe .alert-sub { color: #a7f3d0; }

    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 1rem;
    }

    .risk-critical { background: rgba(239,68,68,0.2); border: 1px solid rgba(239,68,68,0.5); color: #fca5a5; }
    .risk-high { background: rgba(245,158,11,0.2); border: 1px solid rgba(245,158,11,0.4); color: #fcd34d; }
    .risk-moderate { background: rgba(59,130,246,0.2); border: 1px solid rgba(59,130,246,0.4); color: #93c5fd; }
    .risk-low { background: rgba(16,185,129,0.2); border: 1px solid rgba(16,185,129,0.4); color: #6ee7b7; }

    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-indigo), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(99,102,241,0.08);
    }

    .metric-card:hover::before { opacity: 1; }

    .metric-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        margin-bottom: 0.6rem;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
    }

    .metric-value.red { color: #f87171; }
    .metric-value.green { color: #34d399; }
    .metric-value.blue { color: #60a5fa; }
    .metric-value.purple { color: #a78bfa; }

    .metric-bar {
        height: 4px;
        border-radius: 2px;
        background: rgba(255,255,255,0.06);
        margin-top: 0.8rem;
        overflow: hidden;
    }

    .metric-bar-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-bar-fill.red { background: linear-gradient(90deg, #f87171, #ef4444); }
    .metric-bar-fill.green { background: linear-gradient(90deg, #34d399, #10b981); }
    .metric-bar-fill.blue { background: linear-gradient(90deg, #60a5fa, #3b82f6); }
    .metric-bar-fill.purple { background: linear-gradient(90deg, #a78bfa, #8b5cf6); }

    /* Image panels */
    .img-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.2rem;
        transition: all 0.3s ease;
    }

    .img-panel:hover {
        border-color: var(--border-hover);
    }

    .img-panel-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid var(--border-glass);
    }

    .img-panel-header span {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
    }

    .img-panel-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }

    /* Interpretation */
    .interp-box {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }

    .interp-box h5 {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        margin: 0 0 0.8rem 0;
    }

    .interp-box p {
        font-size: 0.88rem;
        line-height: 1.7;
        color: var(--text-secondary);
        margin: 0;
    }

    .interp-box strong {
        color: var(--text-primary);
    }

    /* Info badges */
    .info-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin: 1rem 0;
    }

    .info-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 8px;
        padding: 0.35rem 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #a5b4fc;
    }

    /* Stats dashboard */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }

    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-2px);
    }

    .stat-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }

    .stat-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }

    /* Welcome */
    .welcome-zone {
        background: linear-gradient(135deg, rgba(99,102,241,0.04), rgba(59,130,246,0.04));
        border: 2px dashed rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }

    .welcome-zone:hover {
        border-color: rgba(99,102,241,0.4);
        background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(59,130,246,0.06));
    }

    .welcome-icon {
        font-size: 3.5rem;
        animation: radar 3s ease-in-out infinite;
    }

    @keyframes radar {
        0%, 100% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.1); opacity: 1; }
    }

    .welcome-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1rem 0 0.3rem;
    }

    .welcome-desc {
        font-size: 0.88rem;
        color: var(--text-secondary);
        max-width: 400px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Feature grid */
    .feat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 2.5rem;
    }

    .feat-item {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.5rem 1rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .feat-item:hover {
        border-color: var(--border-hover);
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(99,102,241,0.1);
    }

    .feat-emoji {
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
    }

    .feat-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }

    .feat-desc {
        font-size: 0.72rem;
        color: var(--text-muted);
        line-height: 1.4;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(245,158,11,0.06);
        border: 1px solid rgba(245,158,11,0.15);
        border-radius: 12px;
        padding: 1rem 1.3rem;
        font-size: 0.78rem;
        color: var(--accent-orange);
        margin-top: 1.5rem;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border-glass);
        font-size: 0.72rem;
        color: var(--text-muted);
    }

    /* Sidebar custom */
    .sb-block {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.9rem;
        margin: 0.7rem 0;
    }

    .sb-block-title {
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.35) !important;
        margin: 0 0 0.7rem 0;
        font-weight: 600;
    }

    .sb-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.78rem;
    }

    .sb-row:last-child { border-bottom: none; }
    .sb-row-label { color: rgba(255,255,255,0.45); }
    .sb-row-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: rgba(255,255,255,0.85); font-size: 0.75rem; }

    /* History */
    .hist-entry {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.45rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.35rem;
        font-size: 0.72rem;
        background: rgba(255,255,255,0.02);
    }

    .hist-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .hist-dot-fire { background: #ef4444; }
    .hist-dot-safe { background: #10b981; }

    /* Comparison panel */
    .comparison-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        margin: 1.5rem 0;
        padding: 1rem;
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
    }

    .comp-vs {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple));
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.75rem;
        color: white;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent-indigo), var(--accent-blue)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.3) !important;
    }

    /* File uploader */
    .stFileUploader > div {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# --- GradCAM ---
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
    if pred_class == 'fire':
        if confidence >= 0.9:
            return 'CRITIQUE', 'risk-critical', 'Forte probabilit\u00e9 de d\u00e9part de feu. V\u00e9rification imm\u00e9diate requise.'
        elif confidence >= 0.7:
            return '\u00c9LEV\u00c9', 'risk-high', 'Suspicion importante. Confirmation visuelle recommand\u00e9e.'
        else:
            return 'MOD\u00c9R\u00c9', 'risk-moderate', 'Signal faible. Possibilit\u00e9 de faux positif (reflets, coucher de soleil).'
    else:
        if confidence >= 0.9:
            return 'FAIBLE', 'risk-low', 'Zone s\u00e9curis\u00e9e. Aucun signe de feu identifi\u00e9.'
        else:
            return 'MOD\u00c9R\u00c9', 'risk-moderate', 'Pas de feu d\u00e9tect\u00e9 mais confiance limit\u00e9e.'


def svg_gauge(value, color1, color2, label):
    circumference = 2 * 3.14159 * 45
    offset = circumference * (1 - value)
    uid = label.replace(' ', '_')
    return f"""
    <div style="position:relative; width:120px; height:120px; margin:0 auto;">
        <svg viewBox="0 0 100 100" style="transform:rotate(-90deg); width:120px; height:120px;">
            <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="7"/>
            <circle cx="50" cy="50" r="45" fill="none" stroke="url(#g{uid})" stroke-width="7"
                stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                style="transition: stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1);"/>
            <defs><linearGradient id="g{uid}"><stop offset="0%" stop-color="{color1}"/><stop offset="100%" stop-color="{color2}"/></linearGradient></defs>
        </svg>
        <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:{color1};">{value:.0%}</div>
            <div style="font-size:0.55rem; text-transform:uppercase; letter-spacing:1px; color:var(--text-muted); margin-top:2px;">{label}</div>
        </div>
    </div>
    """


# --- Load model ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_weights')

@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR, num_labels=2, ignore_mismatched_sizes=True
    )
    model.eval()
    return model

model = load_model()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
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


# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem;">
        <div style="width:56px; height:56px; margin:0 auto; background:linear-gradient(135deg,#ef4444,#991b1b);
             border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:1.6rem;
             box-shadow: 0 0 30px rgba(239,68,68,0.25);">\U0001f525</div>
        <h2 style="margin:0.6rem 0 0; font-size:1.1rem; font-weight:800; letter-spacing:-0.02em;">VIGIE IA</h2>
        <p style="margin:0; font-size:0.68rem; opacity:0.4; letter-spacing:2px;">SDIS 44 \u2022 v2.0</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-block">
        <div class="sb-block-title">Mod\u00e8le</div>
        <div class="sb-row"><span class="sb-row-label">Architecture</span><span class="sb-row-val">EfficientNet-B0</span></div>
        <div class="sb-row"><span class="sb-row-label">Entr\u00e9e</span><span class="sb-row-val">224\u00d7224</span></div>
        <div class="sb-row"><span class="sb-row-label">Param\u00e8tres</span><span class="sb-row-val">5.3M</span></div>
        <div class="sb-row"><span class="sb-row-label">Framework</span><span class="sb-row-val">PyTorch + HF</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-block">
        <div class="sb-block-title">Performances</div>
        <div class="sb-row"><span class="sb-row-label">Accuracy</span><span class="sb-row-val" style="color:#10b981 !important;">93%</span></div>
        <div class="sb-row"><span class="sb-row-label">AUC-ROC</span><span class="sb-row-val" style="color:#10b981 !important;">0.984</span></div>
        <div class="sb-row"><span class="sb-row-label">Pr\u00e9cision (feu)</span><span class="sb-row-val">0.99</span></div>
        <div class="sb-row"><span class="sb-row-label">Rappel (feu)</span><span class="sb-row-val">0.91</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Session stats
    total = len(st.session_state.history)
    if total > 0:
        st.markdown(f"""
        <div class="sb-block">
            <div class="sb-block-title">Session</div>
            <div class="sb-row"><span class="sb-row-label">Analyses</span><span class="sb-row-val">{total}</span></div>
            <div class="sb-row"><span class="sb-row-label">Alertes feu</span><span class="sb-row-val" style="color:#f87171 !important;">{st.session_state.total_fires}</span></div>
            <div class="sb-row"><span class="sb-row-label">Normales</span><span class="sb-row-val" style="color:#34d399 !important;">{st.session_state.total_safe}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # History
    if st.session_state.history:
        st.markdown('<div class="sb-block"><div class="sb-block-title">Historique</div>', unsafe_allow_html=True)
        for entry in reversed(st.session_state.history[-6:]):
            dot = "hist-dot-fire" if entry['class'] == 'fire' else "hist-dot-safe"
            icon = "\U0001f525" if entry['class'] == 'fire' else "\u2705"
            st.markdown(f"""
            <div class="hist-entry">
                <div class="hist-dot {dot}"></div>
                <span style="color:rgba(255,255,255,0.4); font-family:'JetBrains Mono',monospace;">{entry['time']}</span>
                <span style="color:rgba(255,255,255,0.7); flex:1;">{icon} {entry['confidence']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; font-size:0.65rem; opacity:0.3; margin-top:2rem; padding-top:0.8rem; border-top:1px solid rgba(255,255,255,0.05);">
        Allaire & Loret<br>Sup de Vinci M1<br>2025-2026
    </div>
    """, unsafe_allow_html=True)


# ========== HERO ==========
st.markdown("""
<div class="hero">
    <div class="hero-inner">
        <div class="hero-left">
            <div class="hero-logo">\U0001f525</div>
            <div>
                <h1 class="hero-title">Vigie IA \u2014 D\u00e9tection Feux de For\u00eat</h1>
                <p class="hero-subtitle">SDIS 44 \u2022 Loire-Atlantique \u2022 Syst\u00e8me d'aide \u00e0 la d\u00e9cision</p>
            </div>
        </div>
        <div class="hero-status">
            <div class="hero-status-dot"></div>
            Syst\u00e8me op\u00e9rationnel
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ========== TABS ==========
tab_analyse, tab_multi = st.tabs(["\U0001f50d  Analyse", "\U0001f4ca  Multi-analyse"])

# ========== TAB 1: ANALYSE ==========
with tab_analyse:
    uploaded_file = st.file_uploader(
        "T\u00e9l\u00e9verser une image",
        type=['jpg', 'jpeg', 'png'],
        help="Formats : JPG, JPEG, PNG",
        label_visibility="collapsed",
        key="single_upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        start_time = time.time()

        img_tensor = val_transform(image)
        input_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = probabilities.argmax().item()
        pred_class = class_names[pred_idx]
        confidence = probabilities[pred_idx].item()
        fire_prob = probabilities[0].item()
        no_fire_prob = probabilities[1].item()

        input_cam = img_tensor.unsqueeze(0).requires_grad_(True)
        grayscale_cam = compute_gradcam(gradcam_model, input_cam, target_layer)
        img_display = (img_tensor * std_t + mean_t).permute(1, 2, 0).numpy().clip(0, 1)
        overlay = overlay_cam_on_image(img_display, grayscale_cam)

        analysis_time = time.time() - start_time

        # Update session
        st.session_state.history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'class': pred_class,
            'confidence': confidence,
            'filename': uploaded_file.name
        })
        if pred_class == 'fire':
            st.session_state.total_fires += 1
        else:
            st.session_state.total_safe += 1

        risk_label, risk_class, interpretation = get_risk_level(pred_class, confidence)

        # Alert
        if pred_class == 'fire':
            st.markdown(f"""
            <div class="alert-panel alert-fire">
                <h2>\U0001f6a8 ALERTE \u2014 FEU D\u00c9TECT\u00c9</h2>
                <p class="alert-sub">Confiance : <strong>{confidence:.1%}</strong> \u2022 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                <div class="risk-badge {risk_class}">\u25cf Niveau : {risk_label}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-panel alert-safe">
                <h2>\u2705 AUCUNE MENACE</h2>
                <p class="alert-sub">Confiance : <strong>{confidence:.1%}</strong> \u2022 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                <div class="risk-badge {risk_class}">\u25cf Niveau : {risk_label}</div>
            </div>
            """, unsafe_allow_html=True)

        # Info badges
        st.markdown(f"""
        <div class="info-row">
            <span class="info-badge">\u23f1 {analysis_time:.2f}s</span>
            <span class="info-badge">\U0001f4c4 {uploaded_file.name}</span>
            <span class="info-badge">\U0001f4d0 {image.size[0]}\u00d7{image.size[1]}px</span>
            <span class="info-badge">\U0001f9e0 EfficientNet-B0</span>
        </div>
        """, unsafe_allow_html=True)

        # Gauges
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        with col_g1:
            st.markdown(f'<div class="metric-card">{svg_gauge(fire_prob, "#f87171", "#ef4444", "FEU")}</div>', unsafe_allow_html=True)
        with col_g2:
            st.markdown(f'<div class="metric-card">{svg_gauge(no_fire_prob, "#34d399", "#10b981", "NORMAL")}</div>', unsafe_allow_html=True)
        with col_g3:
            st.markdown(f'<div class="metric-card">{svg_gauge(confidence, "#60a5fa", "#3b82f6", "CONFIANCE")}</div>', unsafe_allow_html=True)
        with col_g4:
            st.markdown(f"""
            <div class="metric-card" style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:150px;">
                <div class="metric-label">Temps</div>
                <div class="metric-value blue">{analysis_time:.2f}s</div>
                <div style="font-size:0.65rem; color:var(--text-muted); margin-top:0.5rem;">Inf\u00e9rence GPU/CPU</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="img-panel">
                <div class="img-panel-header">
                    <div class="img-panel-dot" style="background:#3b82f6;"></div>
                    <span>Image source (224\u00d7224)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(image.resize((224, 224)), use_container_width=True)
        with col2:
            st.markdown("""
            <div class="img-panel">
                <div class="img-panel-header">
                    <div class="img-panel-dot" style="background:#a855f7;"></div>
                    <span>GradCAM \u2014 Attention (224\u00d7224)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(overlay, use_container_width=True)

        # Interpretation
        st.markdown(f"""
        <div class="interp-box">
            <h5>\U0001f4cb Interpr\u00e9tation de l'analyse</h5>
            <p><strong>R\u00e9sultat :</strong> {interpretation}</p>
            <p style="margin-top:0.8rem;"><strong>GradCAM :</strong> Les zones lumineuses montrent o\u00f9 le mod\u00e8le
            concentre son attention. V\u00e9rifiez qu'elles correspondent \u00e0 des \u00e9l\u00e9ments pertinents
            (flammes, fum\u00e9e) et non \u00e0 des artefacts visuels.</p>
        </div>
        """, unsafe_allow_html=True)

        # Download report
        report = f"""RAPPORT D'ANALYSE - VIGIE IA SDIS 44
{'='*50}
Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Fichier : {uploaded_file.name}
Dimensions : {image.size[0]}x{image.size[1]} pixels
Temps d'analyse : {analysis_time:.3f}s

RESULTAT
{'-'*50}
Prediction : {'FEU DETECTE' if pred_class == 'fire' else 'AUCUN FEU'}
Confiance : {confidence:.1%}
Niveau de risque : {risk_label}
P(feu) = {fire_prob:.4f}
P(normal) = {no_fire_prob:.4f}

INTERPRETATION
{'-'*50}
{interpretation}

AVERTISSEMENT
{'-'*50}
Outil d'aide a la decision uniquement.
Toute alerte doit etre confirmee par un operateur qualifie.
"""
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                "\U0001f4e5 T\u00e9l\u00e9charger le rapport",
                data=report,
                file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
            <strong>\u26a0\ufe0f</strong> Outil d'aide \u00e0 la d\u00e9cision uniquement. Toute alerte doit \u00eatre confirm\u00e9e
            par un op\u00e9rateur qualifi\u00e9 avant intervention.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Welcome
        st.markdown("")
        col_w1, col_w2, col_w3 = st.columns([1, 2, 1])
        with col_w2:
            st.markdown("""
            <div class="welcome-zone">
                <div class="welcome-icon">\U0001f4e1</div>
                <div class="welcome-title">En attente d'image</div>
                <p class="welcome-desc">T\u00e9l\u00e9versez une capture de cam\u00e9ra de surveillance pour lancer l'analyse de d\u00e9tection de feux de for\u00eat.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feat-grid">
            <div class="feat-item">
                <div class="feat-emoji">\U0001f9e0</div>
                <div class="feat-name">Deep Learning</div>
                <div class="feat-desc">EfficientNet-B0 fine-tun\u00e9 sp\u00e9cifiquement</div>
            </div>
            <div class="feat-item">
                <div class="feat-emoji">\u26a1</div>
                <div class="feat-name">Temps r\u00e9el</div>
                <div class="feat-desc">Inf\u00e9rence en moins de 2 secondes</div>
            </div>
            <div class="feat-item">
                <div class="feat-emoji">\U0001f50d</div>
                <div class="feat-name">Explicabilit\u00e9</div>
                <div class="feat-desc">GradCAM pour la transparence IA</div>
            </div>
            <div class="feat-item">
                <div class="feat-emoji">\U0001f3af</div>
                <div class="feat-name">AUC 0.984</div>
                <div class="feat-desc">Valid\u00e9 sur jeu de test ind\u00e9pendant</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ========== TAB 2: MULTI-ANALYSE ==========
with tab_multi:
    st.markdown("""
    <div class="interp-box">
        <h5>\U0001f4f7 Analyse par lot</h5>
        <p>T\u00e9l\u00e9versez plusieurs images pour les analyser simultan\u00e9ment et comparer les r\u00e9sultats.</p>
    </div>
    """, unsafe_allow_html=True)

    multi_files = st.file_uploader(
        "T\u00e9l\u00e9verser plusieurs images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="multi_upload"
    )

    if multi_files:
        results = []
        for f in multi_files:
            img = Image.open(f).convert('RGB')
            t = val_transform(img)
            with torch.no_grad():
                out = model(t.unsqueeze(0))
                probs = torch.softmax(out.logits, dim=-1)[0]
            idx = probs.argmax().item()
            results.append({
                'name': f.name,
                'image': img.resize((224, 224)),
                'class': class_names[idx],
                'confidence': probs[idx].item(),
                'fire_prob': probs[0].item()
            })

        # Summary
        n_fire = sum(1 for r in results if r['class'] == 'fire')
        n_safe = len(results) - n_fire
        st.markdown(f"""
        <div class="comparison-header">
            <div style="text-align:center;">
                <div style="font-size:2rem;">\U0001f525</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:800; color:#f87171;">{n_fire}</div>
                <div style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px;">Alertes</div>
            </div>
            <div class="comp-vs">VS</div>
            <div style="text-align:center;">
                <div style="font-size:2rem;">\U0001f333</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:800; color:#34d399;">{n_safe}</div>
                <div style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px;">Normales</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Grid of results
        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(results):
                    r = results[i + j]
                    with col:
                        border_color = "rgba(239,68,68,0.4)" if r['class'] == 'fire' else "rgba(16,185,129,0.3)"
                        icon = "\U0001f525" if r['class'] == 'fire' else "\u2705"
                        color = "#f87171" if r['class'] == 'fire' else "#34d399"
                        st.markdown(f"""
                        <div style="background:var(--bg-card); border:1px solid {border_color}; border-radius:12px; padding:1rem; margin-bottom:1rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                                <span style="font-size:0.75rem; color:var(--text-secondary);">{r['name'][:20]}</span>
                                <span style="font-family:'JetBrains Mono',monospace; font-weight:700; color:{color};">{icon} {r['confidence']:.0%}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(r['image'], use_container_width=True)


# ========== FOOTER ==========
st.markdown("""
<div class="app-footer">
    <strong>SDIS 44</strong> \u2022 Service D\u00e9partemental d'Incendie et de Secours de Loire-Atlantique<br>
    Vigie IA v2.0 \u2022 Allaire & Loret \u2022 Sup de Vinci Master 1 \u2022 2025-2026<br>
    Urgences : <strong>18</strong> ou <strong>112</strong>
</div>
""", unsafe_allow_html=True)
