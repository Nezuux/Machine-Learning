# -*- coding: utf-8 -*-
import streamlit as st
import torch
import numpy as np
import os
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

    /* Hide default Streamlit elements */
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

    /* Main header */
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

    /* Gauge / metric */
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

    /* Progress bar custom */
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

    /* Welcome cards */
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

    /* Upload dropzone */
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

    .footer-premium a {
        color: var(--accent-blue);
        text-decoration: none;
    }

    /* Scanning animation */
    .scan-line {
        position: relative;
        overflow: hidden;
    }

    .scan-line::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
        animation: scan 2s linear infinite;
    }

    @keyframes scan {
        from { left: -100%; }
        to { left: 100%; }
    }

    /* Sidebar styling */
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

    .stat-row:last-child {
        border-bottom: none;
    }

    .stat-label {
        color: rgba(255,255,255,0.5);
    }

    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: rgba(255,255,255,0.9);
    }

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
</style>
""", unsafe_allow_html=True)

# --- GradCAM implementation (no opencv dependency) ---
def compute_gradcam(model, input_tensor, target_layer):
    """Compute GradCAM heatmap using hooks."""
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
    """Overlay GradCAM heatmap on image. Returns PIL Image."""
    colormap = cm.get_cmap('inferno')
    heatmap = colormap(cam_map)[:, :, :3]
    overlay = (1 - alpha) * img_array + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


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

# Upload section
uploaded_file = st.file_uploader(
    "T\u00e9l\u00e9verser une image de cam\u00e9ra de surveillance",
    type=['jpg', 'jpeg', 'png'],
    help="Formats accept\u00e9s : JPG, JPEG, PNG",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    with st.spinner(''):
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

    # --- Alert ---
    if pred_class == 'fire':
        st.markdown(f"""
        <div class="alert-fire-premium">
            <h2>\U0001f6a8 ALERTE \u2014 D\u00c9PART DE FEU D\u00c9TECT\u00c9</h2>
            <p>Confiance : <strong>{confidence:.1%}</strong> \u2022 {datetime.now().strftime('%d/%m/%Y \u00e0 %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-safe-premium">
            <h2>\u2705 AUCUNE MENACE D\u00c9TECT\u00c9E</h2>
            <p>Confiance : <strong>{confidence:.1%}</strong> \u2022 {datetime.now().strftime('%d/%m/%Y \u00e0 %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Metrics ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    bar_class = "fire" if pred_class == 'fire' else "safe"
    with col_m1:
        st.markdown(f"""
        <div class="metric-gauge">
            <div class="label">\U0001f525 Probabilit\u00e9 Feu</div>
            <div class="value fire">{fire_prob:.1%}</div>
            <div class="confidence-bar"><div class="confidence-bar-fill fire" style="width:{fire_prob*100:.0f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-gauge">
            <div class="label">\U0001f333 Probabilit\u00e9 Normal</div>
            <div class="value safe">{no_fire_prob:.1%}</div>
            <div class="confidence-bar"><div class="confidence-bar-fill safe" style="width:{no_fire_prob*100:.0f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-gauge">
            <div class="label">\U0001f3af Confiance</div>
            <div class="value {bar_class}">{confidence:.1%}</div>
            <div class="confidence-bar"><div class="confidence-bar-fill {bar_class}" style="width:{confidence*100:.0f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-gauge">
            <div class="label">\u23f1 Horodatage</div>
            <div class="timestamp" style="margin-top:0.6rem; display:inline-block;">{datetime.now().strftime('%H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Images ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="image-container">
            <h4>\U0001f4f7 Image source</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("""
        <div class="image-container">
            <h4>\U0001f9e0 GradCAM \u2014 Carte d'attention</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
        st.markdown("""
        <p style="font-size:0.75rem; color: var(--text-secondary); margin-top:0.5rem; text-align:center;">
            Les zones claires indiquent les r\u00e9gions ayant le plus influenc\u00e9 la d\u00e9cision du mod\u00e8le.
        </p>
        """, unsafe_allow_html=True)

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
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(59,130,246,0.2));">\U0001f9e0</div>
            <h4>Deep Learning</h4>
            <p>EfficientNet-B0 fine-tun\u00e9 sur 1000+ images a\u00e9riennes et satellite pour la d\u00e9tection de feux</p>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(245,158,11,0.2));">\u26a1</div>
            <h4>Temps r\u00e9el</h4>
            <p>Analyse instantan\u00e9e des images avec pr\u00e9diction et score de confiance en quelques secondes</p>
        </div>
        """, unsafe_allow_html=True)
    with col_f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(52,211,153,0.2));">\U0001f50d</div>
            <h4>Explicabilit\u00e9</h4>
            <p>Visualisation GradCAM des zones d'attention pour comprendre et valider la d\u00e9cision de l'IA</p>
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
