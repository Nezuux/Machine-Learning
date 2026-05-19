# -*- coding: utf-8 -*-
import streamlit as st
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from datetime import datetime

st.set_page_config(
    page_title='SDIS 44 - Vigie IA Feux de For\u00eat',
    page_icon='\U0001f6a8',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --- Custom CSS style SDIS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');

    .stApp {
        font-family: 'Roboto', sans-serif;
    }

    /* Header banner */
    .sdis-header {
        background: linear-gradient(135deg, #1a237e 0%, #c62828 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .sdis-header img {
        height: 80px;
    }

    .sdis-header-text {
        color: white;
    }

    .sdis-header-text h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: white !important;
    }

    .sdis-header-text p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
        color: #e0e0e0;
    }

    /* Alert boxes */
    .alert-fire {
        background: linear-gradient(135deg, #b71c1c, #e53935);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 20px rgba(183, 28, 28, 0.4);
    }

    .alert-safe {
        background: linear-gradient(135deg, #1b5e20, #43a047);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(27, 94, 32, 0.3);
    }

    .alert-fire h2, .alert-safe h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: white !important;
    }

    .alert-fire p, .alert-safe p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        color: white;
    }

    @keyframes pulse {
        0% { box-shadow: 0 4px 20px rgba(183, 28, 28, 0.4); }
        50% { box-shadow: 0 4px 30px rgba(183, 28, 28, 0.8); }
        100% { box-shadow: 0 4px 20px rgba(183, 28, 28, 0.4); }
    }

    /* Metric cards */
    .metric-card {
        background: #f5f5f5;
        border-left: 4px solid #1a237e;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }

    .metric-card h4 {
        margin: 0;
        color: #1a237e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212121;
        margin: 0.2rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e 0%, #0d1442 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #e0e0e0 !important;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff3e0;
        border-left: 4px solid #e65100;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1.5rem;
        font-size: 0.85rem;
        color: #424242;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #1a237e;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #e8eaf6;
        margin: 1rem 0;
    }

    /* Footer */
    .sdis-footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 2px solid #1a237e;
        color: #616161;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

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

class GradCAMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.efficientnet = model.efficientnet

    def forward(self, x):
        return self.model(x).logits

wrapped_model = GradCAMWrapper(model)
wrapped_model.eval()

target_layer = wrapped_model.efficientnet.encoder.blocks[-1]
cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <img src="https://www.sdis44.fr/wp-content/uploads/logo-sdis44-white.svg" width="180" alt="SDIS 44">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### \U0001f6a8 Vigie IA")
    st.markdown("""
    Syst\u00e8me d'aide \u00e0 la d\u00e9tection de d\u00e9parts de feux
    de for\u00eat par intelligence artificielle.
    """)

    st.markdown("---")
    st.markdown("#### Sp\u00e9cifications techniques")
    st.markdown("""
    | Param\u00e8tre | Valeur |
    |-----------|--------|
    | Mod\u00e8le | EfficientNet-B0 |
    | Entr\u00e9e | 224\u00d7224 RGB |
    | Classes | Feu / Pas de feu |
    | AUC Test | 0.984 |
    | Accuracy | 93% |
    """)

    st.markdown("---")
    st.markdown("#### Protocole d'utilisation")
    st.markdown("""
    1. T\u00e9l\u00e9verser l'image cam\u00e9ra
    2. Analyser la pr\u00e9diction IA
    3. V\u00e9rifier via GradCAM
    4. Confirmer ou infirmer l'alerte
    5. D\u00e9clencher le protocole si confirm\u00e9
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; font-size:0.75rem; opacity:0.7;">
        SDIS de Loire-Atlantique<br>
        12 rue Arago - La Chapelle-sur-Erdre<br>
        Urgences : <strong>18</strong> ou <strong>112</strong>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown("""
<div class="sdis-header">
    <div>
        <img src="https://www.sdis44.fr/wp-content/uploads/logo-sdis44-white.svg" alt="SDIS 44">
    </div>
    <div class="sdis-header-text">
        <h1>\U0001f6a8 VIGIE IA - D\u00e9tection Feux de For\u00eat</h1>
        <p>Service D\u00e9partemental d'Incendie et de Secours de Loire-Atlantique \u2022 Syst\u00e8me d'aide \u00e0 la d\u00e9cision</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Upload section
st.markdown("### \U0001f4f7 Analyse d'image")
uploaded_file = st.file_uploader(
    "T\u00e9l\u00e9verser une image de cam\u00e9ra de surveillance",
    type=['jpg', 'jpeg', 'png'],
    help="Formats accept\u00e9s : JPG, JPEG, PNG - Images de cam\u00e9ras de surveillance for\u00eat"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    with st.spinner('\U0001f50d Analyse en cours par le mod\u00e8le IA...'):
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

        # GradCAM
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        img_display = (img_tensor * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
        overlay = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

    # --- Results ---
    if pred_class == 'fire':
        st.markdown(f"""
        <div class="alert-fire">
            <h2>\U0001f6a8 ALERTE - D\u00c9PART DE FEU D\u00c9TECT\u00c9</h2>
            <p>Confiance du mod\u00e8le : <strong>{confidence:.1%}</strong> \u2022 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-safe">
            <h2>\u2705 RAS - AUCUN D\u00c9PART DE FEU D\u00c9TECT\u00c9</h2>
            <p>Confiance du mod\u00e8le : <strong>{confidence:.1%}</strong> \u2022 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Metrics row
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Probabilit\u00e9 de feu</h4>
            <div class="value">{fire_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Probabilit\u00e9 sans feu</h4>
            <div class="value">{no_fire_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Niveau de confiance</h4>
            <div class="value">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Images
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### \U0001f4f7 Image source")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("#### \U0001f50d Carte d'attention GradCAM")
        st.image(overlay, use_container_width=True)
        st.caption("Les zones rouges indiquent les r\u00e9gions ayant le plus influenc\u00e9 la d\u00e9cision du mod\u00e8le.")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>\u26a0\ufe0f AVERTISSEMENT :</strong> Ce syst\u00e8me est un <strong>outil d'aide \u00e0 la d\u00e9cision</strong> uniquement.
        Toute alerte doit imp\u00e9rativement \u00eatre confirm\u00e9e par un op\u00e9rateur qualifi\u00e9 avant d\u00e9clenchement
        du protocole d'intervention. Le mod\u00e8le peut produire des faux positifs et des faux n\u00e9gatifs.
        <br><br>
        <em>Conform\u00e9ment au r\u00e8glement int\u00e9rieur du SDIS 44, aucune d\u00e9cision op\u00e9rationnelle ne doit
        \u00eatre prise sur la seule base de cette analyse automatis\u00e9e.</em>
    </div>
    """, unsafe_allow_html=True)

else:
    # Welcome state
    st.markdown("")
    col_w1, col_w2, col_w3 = st.columns([1, 2, 1])
    with col_w2:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; background: #f5f5f5; border-radius: 10px; border: 2px dashed #1a237e;">
            <p style="font-size: 3rem; margin: 0;">\U0001f4f7</p>
            <h3 style="color: #1a237e; margin: 0.5rem 0;">T\u00e9l\u00e9verser une image</h3>
            <p style="color: #616161;">Glissez-d\u00e9posez ou s\u00e9lectionnez une image de cam\u00e9ra de surveillance</p>
            <p style="color: #9e9e9e; font-size: 0.8rem;">Formats : JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    # Info cards
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.markdown("""
        <div style="background: #e8eaf6; padding: 1.5rem; border-radius: 8px; text-align: center;">
            <p style="font-size: 2rem; margin: 0;">\U0001f916</p>
            <h4 style="color: #1a237e;">IA Performante</h4>
            <p style="font-size: 0.85rem; color: #424242;">EfficientNet-B0 fine-tun\u00e9<br>AUC 0.984 sur test</p>
        </div>
        """, unsafe_allow_html=True)
    with col_i2:
        st.markdown("""
        <div style="background: #fce4ec; padding: 1.5rem; border-radius: 8px; text-align: center;">
            <p style="font-size: 2rem; margin: 0;">\u26a1</p>
            <h4 style="color: #c62828;">D\u00e9tection rapide</h4>
            <p style="font-size: 0.85rem; color: #424242;">Analyse en quelques secondes<br>R\u00e9ponse imm\u00e9diate</p>
        </div>
        """, unsafe_allow_html=True)
    with col_i3:
        st.markdown("""
        <div style="background: #e8f5e9; padding: 1.5rem; border-radius: 8px; text-align: center;">
            <p style="font-size: 2rem; margin: 0;">\U0001f50d</p>
            <h4 style="color: #1b5e20;">Explicabilit\u00e9</h4>
            <p style="font-size: 0.85rem; color: #424242;">Visualisation GradCAM<br>Zones d'attention identifi\u00e9es</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="sdis-footer">
    <strong>SDIS 44</strong> \u2022 Service D\u00e9partemental d'Incendie et de Secours de Loire-Atlantique<br>
    Vigie IA - Prototype de d\u00e9tection de feux de for\u00eat \u2022 Pour toute urgence : <strong>18</strong> ou <strong>112</strong>
</div>
""", unsafe_allow_html=True)
# -*- coding: utf-8 -*-
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(
    page_title='Détecteur de feux de forêt',
    page_icon='🔥',
    layout='wide'
)

import os

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

class GradCAMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.efficientnet = model.efficientnet

    def forward(self, x):
        return self.model(x).logits

wrapped_model = GradCAMWrapper(model)
wrapped_model.eval()

target_layer = wrapped_model.efficientnet.encoder.blocks[-1]
cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

st.sidebar.title('À propos')
st.sidebar.info(
    'Système d\'aide à la détection de feux de forêt. '
    'Toute alerte doit être confirmée par un opérateur humain avant mobilisation.'
)
st.sidebar.markdown('---')
st.sidebar.markdown('**Modèle :** EfficientNet-B0 (fine-tuné)')
st.sidebar.markdown('**Classes :** no_fire, fire')
st.sidebar.markdown('**Entrée :** Image 224x224 RGB')

with st.sidebar.expander('Comment ça marche ?'):
    st.write(
        '1. Uploadez une image de caméra de surveillance\n'
        '2. Le modèle EfficientNet-B0 analyse l\'image\n'
        '3. Il prédit s\'il y a un feu ou non avec un score de confiance\n'
        '4. Le GradCAM montre les zones qui ont influencé la décision'
    )

st.title('🔥 Détecteur de feux de forêt')
st.markdown(
    'Uploadez une image pour obtenir une prédiction '
    'avec score de confiance et visualisation GradCAM.'
)

uploaded_file = st.file_uploader(
    'Choisir une image',
    type=['jpg', 'jpeg', 'png'],
    help='Formats acceptés : JPG, JPEG, PNG'
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Image originale')
        st.image(image, use_container_width=True)

    with col2:
        st.subheader('Analyse')

        with st.spinner('Analyse en cours...'):
            img_tensor = val_transform(image)
            input_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs.logits, dim=-1)[0]

            pred_idx = probabilities.argmax().item()
            pred_class = class_names[pred_idx]
            confidence = probabilities[pred_idx].item()

            if pred_class == 'fire':
                st.error(f'🔥 **ALERTE FEU D\u00c9TECT\u00c9** - Confiance : {confidence:.1%}')
            else:
                st.success(f'\u2705 **Pas de feu d\u00e9tect\u00e9** - Confiance : {confidence:.1%}')

            st.progress(confidence)
            st.write(f'P(fire) = {probabilities[0]:.4f} | P(no_fire) = {probabilities[1]:.4f}')

            grayscale_cam = cam(input_tensor=input_tensor)[0]
            img_display = (img_tensor * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
            overlay = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

            st.subheader('GradCAM - Zones d\'attention')
            st.image(overlay, caption='Les zones rouges sont celles qui influencent le plus la décision',
                     use_container_width=True)

    st.markdown('---')
    st.error(
        '\u26a0\ufe0f **DISCLAIMER** : Système d\'aide à la détection uniquement. '
        'Toute alerte doit être confirmée par un opérateur humain qualifié '
        'avant toute mobilisation de moyens. Ce modèle peut produire des '
        'faux positifs et des faux négatifs.'
    )
else:
    st.info('👆 Uploadez une image pour commencer l\'analyse.')
