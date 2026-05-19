# Détecteur de Feux de Forêt - EfficientNet-B0

Application Streamlit de détection de feux de forêt par analyse d'images satellite/aériennes.

**Projet Sup de Vinci - Master 1 - Allaire & Loret**

## Performances
- Accuracy : 93%
- AUC : 0.984

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fichiers
- `app.py` : Application Streamlit avec GradCAM
- `model_weights/` : Poids du modèle EfficientNet-B0 fine-tuné
- `requirements.txt` : Dépendances Python
- `Track_C_Detecteur_Feux_de_Foret_Allaire_Loret.ipynb` : Notebook d'entraînement
