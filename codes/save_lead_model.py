import joblib
import os

# Dossier pour stocker les modèles
MODEL_DIR = "models_rf"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_models(model, name, space):
    """Sauvegarde le modèle avec joblib et retourne son chemin"""
    filename = os.path.join(MODEL_DIR, f"random_forest_{name}_space{space}.joblib")
    joblib.dump(model, filename)
    return filename  # Maintenant, la fonction retourne le chemin du fichier sauvegardé
    
def load_models(name, space):
    """Charge un modèle sauvegardé"""
    filename = os.path.join(MODEL_DIR, f"random_forest_{name}_space{space}.joblib")
    return joblib.load(filename) if os.path.exists(filename) else None

