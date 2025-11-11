# streamlit/predict_bert.py
import json
from pathlib import Path
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========= CONFIGURAÇÃO =========
BASE_PATH = Path(__file__).parent

MODEL_DIR = BASE_PATH / "models" / "bert_base_cb_loss_final"
CLASSES_TXT = BASE_PATH / "configs" / "classes.txt"
RETUNE_CONFIG = BASE_PATH / "configs" / "retune_config.json"
THRESHOLDS_FILE = BASE_PATH / "configs" / "thresholds.json"
# ===================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Carrega classes
CLASSES = [l.strip() for l in CLASSES_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]

# ---- Carrega thresholds
def load_thresholds():
    with open(THRESHOLDS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data["thresholds"], dtype=float)

thresholds = load_thresholds()

# ---- Carrega calibração Platt
def load_calibration():
    with open(RETUNE_CONFIG, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    cal = cfg["calibration"]
    artifacts = cal["artifacts"]
    
    # Extrai parâmetros Platt
    A = [artifacts[str(i)]["a"] for i in range(len(CLASSES))]
    B = [artifacts[str(i)]["b"] for i in range(len(CLASSES))]
    
    return "platt", None, A, B

cal_mode, T, A, B = load_calibration()

# ---- Carrega modelo
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    return tokenizer, model

# Carrega o modelo uma vez (sem cache decorator)
tokenizer, model = load_model()

# ---- Função de predição
@torch.inference_mode()
def predict(text: str, max_length: int = 128, topk_fallback: int = 3):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
    logits = model(**enc).logits  # [1, C]

    # Aplica calibração
    if cal_mode == "platt" and A is not None and B is not None:
        A_t = torch.tensor(np.array(A).reshape(1, -1), dtype=logits.dtype, device=logits.device)
        B_t = torch.tensor(np.array(B).reshape(1, -1), dtype=logits.dtype, device=logits.device)
        calibrated_logits = logits * A_t + B_t
        probs = torch.sigmoid(calibrated_logits)
    else:
        probs = torch.sigmoid(logits)

    probs = probs.squeeze(0).detach().cpu().numpy()  # [C]
    pred = (probs >= thresholds).astype(int)
    picked = [(CLASSES[i], float(probs[i])) for i in range(len(CLASSES)) if pred[i] == 1]

    if not picked and topk_fallback > 0:
        idx = np.argsort(probs)[-topk_fallback:][::-1]
        picked = [(CLASSES[i], float(probs[i])) for i in idx]
    
    return picked, probs