import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from safetensors.torch import safe_open

MODEL_PATH = "model/mood_model_7class_final2"

def is_model_valid():
    """Check if the model file exists and is valid"""
    model_file = os.path.join(MODEL_PATH, "model.safetensors")
    if not os.path.exists(model_file):
        return False
    
    try:
        # Try to open the safetensors file to check if it's valid
        with safe_open(model_file, framework="pt") as f:
            # If we can open it, it's valid
            return True
    except Exception as e:
        print(f"Model file is corrupted or invalid: {e}")
        return False

# Check if model is valid, if not, provide fallback
if is_model_valid():
    print("Loading model from local files...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    MODEL_LOADED = True
else:
    print("Model not available or corrupted. Setting up fallback...")
    tokenizer = None
    model = None
    device = None
    MODEL_LOADED = False

# Load label mapping
try:
    with open(f"{MODEL_PATH}/label_mapping.json", "r") as f:
        mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
except FileNotFoundError:
    id2label = {
        0: 'Bahagia', 1: 'Lelah', 2: 'Marah',
        3: 'Netral', 4: 'Sedih', 5: 'Stress', 6: 'Tenang'
    }

def predict_mood_with_analysis(text: str):
    if not MODEL_LOADED:
        return {
            "text": text,
            "predicted_mood": "Error",
            "confidence": "0.00%",
            "error": "Model not available. Git LFS file not properly downloaded on Railway.",
            "suggestion": "The model file exists locally but Railway cannot access Git LFS files properly."
        }
    
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    top_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, top_idx].item()

    return {
        "text": text,
        "predicted_mood": id2label[top_idx],
        "confidence": f"{confidence:.2%}"
    }
