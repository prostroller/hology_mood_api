import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

MODEL_PATH = "model/mood_model_7class_final2"

# Check if model files exist
model_files = ["pytorch_model.bin", "model.safetensors"]
model_exists = any(os.path.exists(os.path.join(MODEL_PATH, f)) for f in model_files)

if not model_exists:
    print(f"Warning: Model weights not found in {MODEL_PATH}")
    print("Available files:", os.listdir(MODEL_PATH) if os.path.exists(MODEL_PATH) else "Directory not found")
    # Set dummy variables for now
    tokenizer = None
    model = None
    device = None
else:
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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
    if model is None or tokenizer is None:
        return {
            "text": text,
            "predicted_mood": "Error",
            "confidence": "0.00%",
            "error": "Model not loaded - missing model weights file (pytorch_model.bin or model.safetensors)"
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
