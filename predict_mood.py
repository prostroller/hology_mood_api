import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from safetensors.torch import safe_open

MODEL_PATH = "model/mood_model_7class_final2"
# Fallback: use a public model when local model fails
FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"

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

def load_fallback_model():
    """Load a pre-trained emotion model from Hugging Face as fallback"""
    print("Loading fallback emotion model from Hugging Face...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Map the fallback model labels to your custom labels
        fallback_mapping = {
            0: 'Marah',      # anger
            1: 'Bahagia',    # joy  
            2: 'Netral',     # optimism
            3: 'Sedih'       # sadness
        }
        
        return tokenizer, model, device, fallback_mapping, True
    except Exception as e:
        print(f"Failed to load fallback model: {e}")
        return None, None, None, None, False

# Check if model is valid, if not, use fallback
if is_model_valid():
    print("Loading model from local files...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    MODEL_LOADED = True
    USING_FALLBACK = False
    
    # Load your custom label mapping
    try:
        with open(f"{MODEL_PATH}/label_mapping.json", "r") as f:
            mapping = json.load(f)
            id2label = {int(k): v for k, v in mapping["id2label"].items()}
    except FileNotFoundError:
        id2label = {
            0: 'Bahagia', 1: 'Lelah', 2: 'Marah',
            3: 'Netral', 4: 'Sedih', 5: 'Stress', 6: 'Tenang'
        }
else:
    print("Local model not available. Trying fallback...")
    tokenizer, model, device, id2label, MODEL_LOADED = load_fallback_model()
    USING_FALLBACK = MODEL_LOADED

def predict_mood_with_analysis(text: str):
    if not MODEL_LOADED:
        return {
            "text": text,
            "predicted_mood": "Error",
            "confidence": "0.00%",
            "error": "No model available. Both local and fallback models failed to load.",
            "suggestion": "Please check your internet connection or try again later."
        }
    
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    top_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, top_idx].item()

    result = {
        "text": text,
        "predicted_mood": id2label[top_idx],
        "confidence": f"{confidence:.2%}"
    }
    
    # Add info about which model is being used
    if USING_FALLBACK:
        result["model_info"] = "Using fallback emotion model (Hugging Face)"
    else:
        result["model_info"] = "Using custom trained model"
    
    return result
