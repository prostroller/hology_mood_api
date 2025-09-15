import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from safetensors.torch import safe_open

MODEL_PATH = "model/mood_model_7class_final2"
# Use a smaller, more reliable fallback model
FALLBACK_MODEL = "j-hartmann/emotion-english-distilroberta-base"

def is_model_valid():
    """Check if the model file exists and is valid"""
    try:
        model_file = os.path.join(MODEL_PATH, "model.safetensors")
        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            return False
        
        # Check file size (should be > 100MB for a real model)
        file_size = os.path.getsize(model_file)
        print(f"Model file size: {file_size / (1024*1024):.1f} MB")
        
        if file_size < 1024:  # Less than 1KB suggests it's a Git LFS pointer
            print("Model file appears to be a Git LFS pointer file")
            return False
        
        # Try to open the safetensors file to check if it's valid
        with safe_open(model_file, framework="pt") as f:
            # If we can open it, it's valid
            print("Model file validation successful")
            return True
    except Exception as e:
        print(f"Model file validation failed: {e}")
        return False

def load_fallback_model():
    """Load a simpler pre-trained emotion model from Hugging Face as fallback"""
    print("Loading fallback emotion model from Hugging Face...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print("Fallback model loaded successfully")
        
        # Map the fallback model labels to Indonesian
        fallback_mapping = {
            0: 'Bahagia',      # anger
            1: 'Lelah',     # disgust -> neutral
            2: 'Marah',     # fear -> neutral  
            3: 'Netral',    # joy
            4: 'Sedih',     # neutral
            5: 'Stress',      # sadness
            6: 'Tenang'      # surprise -> neutral
        }
        
        return tokenizer, model, device, fallback_mapping, True
    except Exception as e:
        print(f"Failed to load fallback model: {e}")
        return None, None, None, None, False

# Initialize variables
tokenizer = None
model = None
device = None
MODEL_LOADED = False
USING_FALLBACK = False
id2label = {
    0: 'Bahagia', 1: 'Lelah', 2: 'Marah',
    3: 'Netral', 4: 'Sedih', 5: 'Stress', 6: 'Tenang'
}

try:
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
            pass  # Use default mapping
            
        print("Custom model loaded successfully")
    else:
        print("Local model not available. Trying fallback...")
        tokenizer, model, device, id2label, MODEL_LOADED = load_fallback_model()
        USING_FALLBACK = MODEL_LOADED
        
except Exception as e:
    print(f"Error during model initialization: {e}")
    MODEL_LOADED = False
    USING_FALLBACK = False

def predict_mood_with_analysis(text: str):
    try:
        if not MODEL_LOADED:
            return {
                "text": text,
                "predicted_mood": "Error",
                "confidence": "0.00%",
                "error": "No model available. Both local and fallback models failed to load.",
                "suggestion": "Model loading failed on server startup."
            }
        
        # Ensure text is not empty
        if not text or not text.strip():
            return {
                "text": text,
                "predicted_mood": "Netral",
                "confidence": "100.00%",
                "model_info": "Default response for empty text"
            }
        
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        top_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, top_idx].item()

        result = {
            "text": text,
            "predicted_mood": id2label.get(top_idx, "Netral"),
            "confidence": f"{confidence:.2%}"
        }
        
        # Add info about which model is being used
        if USING_FALLBACK:
            result["model_info"] = "Using fallback emotion model (Hugging Face)"
        else:
            result["model_info"] = "Using custom trained model"
        
        return result
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {
            "text": text,
            "predicted_mood": "Error",
            "confidence": "0.00%",
            "error": f"Prediction failed: {str(e)}",
            "suggestion": "Please try again or contact support."
        }
