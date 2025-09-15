from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

class MoodRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Mood API is running!"}

@app.get("/health")
def health_check():
    try:
        from predict_mood import MODEL_LOADED, USING_FALLBACK
        return {
            "status": "healthy",
            "model_loaded": MODEL_LOADED,
            "using_fallback": USING_FALLBACK,
            "message": "API is operational"
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e),
            "message": "Model import failed"
        }

@app.post("/predict_mood")
def predict_mood(request: MoodRequest):
    try:
        from predict_mood import predict_mood_with_analysis
        return predict_mood_with_analysis(request.text)
    except Exception as e:
        return {
            "text": request.text,
            "predicted_mood": "Error",
            "confidence": "0.00%",
            "error": f"Import error: {str(e)}",
            "suggestion": "Model failed to import"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
