from fastapi import FastAPI
from pydantic import BaseModel
from predict_mood import predict_mood_with_analysis
import uvicorn
import os

app = FastAPI()

class MoodRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Mood API is running!"}

@app.post("/predict_mood")
def predict_mood(request: MoodRequest):
    return predict_mood_with_analysis(request.text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
