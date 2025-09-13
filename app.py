from fastapi import FastAPI
from pydantic import BaseModel
from predict_mood import predict_mood_with_analysis

app = FastAPI()

class MoodRequest(BaseModel):
    text: str

@app.post("/predict_mood")
def predict_mood(request: MoodRequest):
    return predict_mood_with_analysis(request.text)
