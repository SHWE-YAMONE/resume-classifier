import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import load_model_and_tokenizer, predict_resume_topk
import uvicorn

app = FastAPI()

class ResumeRequest(BaseModel):
    resume_text: str

model, tokenizer, label_encoder = load_model_and_tokenizer(
    model_dir="models/fold_1",
    label_path="outputs/label_classes.json"
)

@app.post("/predict/")
async def predict_transformer_endpoint(data: ResumeRequest):
    try:
        predictions = predict_resume_topk(data.resume_text, model, tokenizer, label_encoder, top_k=5)
        return {"top5_job_titles": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
