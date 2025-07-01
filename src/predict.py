import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import load_label_encoder

def load_model_and_tokenizer(model_dir, label_path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    label_encoder = load_label_encoder(label_path)
    return model, tokenizer, label_encoder

def predict_resume_topk(text, model, tokenizer, label_encoder, top_k=5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    top_indices = probs.argsort()[-top_k:][::-1]
    return [{"job_title": label_encoder.classes_[i], "percentage": float(probs[i] * 100)} for i in top_indices]
