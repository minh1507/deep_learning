import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import id2label, label2id, compute_metrics
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("model").to(device)
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer", use_fast=False)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return id2label[pred]

def evaluate_on_test(file_path="data/test.csv"):
    df = pd.read_csv(file_path)
    texts = df["text"].tolist()
    labels = [label2id[str(lbl)] for lbl in df["label"].tolist()]
    
    preds = []
    for text in tqdm(texts, desc="🔍 Đánh giá test set"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        preds.append(pred)

    metrics = compute_metrics(preds, labels)
    return metrics
