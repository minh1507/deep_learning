from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import os
import json

# Load lại model và tokenizer
model = AutoModelForSequenceClassification.from_pretrained("model").eval()
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")

# Load lại label
with open("utils.py", encoding="utf-8") as f:
    exec(f.read())

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        pred_label = id2label[pred_id]
        confidence = probs[0][pred_id].item()

    return jsonify({
        "label": pred_label,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
