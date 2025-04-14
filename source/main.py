from flask import Flask, request, jsonify
import mlflow.pytorch
import torch
import torch.nn.functional as F
import re
import os
import pickle
from torch.nn.utils.rnn import pad_sequence

# ----- CẤU HÌNH -----
app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("models", "latest")
MAX_LEN = 100

# ----- LOAD MODEL -----
model = mlflow.pytorch.load_model(MODEL_PATH)
model.eval()

# ----- LOAD VOCAB + LABEL ENCODER -----
with open(os.path.join(MODEL_PATH, "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

with open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# ----- TOKENIZER -----
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def encode_text(text):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]

# ----- API ENDPOINTS -----
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        texts = data.get("texts", [])

        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "Vui lòng gửi 1 list các văn bản trong trường 'texts'"}), 400

        encoded = [torch.tensor(encode_text(t)) for t in texts]
        padded = pad_sequence(encoded, batch_first=True, padding_value=0)[:, :MAX_LEN]

        with torch.no_grad():
            outputs = model(padded.to(DEVICE))
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().numpy()

        labels = label_encoder.inverse_transform(preds)
        return jsonify({"predictions": labels.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🚀 Deep Learning Text Classification API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
