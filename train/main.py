import os
import argparse
import shutil
import pandas as pd
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re
import pickle
# ----- CONFIG -----
MAX_LEN = 100
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- TOKENIZER & VOCAB -----
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode_text(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]

# ----- DATASET -----
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_encoder):
        self.texts = [torch.tensor(encode_text(t, vocab)) for t in texts]
        self.labels = torch.tensor(label_encoder.transform(labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded[:, :MAX_LEN], torch.tensor(labels)

# ----- MODEL -----
class TextClassifierGRU(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.gru = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# ----- TRAINING -----
def train_and_log(df, chunk_index, run_name_suffix="", experiment_name="TextClassification"):
    df = df.dropna(subset=["text", "label"])
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vocab = build_vocab(X_train)
    train_ds = TextDataset(X_train.tolist(), y_train.tolist(), vocab, label_encoder)
    val_ds = TextDataset(X_val.tolist(), y_val.tolist(), vocab, label_encoder)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = TextClassifierGRU(len(vocab), len(label_encoder.classes_)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"chunk_{chunk_index}{run_name_suffix}") as run:
        for epoch in range(EPOCHS):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                logits = model(x_batch)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred)
                truths.extend(y_batch.numpy())

        acc = accuracy_score(truths, preds)
        report = classification_report(truths, preds, output_dict=True)

        # Logging
        mlflow.log_param("chunk_index", chunk_index)
        mlflow.log_param("mode", run_name_suffix.strip("_"))
        mlflow.log_metric("accuracy", acc)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"f1_{label}", metrics.get("f1-score", 0))

        # Save model
        model_path = "models/latest"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)
        mlflow.pytorch.save_model(model, model_path)

        print(f"✅ Model saved to {model_path}")
        print(f"📊 MLflow Run ID: {run.info.run_id}")

        with open(os.path.join(model_path, "vocab.pkl"), "wb") as f:
            pickle.dump(vocab, f)

        with open(os.path.join(model_path, "label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)

# ----- LOAD & MAIN -----
def load_data(chunk_index=0, limit=None):
    path = f"./data/train_chunk_{chunk_index}.csv"
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return df

def main(args):
    if args.mode == "test":
        print("🔍 Test mode: 500 rows from chunk_0")
        df = load_data(0, limit=500)
        train_and_log(df, chunk_index=0, run_name_suffix="_test")

    elif args.mode == "chunk":
        print(f"🧩 Training chunk {args.chunk}")
        df = load_data(args.chunk)
        train_and_log(df, chunk_index=args.chunk)

    elif args.mode == "full":
        print("🚀 Training all chunks from 1 to 9")
        for i in range(1, 10):
            print(f"\n▶️ Training chunk {i}")
            df = load_data(i)
            train_and_log(df, chunk_index=i, run_name_suffix="_full")

    else:
        print("❌ Invalid mode. Use: test, chunk, or full.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="Mode: test, chunk, full")
    parser.add_argument("--chunk", type=int, help="Chunk number (for 'chunk' mode)")
    args = parser.parse_args()

    main(args)
