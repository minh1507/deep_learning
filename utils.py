from transformers import AutoTokenizer
import pandas as pd
import os

# Duyệt qua tất cả train_chunk để lấy nhãn duy nhất
labels = set()
for i in range(10):
    df = pd.read_csv(f"data/train_chunk_{i}.csv")
    labels.update(df["label"].unique())

labels = sorted(list(labels))
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
