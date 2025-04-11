from datasets import load_dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import unicodedata

def normalize_label(label):
    label = str(label).strip().lower()
    label = unicodedata.normalize("NFD", label)
    label = label.encode("ascii", errors="ignore").decode("utf-8")
    return label

# ⚠️ Bổ sung config: ag_news
dataset = load_dataset("cestwc/text_classification", "ag_news")["train"]
df = pd.DataFrame(dataset.to_dict())
df = df.rename(columns={"content": "text"})

df = df.dropna(subset=["text", "label"])
df = df[(df["text"].str.strip() != "") & (df["label"].astype(str).str.strip() != "")]
df["label"] = df["label"].apply(normalize_label)

print("📌 Nhãn:", sorted(df["label"].unique()))
print(f"✅ Sau khi làm sạch: {len(df)} dòng")

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])

os.makedirs("data", exist_ok=True)
test_df.to_csv("data/test.csv", index=False, encoding="utf-8-sig")

num_chunks = 10
chunk_size = len(train_df) // num_chunks

for i in range(num_chunks):
    chunk_df = train_df.iloc[i*chunk_size : (i+1)*chunk_size if i < num_chunks-1 else None]
    chunk_df.to_csv(f"data/train_chunk_{i}.csv", index=False, encoding="utf-8-sig")

print("✅ Đã lưu các file vào data/")
