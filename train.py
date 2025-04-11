import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from utils import tokenizer, label2id  # Đảm bảo utils.py có đúng 2 biến này

class NewsDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=256,  # ✅ Hạn chế độ dài tránh lỗi
        )
        self.encodings.pop("token_type_ids", None)  # ✅ Loại bỏ nếu không dùng
        self.labels = [label2id[lbl] for lbl in df["label"].tolist()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx]),
        }

# Thiết bị tính toán
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model từ checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=len(label2id)
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Training theo từng chunk
for i in range(10):
    print(f"\n🔁 Training chunk {i}")
    df = pd.read_csv(f"data/train_chunk_{i}.csv", encoding="utf-8-sig")
    dataset = NewsDataset(df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader),
    )

    model.train()
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# Lưu model và tokenizer
os.makedirs("model/tokenizer", exist_ok=True)
model.save_pretrained("model")
tokenizer.save_pretrained("model/tokenizer")
print("✅ Đã lưu model vào thư mục `model/`")
