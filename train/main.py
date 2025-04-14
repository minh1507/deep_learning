import os
import argparse
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def load_data(chunk_index=0, limit=None):
    path = f"./data/train_chunk_{chunk_index}.csv"
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return df

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")),
    ])

def train_and_log(df, chunk_index, run_name_suffix="", experiment_name="TextClassification"):
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"chunk_{chunk_index}{run_name_suffix}") as run:
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)

        mlflow.log_param("chunk_index", chunk_index)
        mlflow.log_param("mode", run_name_suffix.strip("_"))
        mlflow.log_metric("accuracy", acc)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"f1_{label}", metrics.get("f1-score", 0))

        model_path = "models/latest"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)
        mlflow.sklearn.save_model(pipeline, os.path.join(model_path, "model.pkl"))

        print(f"✅ Đã lưu model mới nhất vào {model_path}")
        print(f"📊 MLflow Run: {run.info.run_id}")
        print("👉 Mở giao diện MLflow với lệnh: `mlflow ui`")

def main(args):
    if args.mode == "test":
        print("🔍 Chế độ test nhanh với 500 dòng đầu chunk_0")
        df = load_data(0, limit=500)
        train_and_log(df, chunk_index=0, run_name_suffix="_test")

    elif args.mode == "chunk":
        print(f"🧩 Train chunk {args.chunk}")
        df = load_data(args.chunk)
        train_and_log(df, chunk_index=args.chunk)

    elif args.mode == "full":
        print("🚀 Train toàn bộ chunk từ 1 đến 9")
        for i in range(1, 10):
            print(f"\n▶️ Training chunk {i}")
            df = load_data(i)
            train_and_log(df, chunk_index=i, run_name_suffix="_full")

    else:
        print("❌ Lựa chọn mode không hợp lệ. Dùng: test, chunk, hoặc full.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="Chế độ chạy: test, chunk, full")
    parser.add_argument("--chunk", type=int, help="Số chunk (dùng cho chế độ 'chunk')")
    args = parser.parse_args()

    main(args)
