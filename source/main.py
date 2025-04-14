from flask import Flask, request, jsonify
import mlflow.sklearn
import os

app = Flask(__name__)

model_path = os.path.join(".", "models", "latest", "model.pkl")
model = mlflow.sklearn.load_model(model_path)

# Mapping từ index → tên nhãn
label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Tech"
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        texts = data.get("texts", [])

        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "Vui lòng gửi 1 list các văn bản trong trường 'texts'"}), 400

        predictions = model.predict(texts)
        label_names = [label_map.get(p, str(p)) for p in predictions]

        return jsonify({
            "predictions": label_names
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🚀 Text Classification API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
