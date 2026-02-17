from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS

import torch

app = Flask(__name__)

model_path = "fake_news_model"

app = Flask(__name__)
CORS(app)


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    label = "Fake" if predicted_class == 0 else "True"
    return label, round(confidence * 100, 2)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    label, confidence = predict_news(text)

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
