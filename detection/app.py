from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("./results/saved_model")
model = AutoModelForSequenceClassification.from_pretrained("./results/saved_model")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    label = "Real" if prediction.item() == 1 else "Fake"

    return jsonify({
        "prediction": label,
        "confidence": f"{confidence.item()*100:.2f}%"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
