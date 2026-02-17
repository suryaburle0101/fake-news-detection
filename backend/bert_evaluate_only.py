import pandas as pd
import torch
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# =========================
# 1. Load Dataset
# =========================

fake_df = pd.read_csv("D:/projects/detection/Fake.csv")
true_df = pd.read_csv("D:/projects/detection/True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)

df["text"] = df["title"] + " " + df["text"]
df = df[["text", "label"]]

df.drop_duplicates(inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total samples:", len(df))

# =========================
# 2. Train/Test Split
# =========================

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

test_dataset = Dataset.from_dict({
    "text": test_texts.tolist(),
    "label": test_labels.tolist()
})

# =========================
# 3. Load Saved Model
# =========================

model_path = "fake_news_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================
# 4. Tokenize Test Data
# =========================

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# =========================
# 5. Predict
# =========================

pred_labels = []
true_labels = []

with torch.no_grad():
    for batch in test_dataset:
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        pred_labels.append(preds.item())
        true_labels.append(batch["label"].item())

# =========================
# 6. Metrics
# =========================

cm = confusion_matrix(true_labels, pred_labels)
acc = accuracy_score(true_labels, pred_labels)

print("\nBERT Accuracy:", acc)
print("\nBERT Confusion Matrix:")
print(cm)
