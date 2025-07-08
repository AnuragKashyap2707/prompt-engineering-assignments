from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
import torch

# Load dataset and split again
dataset = load_from_disk("tokenized_student_feedback_dataset")
dataset = dataset.train_test_split(test_size=0.2)

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("student_feedback_finetuned_model")
tokenizer = BertTokenizer.from_pretrained("student_feedback_finetuned_model")

# Evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# TrainingArguments (needed by Trainer even for evaluation)
training_args = TrainingArguments(
    output_dir="./eval_output",
    per_device_eval_batch_size=4,
    do_train=False,
    do_eval=True,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# Evaluate
results = trainer.evaluate()
print("📊 Evaluation Results:")
print(results)

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get raw predictions and true labels
predictions = trainer.predict(dataset["test"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

# 📋 Detailed classification report
target_names = ["Negative", "Neutral", "Positive"]
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# 📊 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

