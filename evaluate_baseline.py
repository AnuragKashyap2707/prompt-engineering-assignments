from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load dataset and split it (fixes the error)
dataset = load_from_disk("tokenized_student_feedback_dataset")
dataset = dataset.train_test_split(test_size=0.2)

# Load base model (not fine-tuned)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted")
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="baseline_eval_logs",
    per_device_eval_batch_size=4
)

# Evaluate using Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset["test"],  # now this works
    compute_metrics=compute_metrics
)

print("🔍 Running baseline model evaluation...")
results = trainer.evaluate()
print("📉 Baseline Evaluation Results:", results)
