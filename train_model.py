from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from datasets import load_from_disk
import torch
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Load tokenized dataset
dataset = load_from_disk("tokenized_student_feedback_dataset")

# Step 2: Define label mappings
label2id = {
    "Neutral": 0,
    "Positive": 1,
    "Negative": 2
}
id2label = {v: k for k, v in label2id.items()}

# Step 3: Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# Step 4: Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Using device:", device)
model.to(device)

# Step 5: Split dataset
dataset = dataset.train_test_split(test_size=0.2)

# Step 6: Define basic training arguments
training_args = TrainingArguments(
    output_dir="student_feedback_finetuned_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs"
)

# Step 7: Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Step 8: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# Step 9: Train and Save
print("🚀 Starting training...")
trainer.train()

print("💾 Saving model and tokenizer...")
trainer.save_model("student_feedback_finetuned_model")
tokenizer.save_pretrained("student_feedback_finetuned_model")

print("✅ Model training and saving complete.")
