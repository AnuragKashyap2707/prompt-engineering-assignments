from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load dataset and model
dataset = load_from_disk("tokenized_student_feedback_dataset")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted")
    }

# Define configurations
configs = [
    {"lr": 2e-5, "batch": 8, "epochs": 3},
    {"lr": 3e-5, "batch": 16, "epochs": 4},
    {"lr": 5e-5, "batch": 8, "epochs": 5}
]

results = []

# Loop through each config
for i, config in enumerate(configs):
    print(f"\nRunning config {i+1}: {config}")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir=f"./tmp_trainer/run_{i+1}",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch"],
        num_train_epochs=config["epochs"],
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    eval_result.update(config)
    results.append(eval_result)

# Print summary
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("hyperparameter_results.csv", index=False)
print("\n=== Hyperparameter Tuning Results ===\n")
print(df)

