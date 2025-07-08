from transformers import BertForSequenceClassification, BertTokenizer, Trainer
from datasets import Dataset
import pandas as pd
import torch
import numpy as np

# ✅ Step 1: Load raw Excel data
df = pd.read_excel(r"C:\Users\AK\.cache\kagglehub\datasets\brarajit18\student-feedback-dataset\versions\1\finalDataset0.2.xlsx")


# ✅ Step 2: Only keep relevant columns
df = df[["teaching", "label"]]
df = df.dropna()  # ensure no NaNs

# ✅ Step 3: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# ✅ Step 4: Split into train/test for analysis
dataset = dataset.train_test_split(test_size=0.2)
test_dataset = dataset["test"]

# ✅ Step 5: Load fine-tuned model and tokenizer
model_dir = "student_feedback_finetuned_model"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# ✅ Step 6: Tokenize "teaching" column
def tokenize(batch):
    return tokenizer(batch["teaching"], truncation=True, padding=True, max_length=256)

encoded_test_dataset = test_dataset.map(tokenize, batched=True)

# ✅ Step 7: Setup Trainer
trainer = Trainer(model=model)

# ✅ Step 8: Predict
predictions = trainer.predict(encoded_test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids
teaching_texts = test_dataset["teaching"]

# ✅ Step 9: Label mapping
id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}

# ✅ Step 10: Collect misclassified examples
misclassified = []
for i in range(len(teaching_texts)):
    if pred_labels[i] != true_labels[i]:
        misclassified.append({
            "teaching": teaching_texts[i],
            "true_label": id2label[true_labels[i]],
            "predicted_label": id2label[pred_labels[i]]
        })

# ✅ Step 11: Save to CSV
df_misclassified = pd.DataFrame(misclassified)
df_misclassified.to_csv("misclassified_examples.csv", index=False)

print(f"✅ Saved {len(df_misclassified)} misclassified examples to misclassified_examples.csv")



