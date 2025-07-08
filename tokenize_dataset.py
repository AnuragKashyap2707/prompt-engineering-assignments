from datasets import load_from_disk
from transformers import BertTokenizer
from datasets import DatasetDict

# Load dataset saved in previous step
dataset = load_from_disk("student_feedback_dataset")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Keep only necessary columns
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# Save tokenized dataset
tokenized_dataset.save_to_disk("tokenized_student_feedback_dataset")

print("✅ Tokenization complete and saved to: tokenized_student_feedback_dataset")


