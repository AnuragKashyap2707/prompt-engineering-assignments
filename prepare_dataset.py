import pandas as pd
from datasets import Dataset

# Step 1: Load your Excel file
df = pd.read_excel(r"C:\Users\AK\.cache\kagglehub\datasets\brarajit18\student-feedback-dataset\versions\1\finalDataset0.2.xlsx")

# Step 2: Rename columns to match expected HuggingFace format
df = df.rename(columns={"teaching": "text", "label": "label"})  # Update this if column names are different

# Step 3: Check value counts
print("✅ Label counts:\n", df["label"].value_counts())

# Step 4: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Step 5: Save to disk for future use
dataset.save_to_disk("student_feedback_dataset")
print("📦 Dataset saved to: student_feedback_dataset")
