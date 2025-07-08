import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Step 1: Use the correct full path to the Excel file
excel_path = r"C:\Users\AK\.cache\kagglehub\datasets\brarajit18\student-feedback-dataset\versions\1\finalDataset0.2.xlsx"
df = pd.read_excel(excel_path)

# Step 2: Rename the target column to 'label'
df.rename(columns={"teaching": "label"}, inplace=True)

# Step 3: Show basic preview and label distribution
print("🔍 Preview of dataset:")
print(df.head())

print("\n📊 Label distribution:")
print(df["label"].value_counts())

# Step 4: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Step 5: Train-validation split
dataset = dataset.train_test_split(test_size=0.2)

# Step 6: Save to disk
dataset.save_to_disk("student_feedback_dataset")

print("✅ Dataset prepared and saved to: student_feedback_dataset/")


