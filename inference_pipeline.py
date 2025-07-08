from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# Load fine-tuned model and tokenizer
model_dir = "student_feedback_finetuned_model"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping
id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}

# Inference function
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class_id].item()

    return id2label[predicted_class_id], confidence

# Example predictions
if __name__ == "__main__":
    examples = [
        "The teaching is terrible and unfair.",       # Expect: Negative
        "Library is okay but can improve.",           # Expect: Neutral
        "The course was good."   # Expect: Positive
    ]

    for text in examples:
        sentiment, confidence = predict_sentiment(text)
        print(f"📝 Input: {text}")
        print(f"🎯 Predicted Sentiment: {sentiment} ({confidence*100:.1f}% confidence)\n")

