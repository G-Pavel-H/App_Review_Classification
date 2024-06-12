import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# Assuming df is your DataFrame with 'review' column for texts and 'label1', 'label2', 'label3', 'label4' for labels
# For example: df = pd.read_csv('path/to/your/dataset.csv')

class MultiLabelDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)  # Ensure float32 for BCEWithLogitsLoss
        return item

    def __len__(self):
        return len(self.labels)

def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=128)

# Adaptation starts here
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Example of preparing the dataset
df = pd.read_csv("/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Data/Datasets/Multi-label/dataset_gpt_multi_label_100.csv")
labels = df[['bug_report', 'user_experience', 'rating', 'feature_request']].values

X_train, X_val, y_train, y_val = train_test_split(df['review'], labels, test_size=0.1)

train_encodings = tokenize_function(X_train.tolist())
val_encodings = tokenize_function(X_val.tolist())

train_dataset = MultiLabelDataset(train_encodings, y_train)
val_dataset = MultiLabelDataset(val_encodings, y_val)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4, problem_type="multi_label_classification")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="steps",
    logging_steps=50,
)

def compute_metrics(p):
    predictions, labels = p
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    threshold = 0.5
    predictions = (predictions > threshold).astype(int)
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    return {'precision': precision, 'recall': recall, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Make predictions on the validation set
predictions = trainer.predict(val_dataset)

# Apply sigmoid to the predictions because we're dealing with multi-label classification
# This transforms logits to probabilities
pred_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()

# Choose a threshold to convert probabilities to binary predictions
# This threshold can be adjusted based on your specific needs or optimized based on a validation set
threshold = 0.5
binary_predictions = (pred_probs > threshold).astype(int)

# True labels
true_labels = predictions.label_ids

# Compute metrics
precision = precision_score(true_labels, binary_predictions, average='micro')
recall = recall_score(true_labels, binary_predictions, average='micro')
f1 = f1_score(true_labels, binary_predictions, average='micro')
accuracy = accuracy_score(true_labels, binary_predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Accuracy: {accuracy}")