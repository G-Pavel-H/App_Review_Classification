import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset from an Excel file
df = pd.read_excel('/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Models/RoBERTa/dataset_balanced_reduced_4000.xlsx') 

# Select the text (features) and label columns
X = df['review']  # Replace 'review' with your text column name
y = df['label']   # Replace 'label' with your label column name

# Split the dataset into a training set and a validation set
# Use stratify to ensure each class is split according to the same ratio
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Encode the labels to a numeric format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Initialize the tokenizer for RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Define a tokenization function that will be used to process the text data
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# Tokenize the training and validation data
train_encodings = tokenize_function(X_train.tolist())
val_encodings = tokenize_function(X_val.tolist())

# Create a custom dataset class for handling the tokenized data
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects for training and validation
train_dataset = ReviewDataset(train_encodings, y_train_encoded)
val_dataset = ReviewDataset(val_encodings, y_val_encoded)

# Load the pre-trained RoBERTa model for sequence classification with the number of labels
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))

# Set the training arguments (hyperparameters for the training process)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    learning_rate=3e-5,
    max_grad_norm=1.0,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_strategy="epoch",
    save_total_limit=2,
    lr_scheduler_type='linear'
)


# Initialize the Trainer with the model, training arguments, and dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.predictions.argmax(-1), p.label_ids)}
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
results = trainer.evaluate()

# Get predictions from the model on the validation set
predictions = trainer.predict(val_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)

# Calculate the accuracy of the model
accuracy = (pred_labels == y_val_encoded).mean()
print(f"Accuracy: {accuracy}")

# Calculate the precision, recall, and F1 score
report = classification_report(y_val_encoded, pred_labels, target_names=label_encoder.classes_)

print(report)

# Save the model
# model.save_pretrained("path_to_save_model")  
