import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from transformers import TrainerCallback

def main_model(file_name, ext):
    # Load the dataset
    df = pd.read_excel(f"/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Data/Datasets/{file_name}.{ext}")

    # Select the text and label columns
    X = df['review'].values  # Numpy array is often more convenient for sklearn operations
    y = df['label'].values

    X_train_CV, X_test_full, y_train_CV, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Encode the labels to a numeric format
    label_encoder = LabelEncoder()
    y_train_CV_encoded = label_encoder.fit_transform(y_train_CV)
    y_test_full_encoded = label_encoder.transform(y_test_full)

    # Initialize the tokenizer for RoBERTa
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # Tokenization function
    def tokenize_function(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    loss_logging_callback = LossLoggingCallback()

    # Stratified K-Fold Cross-Validation
    n_splits = 2
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Variables to accumulate scores
    best_accuracy = 0
    best_model = None
    accuracy_scores = []
    metrics_df = pd.DataFrame()
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_CV, y_train_CV_encoded)):
        print(f"Fold {fold+1}/{n_splits}")

        # Split the data
        X_train, X_val = X_train_CV[train_index], X_train_CV[val_index]
        y_train, y_val = y_train_CV_encoded[train_index], y_train_CV_encoded[val_index]

        # Tokenize the data
        train_encodings = tokenize_function(X_train.tolist())
        val_encodings = tokenize_function(X_val.tolist())

        # Create dataset objects
        train_dataset = ReviewDataset(train_encodings, y_train)
        val_dataset = ReviewDataset(val_encodings, y_val)

        # Initialize the model for each fold
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))

        # Define training arguments for each fold, adjust hyperparameters as needed
        training_args = TrainingArguments(
            output_dir=f"/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Models/RoBERTa/Output/{file_name}_results",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Models/RoBERTa/Output/{file_name}_logs",
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            max_grad_norm=1.0,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_strategy="epoch",
            save_total_limit=2,
            lr_scheduler_type='linear'
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda p: {"accuracy": accuracy_score(p.predictions.argmax(-1), p.label_ids)},
            callbacks=[loss_logging_callback]
        )

        # Train
        trainer.train()

        loss_logging_callback.save_logs_to_excel(f"/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Models/RoBERTa/Output/{file_name}_loss_results.xlsx")

        # Evaluate
        results = trainer.evaluate()
        accuracy_scores.append(results['eval_accuracy'])

        if results['eval_accuracy'] > best_accuracy:
            best_accuracy = results['eval_accuracy']
            best_model = model  # Assign the best model

        # Get predictions and true labels
        predictions = trainer.predict(val_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = y_val

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        label_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

        # Calculate precision, recall, and F1-score
        report_dict = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0, target_names=label_names)
        # avg_metrics = report_dict['weighted avg']  # Use 'macro avg' or 'weighted avg' based on your preference

        # Append the metrics for this fold to the DataFrame
        metrics_df = metrics_df.append({
            ('Fold', ''): fold + 1,
            ('Accuracy', ''): accuracy,
            ('Bug Report', 'P'): report_dict['bug report']['precision'],
            ('Bug Report', 'R'): report_dict['bug report']['recall'],
            ('Bug Report', 'F1'): report_dict['bug report']['f1-score'],
            ('Feature Request', 'P'): report_dict['feature request']['precision'],
            ('Feature Request', 'R'): report_dict['feature request']['recall'],
            ('Feature Request', 'F1'): report_dict['feature request']['f1-score'],
            ('Rating', 'P'): report_dict['rating']['precision'],
            ('Rating', 'R'): report_dict['rating']['recall'],
            ('Rating', 'F1'): report_dict['rating']['f1-score'],
            ('User Experience', 'P'): report_dict['user experience']['precision'],
            ('User Experience', 'R'): report_dict['user experience']['recall'],
            ('User Experience', 'F1'): report_dict['user experience']['f1-score']
        }, ignore_index=True)

    # Save the DataFrame to a CSV file after completing all folds
    metrics_df.columns = pd.MultiIndex.from_tuples([(c,) if isinstance(c, str) else c for c in metrics_df.columns])
    metrics_df.to_excel(f"/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Models/RoBERTa/Output/{file_name}_metrics_results.xlsx", index=True)

    test_encodings = tokenizer(X_test_full.tolist())
    test_dataset = ReviewDataset(test_encodings, y_test_full_encoded)

    trainer_full = Trainer(
        model=best_model
    )
    
    predictions_full = trainer_full.predict(test_dataset)
    pred_labels_full = np.argmax(predictions.predictions, axis=-1)
    true_labels_full = y_test_full_encoded

    accuracy_full = accuracy_score(true_labels_full, pred_labels_full)
    label_names_full = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

    # Calculate precision, recall, and F1-score
    report_dict_full = classification_report(true_labels_full, pred_labels_full, output_dict=True, zero_division=0, target_names=label_names_full)

    full_metrics_df = pd.DataFrame()

    full_metrics_df = full_metrics_df.append({
            ('Accuracy', ''): accuracy_full,
            ('Bug Report', 'P'): report_dict_full['bug report']['precision'],
            ('Bug Report', 'R'): report_dict_full['bug report']['recall'],
            ('Bug Report', 'F1'): report_dict_full['bug report']['f1-score'],
            ('Feature Request', 'P'): report_dict_full['feature request']['precision'],
            ('Feature Request', 'R'): report_dict_full['feature request']['recall'],
            ('Feature Request', 'F1'): report_dict_full['feature request']['f1-score'],
            ('Rating', 'P'): report_dict_full['rating']['precision'],
            ('Rating', 'R'): report_dict_full['rating']['recall'],
            ('Rating', 'F1'): report_dict_full['rating']['f1-score'],
            ('User Experience', 'P'): report_dict_full['user experience']['precision'],
            ('User Experience', 'R'): report_dict_full['user experience']['recall'],
            ('User Experience', 'F1'): report_dict_full['user experience']['f1-score']
        }, ignore_index=True)

    full_metrics_df.columns = pd.MultiIndex.from_tuples([(c,) if isinstance(c, str) else c for c in full_metrics_df.columns])
    full_metrics_df.to_excel(f"/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Models/RoBERTa/Output/{file_name}_metrics_results_full.xlsx", index=True)

# Custom dataset class
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

class LossLoggingCallback(TrainerCallback):
    """A custom callback to log training and validation loss."""
    def __init__(self):
        super().__init__()
        self.log_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # This method captures both training and evaluation logs, so it's more general than on_epoch_end
        if logs is not None:
            # Capture both training and evaluation steps
            if 'loss' in logs:  # Indicates a training step
                self.log_history.append({
                    'epoch': state.epoch,
                    'training_loss': logs.get('loss'),
                })
            elif 'eval_loss' in logs:  # Indicates an evaluation step
                # Make sure to capture the last training loss as well
                last_training_loss = self.log_history[-1]['training_loss'] if self.log_history else None
                self.log_history.append({
                    'epoch': state.epoch,
                    'training_loss': last_training_loss,  # Include last known training loss for reference
                    'validation_loss': logs.get('eval_loss'),
                    'eval_runtime':logs.get('eval_runtime')
                })

    def save_logs_to_excel(self, file_name):
        """Save the recorded logs to a CSV file."""
        pd.DataFrame(self.log_history).to_excel(file_name, index=False)

main_model("dataset_balanced_reduced_100_per_class", "xlsx")