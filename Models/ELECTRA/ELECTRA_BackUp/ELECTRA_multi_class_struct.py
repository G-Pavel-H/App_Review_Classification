import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import ElectraTokenizer, ElectraForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from transformers import TrainerCallback
import os
import shutil

def main_model(file_name, ext, type):

    path_type = "Balanced" if type == 1 else "Unbalanced"
    path_to_project = "/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject"
 
    df = pd.read_excel(f"{path_to_project}/Data/Datasets/{path_type}/{file_name}.{ext}")

    results_dir = f"{path_to_project}/Models/ELECTRA/Output/{path_type}/{file_name}"
    dump_dir = results_dir+"/Dump"

    os.mkdir(results_dir)
    os.mkdir(dump_dir)

    # Select the text and label columns
    X = df['review'].values  
    y = df['label'].values

    X_train_CV, X_test_full, y_train_CV, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Encode the labels to a numeric format
    label_encoder = LabelEncoder()
    y_train_CV_encoded = label_encoder.fit_transform(y_train_CV)
    y_test_full_encoded = label_encoder.transform(y_test_full)

    # Initialize the tokenizer for RoBERTa
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    # Tokenization function
    def tokenize_function(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    loss_logging_callback = LossLoggingCallback()

    # Stratified K-Fold Cross-Validation
    n_splits = 5
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
        train_encodings = tokenize_function([remove_emojis(str(item)) if item is not None else "" for item in X_train.tolist()])
        val_encodings = tokenize_function([remove_emojis(str(item)) if item is not None else "" for item in X_val.tolist()])

        # Create dataset objects
        train_dataset = ReviewDataset(train_encodings, y_train)
        val_dataset = ReviewDataset(val_encodings, y_val)

        # Initialize the model for each fold
        model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator', num_labels=len(label_encoder.classes_))

        # Define training arguments for each fold, adjust hyperparameters as needed
        training_args = TrainingArguments(
            output_dir=f"{dump_dir}/res",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{dump_dir}/logs",
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=5e-5,
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

        loss_logging_callback.save_logs_to_excel(f"{results_dir}/{file_name}_fold_loss.xlsx")

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
    metrics_df.to_excel(f"{results_dir}/{file_name}_fold_metrics.xlsx", index=True)

    # Evaluate the best model on the test set
    test_encodings = tokenize_function([remove_emojis(str(item)) if item is not None else "" for item in X_test_full.tolist()])
    test_dataset = ReviewDataset(test_encodings, y_test_full_encoded)
    test_trainer = Trainer(model=best_model)
    test_results = test_trainer.predict(test_dataset)
    test_predictions = np.argmax(test_results.predictions, axis=-1)
    test_accuracy = accuracy_score(y_test_full_encoded, test_predictions)

    label_names_full = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

    # Calculate precision, recall, and F1-score
    report_dict_full = classification_report(y_test_full_encoded, test_predictions, output_dict=True, zero_division=0, target_names=label_names_full)

    full_metrics_df = pd.DataFrame()

    full_metrics_df = full_metrics_df.append({
            ('Accuracy', ''): test_accuracy,
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
    full_metrics_df.to_excel(f"{results_dir}/{file_name}_metrics_results_full_test.xlsx", index=True)

    print(f"Test Accuracy: {test_accuracy}")

    # Generate and print the classification report
    print(classification_report(y_test_full_encoded, test_predictions, target_names=label_encoder.classes_))

    shutil.rmtree(dump_dir)

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
    """A custom callback to log training and validation loss, ensuring no duplicates or missing entries."""
    def __init__(self):
        super().__init__()
        self.log_history = {}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        epoch = state.epoch
        if logs is not None:
            if 'loss' in logs:  # Training step
                # Update or set the training loss for the epoch
                if epoch in self.log_history:
                    self.log_history[epoch]['training_loss'] = logs.get('loss')
                else:
                    self.log_history[epoch] = {'epoch': epoch, 'training_loss': logs.get('loss'), 'validation_loss': None, 'eval_runtime': None}
            elif 'eval_loss' in logs:  # Evaluation step
                # Update the validation loss for the epoch, ensuring training loss is carried forward if already logged
                if epoch not in self.log_history:
                    self.log_history[epoch] = {'epoch': epoch, 'training_loss': None, 'validation_loss': logs.get('eval_loss'), 'eval_runtime': logs.get('eval_runtime')}
                else:
                    self.log_history[epoch].update({'validation_loss': logs.get('eval_loss'), 'eval_runtime': logs.get('eval_runtime')})

    def save_logs_to_excel(self, file_name):
        """Convert the log history to a DataFrame and save to an Excel file, ensuring consistent structure."""
        # Convert log history to a list of dicts, ensuring consistent order
        log_list = [self.log_history[key] for key in sorted(self.log_history.keys())]
        pd.DataFrame(log_list).to_excel(file_name, index=False)



def remove_emojis(text):
    # Regex pattern to match most emoji characters
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
                           
    return emoji_pattern.sub(r'', text) # Replace emojis with empty string


# file_names = os.listdir("/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Data/Datasets/Balanced")

directory_path_balanced = "/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Data/Datasets/Balanced"
files_balanced = [(file, os.path.getsize(os.path.join(directory_path, file)))
                   for file in os.listdir(directory_path)
                   if os.path.isfile(os.path.join(directory_path, file))]

files_balanced.sort(key=lambda x: x[1])
for file in files_balanced:
    print("------------------")
    print(f"Now using following dataset:   {file[0].split('.')[0]}")
    print("------------------------------------")
    main_model(file[0].split('.')[0], "xlsx", 1)