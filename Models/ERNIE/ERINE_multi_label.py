import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from transformers import TrainerCallback
import os
import shutil
import re
import time
from pathlib import Path

def main_model(file_name, ext):

    current_file_path = Path(__file__).parent

    path_to_project = current_file_path.parents[1]

    df = pd.read_csv(f"{path_to_project}/Data/Datasets/Multi-label/{file_name}.{ext}")

    results_dir = f"{path_to_project}/Models/ERNIE/Output/Multi-label/{file_name}"
    dump_dir = results_dir+"/Dump"

    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
        
    os.mkdir(results_dir)
    os.mkdir(dump_dir)

    df = df[df['review'].notna() & (df['review'] != '')]
    df['review'] = df['review'].str.replace('[^\x20-\x7E]', '', regex=True)

    X = df['review'].values 
    y = df[['bug report', 'user experience', 'rating', 'feature request']].values

    X_train_CV, X_test_full, y_train_CV, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en")

    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True, max_length=128)

    loss_logging_callback = LossLoggingCallback()

    # K-Fold Cross-Validation
    n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Variables to accumulate scores
    best_f1 = 0
    best_model = None
    metrics_df = pd.DataFrame()

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_CV, y_train_CV)):
        print(f"Fold {fold+1}/{n_splits}")
        start_time = time.time()

        X_train, X_val = X_train_CV[train_index], X_train_CV[val_index]
        y_train, y_val = y_train_CV[train_index], y_train_CV[val_index]

        train_encodings = tokenize_function(X_train.tolist())
        val_encodings = tokenize_function(X_val.tolist())

        train_dataset = MultiLabelDataset(train_encodings, y_train)
        val_dataset = MultiLabelDataset(val_encodings, y_val)

        model = AutoModelForSequenceClassification.from_pretrained("nghuyong/ernie-2.0-base-en", num_labels=4, problem_type="multi_label_classification")

        training_args = TrainingArguments(
            output_dir=f"{dump_dir}/res",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.1,
            logging_dir=f"{dump_dir}/logs",
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            max_grad_norm=1.0,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_strategy="epoch",
            save_total_limit=2,
            lr_scheduler_type='linear'
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
            callbacks=[loss_logging_callback]
        )

        trainer.train()

        loss_logging_callback.save_logs_to_excel(f"{results_dir}/fold_loss.xlsx")

        results = trainer.evaluate()
        
        if results['eval_f1'] > best_f1:
            best_f1 = results['eval_f1']
            best_model = model  


        predictions = trainer.predict(val_dataset)
        pred_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
        threshold = 0.5
        binary_predictions = (pred_probs > threshold).astype(int)

        # True labels
        true_labels = predictions.label_ids
        f1 = f1_score(true_labels, binary_predictions, average='micro')

        report_dict = classification_report(true_labels, binary_predictions, output_dict=True, zero_division=0, target_names=['bug report', 'user experience', 'rating', 'feature request'])
        # avg_metrics = report_dict['weighted avg']  # Use 'macro avg' or 'weighted avg' based on your preference
        end_time = time.time()
        # Append the metrics for this fold to the DataFrame
        metrics_df = metrics_df.append({
            ('Fold', ''): fold + 1,
            ('F1-Score', ''): f1,
            ('Train Time', ''): str(end_time - start_time)+" s",
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

    metrics_df.columns = pd.MultiIndex.from_tuples([(c,) if isinstance(c, str) else c for c in metrics_df.columns])
    metrics_df.to_excel(f"{results_dir}/fold_metrics.xlsx", index=True)

    test_encodings = tokenize_function(X_test_full.tolist())
    test_dataset = MultiLabelDataset(test_encodings, y_test_full)
    test_trainer = Trainer(model=best_model)
    test_predictions = test_trainer.predict(test_dataset)
    test_pred_probs = torch.sigmoid(torch.tensor(test_predictions.predictions)).numpy()
    threshold = 0.5
    test_binary_predictions = (test_pred_probs > threshold).astype(int)

    test_true_labels = test_predictions.label_ids
    test_f1 = f1_score(test_true_labels, test_binary_predictions, average='micro')

    test_report_dict = classification_report(test_true_labels, test_binary_predictions, output_dict=True, zero_division=0, target_names=['bug report', 'user experience', 'rating', 'feature request'])
    # avg_metrics = report_dict['weighted avg']  # Use 'macro avg' or 'weighted avg' based on your preference
    # Append the metrics for this fold to the DataFrame
    test_metrics_df = pd.DataFrame()

    test_metrics_df = test_metrics_df.append({
            ('F1', ''): test_f1,
            ('Bug Report', 'P'): test_report_dict['bug report']['precision'],
            ('Bug Report', 'R'): test_report_dict['bug report']['recall'],
            ('Bug Report', 'F1'): test_report_dict['bug report']['f1-score'],
            ('Feature Request', 'P'): test_report_dict['feature request']['precision'],
            ('Feature Request', 'R'): test_report_dict['feature request']['recall'],
            ('Feature Request', 'F1'): test_report_dict['feature request']['f1-score'],
            ('Rating', 'P'): test_report_dict['rating']['precision'],
            ('Rating', 'R'): test_report_dict['rating']['recall'],
            ('Rating', 'F1'): test_report_dict['rating']['f1-score'],
            ('User Experience', 'P'): test_report_dict['user experience']['precision'],
            ('User Experience', 'R'): test_report_dict['user experience']['recall'],
            ('User Experience', 'F1'): test_report_dict['user experience']['f1-score']
        }, ignore_index=True)

    test_metrics_df.columns = pd.MultiIndex.from_tuples([(c,) if isinstance(c, str) else c for c in test_metrics_df.columns])
    test_metrics_df.to_excel(f"{results_dir}/metrics_results_full_test.xlsx", index=True)

    print(f"Test F1: {test_f1}")

    # Generate and print the classification report
    print(test_report_dict)

    shutil.rmtree(dump_dir)

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
class LossLoggingCallback(TrainerCallback):
    """A custom callback to log training and validation loss."""
    def __init__(self):
        super().__init__()
        self.log_history = []
        self.log_train_loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # This method captures both training and evaluation logs, so it's more general than on_epoch_end
        if logs is not None:
            # Capture both training and evaluation steps
            if 'loss' in logs:  # Indicates a training step
                self.log_train_loss_history.append({
                    'epoch': state.epoch,
                    'training_loss': logs.get('loss'),
                })
            elif 'eval_loss' in logs:  # Indicates an evaluation step
                # Make sure to capture the last training loss as well
                last_training_loss = self.log_train_loss_history[-1]['training_loss'] if self.log_train_loss_history else None
                self.log_history.append({
                    'epoch': state.epoch,
                    'training_loss': last_training_loss,  # Include last known training loss for reference
                    'validation_loss': logs.get('eval_loss'),
                    'eval_runtime':logs.get('eval_runtime')
                })

    def save_logs_to_excel(self, file_name):
        """Save the recorded logs to a Excel file."""
        pd.DataFrame(self.log_history).to_excel(file_name, index=False)


current_file_path = Path(__file__).parent
path_to_project = current_file_path.parents[1]

directory_path_multi = path_to_project / 'Data' / 'Datasets' / 'Multi-label'

files_multi = [(file.name, file.stat().st_size) 
               for file in directory_path_multi.iterdir() 
               if file.is_file() and not file.name.startswith('.')]

files_multi.sort(key=lambda x: x[1])

# Iterating through sorted files and printing details
for file in files_multi:
    print("------------------")
    print(f"Now using following dataset:   {file[0].split('.')[0]}")
    print("------------------------------------")
    main_model(file[0].split('.')[0], file[0].split('.')[1])

    
# directory_path_multi = "/Users/pavelghazaryan/Desktop/ThirdYear/FinalProject/Data/Datasets/Multi-label"
# files_multi = [(file, os.path.getsize(os.path.join(directory_path_multi, file)))
#                    for file in os.listdir(directory_path_multi)
#                    if os.path.isfile(os.path.join(directory_path_multi, file))]

# files_multi.sort(key=lambda x: x[1])
# for file in files_multi:
#     print("------------------")
#     print(f"Now using following dataset:   {file[0].split('.')[0]}")
#     print("------------------------------------")
#     # main_model(file[0].split('.')[0], file[0].split('.')[1], 1)

# # main_model("dataset_gpt_multi_label_100", "csv")
