import torch
import numpy as np
from datasets import load_metric, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
from sklearn.model_selection import KFold, ParameterGrid

data_files = {"train": "train.csv", "test": "test.csv"}

dataset = load_dataset("full_data", data_files=data_files)
imdb = dataset

small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(100000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(10000))])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    load_mape = evaluate.load("mape")
    load_mae = evaluate.load("mae")
    
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)
    
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels,average="weighted")["f1"]
    
    predictions = predictions+1
    labels = labels+1
    mae = load_mae.compute(predictions=predictions, references=labels)["mae"]
    mape = load_mape.compute(predictions=predictions, references=labels)["mape"]

    return {"accuracy": accuracy, "f1": f1, "mae":mae, "mape":mape}

repo_name = "finetuning-sentiment-model-3000-samples"

# Define hyperparameters for grid search
param_grid = {
    "learning_rate": [1e-3,1e-2,1e-1],
    "per_device_train_batch_size": [16,32,64,128,256],
    "num_train_epochs": [5,10,15,20,25,30],
    "weight_decay": [0.01,0.001],
}

# Use K-Fold cross-validation for evaluation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the best metrics
best_metrics = {"eval_accuracy": 0, "f1": 0, "mae":0, "mape":0}
best_hyperparams = {}

# Perform grid search
for params in ParameterGrid(param_grid):
    print(f"\nTraining with hyperparameters: {params}")
    training_args = TrainingArguments(
        output_dir=repo_name,
        save_strategy="epoch",
        push_to_hub=False,
        **params
    )
    
    fold_idx = 1
    fold_metrics = {"eval_accuracy": 0, "f1": 0, "mae":0, "mape":0}

    # Train and evaluate the model for each fold
    for train_index, eval_index in k_fold.split(tokenized_train):
        print(f"\nFold {fold_idx}")
        
        # Get train and evaluation datasets for the fold
        train_dataset = tokenized_train.select(train_index)
        eval_dataset = tokenized_train.select(eval_index)

        # Train
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        
        # Evaluate on the test set for the fold
        eval_results = trainer.evaluate(tokenized_test)
        print(f"Eval results for fold {fold_idx}: {eval_results}")

        # Update the best metrics if necessary
        if eval_results["eval_accuracy"] > fold_metrics["eval_accuracy"]:
            fold_metrics = eval_results
        fold_idx += 1

    # Print the average metrics for the folds
    print(f"\nAverage metrics for hyperparameters {params}:")
    print(f"Accuracy: {fold_metrics['eval_accuracy']}")
    print(f"F1: {fold_metrics['eval_f1']}")
    print(f"MAE: {fold_metrics['eval_mae']}")
    print(f"MAPE: {fold_metrics['eval_mape']}")

    # Update the best metrics and hyperparameters if necessary
    if fold_metrics["eval_accuracy"] > best_metrics["eval_accuracy"]:
        best_metrics = fold_metrics
        best_hyperparams = params

print(f"\nBest hyperparameters: {best_hyperparams}")
print(f"Best metrics:")
print(f"Accuracy: {best_metrics['eval_accuracy']}")
print(f"F1: {best_metrics['eval_f1']}")
print(f"MAE: {best_metrics['eval_mae']}")
print(f"MAPE: {best_metrics['eval_mape']}")


