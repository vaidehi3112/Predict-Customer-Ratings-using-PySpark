# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, GlobalMaxPooling1D, Embedding
from transformers import TFBertModel,BertTokenizer
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid

# Load training and testing data
train_df = pd.read_csv('train.csv', nrows=100000)
test_df = pd.read_csv('test.csv', nrows=10000)

# Load BERT tokenizer
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define maximum sequence length for tokenization
max_seq_length = 128

# Tokenize train data
train_data = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
train_tokens = tokenizer(train_data, max_length=max_seq_length, truncation=True, padding=True)["input_ids"]

# Tokenize test data
test_data = test_df['text'].tolist()
test_labels = test_df['label'].tolist()
test_tokens = tokenizer(test_data, max_length=max_seq_length, truncation=True, padding=True)["input_ids"]

# Define a function to create the LSTM model
def create_model(params):
    model = Sequential([
        Embedding(input_dim=tokenizer.vocab_size, output_dim=32, input_length=None),
        Bidirectional(LSTM(params['lstm_units'], return_sequences=True)),
        GlobalMaxPooling1D(),
        Dropout(params['dropout_rate']),
        Dense(params['num_filters'], activation='relu'),
        Dropout(params['dropout_rate']),
        Dense(5, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(lr=params['learning_rate'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define hyperparameters and parameter grid for grid search
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

param_grid = {'dropout_rate': [0.3, 0.2],
              'lstm_units': [16,32,64],
              'num_filters': [16,32,64],
              'num_train_epochs': [5,8,10],
              'per_device_train_batch_size': [8,16,32],
              'learning_rate': [1e-3, 1e-2, 1e-1]}

# Prepare test data
X_test = np.array(test_tokens)
y_test = np.array(test_labels)
best_metrics = {"eval_accuracy": 0, "mae": 0}

# Grid search loop
for params in ParameterGrid(param_grid):
    print(f"\nTraining with hyperparameters: {params}")
    fold_idx = 1
    fold_metrics = {"eval_accuracy": [], "mae": []}

     # Train and evaluate the model for each fold
    for train_index, eval_index in kf.split(train_tokens):
        print(f"\nFold {fold_idx}")
        
        # Get train and evaluation datasets for the fold
        X_train, X_eval = np.array(train_tokens)[train_index], np.array(train_tokens)[eval_index]
        y_train, y_eval = np.array(train_labels)[train_index], np.array(train_labels)[eval_index]
        
        # Create model
        model = create_model(params)
        
        # Train model
        model.fit(
            X_train, y_train,
            batch_size=params['per_device_train_batch_size'],
            epochs=params['num_train_epochs'],
            verbose=1,
            shuffle=True
        )

       # Evaluate on the validation set for the fold
        y_pred = model.predict(X_eval).squeeze()

        fold_metrics["eval_accuracy"].append(accuracy_score(y_eval, np.argmax(y_pred, axis=1)))
        fold_metrics["mae"].append(mean_absolute_error(y_eval, np.argmax(y_pred, axis=1)))
        
        fold_idx += 1

    # Print the average metrics for the folds
    print(f"\nAverage metrics for the hyperparameters {params}:")
    print(f"Accuracy: {np.mean(fold_metrics['eval_accuracy'])}")
    print(f"MAE: {np.mean(fold_metrics['mae'])}")

    # Update the best metrics and hyperparameters if necessary
    if np.mean(fold_metrics["eval_accuracy"]) > best_metrics["eval_accuracy"]:
        best_metrics = {
            "eval_accuracy": np.mean(fold_metrics["eval_accuracy"]),
            "mae": np.mean(fold_metrics["mae"])
        }
        best_hyperparams = params

# Print the best hyperparameters and metrics
print(f"\nBest hyperparameters: {best_hyperparams}")
print(f"Best metrics:")
print(f"Accuracy: {best_metrics['eval_accuracy']}")
print(f"MAE: {best_metrics['mae']}")

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
mae = mean_absolute_error(y_test, predicted_labels)

#print test results
print("Test Classification Accuracy is:", test_acc)
print("Test MAE is:", mae)
