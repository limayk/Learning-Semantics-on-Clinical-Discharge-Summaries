import numpy as np
import helperfunctions
import xgboost as xgb
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import os
import pandas as pd

def main():
    # Path to the dataset file
    data_path = 'datasets/OPENAI_merged_500_summary_embedding.csv'  # correct path???????

    # Load dataset using helper function
    # Returns: labels (target values), data (features), and vector_words (feature names)
    labels, data, vector_words = helperfunctions.loaddataset(data_path)
    print(labels.shape)
    print(data.shape)
    
    # Extract feature names from vector_words
    feature_names = list(np.array(vector_words)[:, 1])
    
    # Create train/validation/test splits based on dataset size
    if len(data) > 10000:
        # For large datasets, use a balanced dataset creation function
        train_set, train_labels, val_set, val_labels, test_set, test_labels = create_balanced_datasets_from_arrays(data, labels)
    else:
        # For smaller datasets, use simple ratio-based splitting
        # Training set: 80% of data
        train_set, train_labels = (data[0:len(data)*8//10], labels[0:len(data)*8//10]) 
        # Validation set: 10% of data
        val_set, val_labels = (data[len(data)*8//10:len(data)*9//10], labels[len(data)*8//10:len(data)*9//10]) 
        # Test set: 10% of data
        test_set, test_labels = (data[len(data)*9//10:len(data)], labels[len(data)*9//10:len(data)]) 
    
    # Print the shapes of our data splits
    print("trainshapes:", train_set.shape, train_labels.shape)
    print("valshapes:", val_set.shape, val_labels.shape) 
    print("testshapes:", test_set.shape, test_labels.shape)
    
    # Balance the class distribution in training and validation sets
    if True:  # Currently enabled - creates balanced datasets with 50/50 class distribution
        train_set, train_labels = helperfunctions.clean_trainset(train_set, train_labels)
        val_set, val_labels = helperfunctions.clean_trainset(val_set, val_labels)
    
    # Print shapes again after balancing
    print("trainshapes:", train_set.shape, train_labels.shape)
    print("valshapes:", val_set.shape, val_labels.shape)
    print("testshapes:", test_set.shape, test_labels.shape)

    # Random Forest hyperparameters
    number_of_rounds = 1  # For Random Forests, we only need 1 round as we create multiple trees at once
    early_stopping = None  # Not needed for Random Forests

    # Create XGBoost compatible data matrices
    dtrain = xgb.DMatrix(train_set, label=train_labels, feature_names=feature_names)
    dval = xgb.DMatrix(val_set, label=val_labels, feature_names=feature_names)
    dtest = xgb.DMatrix(test_set, label=test_labels, feature_names=feature_names)
    evallist = [(dtrain, 'train'), (dval, 'eval')]

    # Option to load a pre-trained model instead of training a new one
    if False:  # Set to True to load existing model (currently disabled)
        bst = xgb.Booster({'nthread': 4})  # Initialize model
        bst.load_model(f"/saved_weights/{number_of_rounds}_rounds.ubj")  # Load model data
    else:
        # Random Forest parameters
        param = {
            'max_depth': 8,           # Maximum depth of trees
            'eta': 0.1,               # Learning rate (step size shrinkage)
            'objective': 'binary:logistic',  # Binary classification
            'lambda': 1.0,            # L2 regularization
            'alpha': 0.0,             # L1 regularization
            'subsample': 0.8,         # Use 80% of data per tree to prevent overfitting
            'colsample_bytree': 0.8,  # Use 80% of features per tree
            'colsample_bynode': 0.8,  # Random feature selection at each split
            'nthread': -1,            # Use all available CPU cores
            'eval_metric': ['logloss', 'error'],  # Metrics to evaluate during training
            'booster': 'gbtree',      # Use tree-based booster
            'num_parallel_tree': 100, # This makes it a Random Forest (>1)
            'seed': 42                # For reproducibility
        }

        # Train the Random Forest model
        evals_result = {}
        bst = xgb.train(param, dtrain, number_of_rounds, evals=evallist, evals_result=evals_result)

    # Save the trained model to file
    bst.save_model(f"saved_weights/{number_of_rounds}_rounds.model.ubj")


    # Evaluate on validation set
    y_pred = bst.predict(dval)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    error_rate = np.mean(val_labels != y_pred_binary)
    print(f"Final Val Accuracy: errorrate: {error_rate:.4f}")

    # Evaluate on test set
    y_pred_test = bst.predict(dtest)
    y_pred_binary_test = (y_pred_test > 0.5).astype(int)  # Convert probabilities to binary predictions
    error_rate = np.mean(test_labels != y_pred_binary_test)
    print(f"Final Test Accuracy: errorrate: {error_rate:.4f}")

    # Generate ROC curve and confusion matrix for the test results
    plot_roc_curve(
        test_labels, 
        y_pred_test, 
        "H:/Master Stanford/CS 229/project/Machine Learning algorithms/results/Random_Forests/", 
        "Random_Forest_with_OPENAI_merged_500_summary_embedding"
    )
    plot_confusion_matrix(
        test_labels, 
        y_pred_binary_test, 
        f"results/Random_Forests/Random_Forest_confusion_with_OPENAI_merged_500_summary_embedding.png"
    )
    
    # Feature importance visualization
    # xgb.plot_importance(bst, importance_type='gain', max_num_features=20, height=0.5)
    # plt.title('Feature Importance (by gain)')
    # plt.show()

def make_plot(result):
    """
    Creates and displays a figure with two subplots showing the training and validation metrics.
    
    Args:
        result (dict): Dictionary containing training history with 'train' and 'eval' metrics
    """
    epochs = len(result['train']['logloss'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    plt.plot(x_axis, result['train']['logloss'], label='Train')
    plt.plot(x_axis, result['eval']['logloss'], label='Test')
    plt.title('XGBoost Log Loss')
    plt.xlabel('Boosting Round')
    plt.ylabel('Log Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, result['train']['error'], label='Train')
    plt.plot(x_axis, result['eval']['error'], label='Test')
    plt.title('XGBoost Classification Error')
    plt.xlabel('Boosting Round')
    plt.ylabel('Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(labels, predictions, save_path):
    """
    Plots and saves a confusion matrix heatmap.
    
    Args:
        labels: Ground-truth labels (0 or 1)
        predictions: Predicted class indices (0 or 1)
        save_path: Path to save the confusion matrix plot
    """
    # Ground truth ie correct labels
    true_classes = labels 

    # Compute and plot confusion matrix
    cm = confusion_matrix(true_classes, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(labels, soft_probs, save_path, model_name):
    """
    Plots and saves an ROC curve with AUC.
    
    Args:
        labels: Ground-truth labels (0 or 1)
        soft_probs: Predicted probabilities for the positive class
        save_path: Base path to save the ROC plot and CSV data
        model_name: Name of the model for file naming and plot labeling
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create folder if needed

    # Ground truth labels
    true_classes = labels  

    # Predicted probabilities for the positive class
    pred_prob_class1 = soft_probs 

    # Compute ROC curve and Area Under Curve (AUC)
    fpr, tpr, _ = roc_curve(true_classes, pred_prob_class1)
    roc_auc = auc(fpr, tpr)

    # Save ROC values to CSV for potential later comparison
    csv_path = os.path.splitext(save_path)[0] + f"_{model_name}.csv"
    df = pd.DataFrame({
        'fpr': fpr, 
        'tpr': tpr, 
        'model': [model_name] * len(fpr), 
        'auc': [roc_auc] * len(fpr)
    })
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)  # Append without headers
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)  # Create new file with headers

    # Create and save ROC curve plot
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')  # Random baseline
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path + f"_{model_name}.png")
    plt.close()

def create_balanced_datasets_from_arrays(X, y, train_size=8000, val_size=1000, test_size=1000, random_seed=42):
    """
    Creates balanced train, validation, and test datasets with equal class distribution.
    
    Args:
        X: Feature data (numpy array)
        y: Labels (numpy array)
        train_size: Number of samples in training set (default: 8000)
        val_size: Number of samples in validation set (default: 1000)
        test_size: Number of samples in test set (default: 1000)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Six numpy arrays: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Combine X and y into pairs for easier handling
    data_pairs = list(zip(X, y))
    
    # Separate data by label
    class_0_data = [pair for pair in data_pairs if pair[1] == 0]
    class_1_data = [pair for pair in data_pairs if pair[1] == 1]
    
    # Calculate how many samples of each class we need
    train_per_class = train_size // 2
    val_per_class = val_size // 2
    test_per_class = test_size // 2
    
    total_needed_per_class = train_per_class + val_per_class + test_per_class
    
    # Check if we have enough samples of each class
    if len(class_0_data) < total_needed_per_class or len(class_1_data) < total_needed_per_class:
        raise ValueError(f"Not enough samples in one or both classes. Need {total_needed_per_class} samples per class.")
    
    # Shuffle the data
    random.shuffle(class_0_data)
    random.shuffle(class_1_data)
    
    # Split the data manually
    class_0_test = class_0_data[:test_per_class]
    class_0_val = class_0_data[test_per_class:test_per_class + val_per_class]
    class_0_train = class_0_data[test_per_class + val_per_class:test_per_class + val_per_class + train_per_class]
    
    class_1_test = class_1_data[:test_per_class]
    class_1_val = class_1_data[test_per_class:test_per_class + val_per_class]
    class_1_train = class_1_data[test_per_class + val_per_class:test_per_class + val_per_class + train_per_class]
    
    # Combine and shuffle the balanced sets
    train_data = class_0_train + class_1_train
    val_data = class_0_val + class_1_val
    test_data = class_0_test + class_1_test
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Unzip the paired data back into X and y
    X_train, y_train = zip(*train_data) if train_data else ([], [])
    X_val, y_val = zip(*val_data) if val_data else ([], [])
    X_test, y_test = zip(*test_data) if test_data else ([], [])
    
    # Convert to lists for consistent return type
    X_train, y_train = list(X_train), list(y_train)
    X_val, y_val = list(X_val), list(y_val)
    X_test, y_test = list(X_test), list(y_test)
 
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

if __name__ == '__main__':
    main()