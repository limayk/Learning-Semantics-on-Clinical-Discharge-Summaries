import matplotlib.pyplot as plt
import numpy as np

def clean_trainset(dataset, labels):
    """
    Balances a dataset by randomly removing samples until both classes have equal representation.
    
    Args:
        dataset: Input features array
        labels: Target labels array
        
    Returns:
        Tuple of (balanced dataset, balanced labels)
    """
    np.random.seed(42)  # Set seed for reproducibility
    
    # Handle multi-dimensional labels (one-hot encoded)
    if len(labels.shape) > 1:
        inequality = np.sum(labels[:, 0]) / len(labels)
        print(inequality)
        while inequality != 0.5:  # Loop until perfect balance (50% of each class)
            int_rand = np.random.randint(len(labels))
            print(inequality)
            # Remove samples from majority class
            if inequality > 0.5 and labels[int_rand][0] == 1:
                    dataset = np.delete(dataset, int_rand, 0)
                    labels = np.delete(labels, int_rand, 0)
            elif inequality < 0.5 and labels[int_rand][0] == 0:
                dataset = np.delete(dataset, int_rand, 0)
                labels = np.delete(labels, int_rand, 0)
            inequality = np.sum(labels[:, 0]) / len(labels)
    # Handle single-dimensional labels
    else:
        inequality = np.sum(labels) / len(labels)
        print(inequality)
        while inequality != 0.5:  # Loop until perfect balance (50% of each class)
            int_rand = np.random.randint(len(labels))
            print(inequality)
            # Remove samples from majority class
            if inequality > 0.5 and labels[int_rand] == 1:
                    dataset = np.delete(dataset, int_rand, 0)
                    labels = np.delete(labels, int_rand, 0)
            elif inequality < 0.5 and labels[int_rand] == 0:
                dataset = np.delete(dataset, int_rand, 0)
                labels = np.delete(labels, int_rand, 0)
            inequality = np.sum(labels) / len(labels)
    return dataset, labels

def loaddataset(data_path):
    """
    Loads dataset from a CSV file, separating features and labels.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Tuple of (labels, data, vector_words)
        - labels: Target values
        - data: Feature values
        - vector_words: List of [index, feature_name] pairs
    """
    # Read header line to get feature names
    with open(data_path, "r") as csv_file:
        feature_words = csv_file.readline().strip().split(',')

    # Identify label columns (starting with 'Y') and feature columns
    label_column = [i for i in range(len(feature_words)) if feature_words[i].startswith('Y')]
    featurecolumns = [i for i in range(len(feature_words)) if not feature_words[i].startswith('Y')]
    vector_words = [[i, feature_words[i]] for i in range(len(feature_words)) if not feature_words[i].startswith('Y')]
    
    # Load data from CSV
    labels = np.loadtxt(data_path, delimiter = ",", skiprows=1, usecols = label_column)
    data = np.loadtxt(data_path, delimiter = ",", skiprows=1, usecols = featurecolumns)

    return labels, data, vector_words

def make_intercept(x):
    """
    Adds an intercept term (column of ones) to the feature matrix.
    
    Args:
        x: Original feature matrix
        
    Returns:
        Feature matrix with intercept column added as first column
    """
    with_intercept = np.zeros((len(x), len(x[0])+1), dtype = x.dtype)
    with_intercept[:, 0] = 1  # Set first column to ones (intercept term)
    with_intercept[:, 1:] = x  # Copy original features to remaining columns

    return with_intercept

def plot_training(list1, name_of_list1, list2, name_of_list2, save_path):
    """
    Plots two lists (e.g., training and validation metrics) and saves the figure.
    
    Args:
        list1: First list of values to plot
        name_of_list1: Label for first list
        list2: Second list of values to plot
        name_of_list2: Label for second list
        save_path: Path to save the resulting plot
    """
    x_values = np.linspace(1, len(list1), len(list1))
    plt.plot(x_values, list1, label=name_of_list1)
    plt.plot(x_values, list2, label=name_of_list2)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def print_results(error_rates, reg, result_path, vector_words=None, weights=None):
    """
    Prints and saves model evaluation results.
    
    Args:
        error_rates: Tuple of (error_rate, false_negative, false_positive)
        reg: Regularization parameter value
        result_path: Path to save results
        vector_words: List of feature names with indices (optional)
        weights: Model weights (optional)
    """
    ## Writing the results into the file
    with open(result_path, "w") as text_file:
            
            error_rate, false_negative, false_positive = (error_rates[0], error_rates[1], error_rates[2])

            # Print regularization and error rates
            print(f"Regularization = {reg}, error_rate, false_negative, false_positive: {error_rate}, {false_negative}, {false_positive}")

            # If weights and feature names are provided, analyze feature importance
            if weights is not None and vector_words is not None:

                # Sort weights to identify most important features
                max_indizes = np.argsort(weights)

                # Print most positive weights (top 5 features)
                print("Max_values and words:")
                print("bias: ", weights[0])
                for i in range(1, 6):
                    print(vector_words[max_indizes[-i]], weights[max_indizes[-i]])

                # Print most negative weights (bottom 5 features)
                print("Min_values and words:")
                for i in range(0, 5):
                    print(vector_words[max_indizes[i]], weights[max_indizes[i]])
                
                # Write results to file
                text_file.write(f"Regularization = {reg}, error_rate, false_negative, false_positive: {error_rate}, {false_negative}, {false_positive}")
                text_file.write("\n")
    text_file.close()