import numpy as np
import helperfunctions
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import random


def main():
    data_path = 'datasets/our_bert_cos_similarity.csv' #correct path???????

    # list of labels: (whole dataset)
    # list of feature values: (whole dataset)
    # vector_words: Tuples(index, key_words), starting with index: 1
    labels, data, vector_words = helperfunctions.loaddataset(data_path)
    data = helperfunctions.make_intercept(data) #intercept at index: 0
    vector_words.insert(0, [0, 1]) #add index 0 as the intercept

    if len(data)>10000:
        train_set, train_labels, val_set, val_labels, test_set, test_labels = create_balanced_datasets_from_arrays(data, labels)

    else:
        #make datasets:
        train_set, train_labels = (data[0:len(data)*8//10], labels[0:len(data)*8//10]) #current distribution: 80% train
        val_set, val_labels = (data[len(data)*8//10:len(data)*9//10], labels[len(data)*8//10:len(data)*9//10]) #current distribution: 10% train
        test_set, test_labels = (data[len(data)*9//10:len(data)], labels[len(data)*9//10:len(data)]) #current distribution: 10% train
        #printout the shapes
    print("trainshapes:", train_set.shape, train_labels.shape)
    print("trainshapes:", val_set.shape, val_labels.shape)
    print("trainshapes:", test_set.shape, test_labels.shape)
    print(train_labels[:20])
    #define hyperparameters
    learning_rate = 0.01
    max_iteration = 1e6
    epsilon = 1e-6
    theta_start = None #None means zero vector
    feature_size = len(vector_words)-1
    regularization_terms = [1] #list of regularizations

    saved_parameters = True #use saved parameters?

    # Make the Logistic Regrassion
    if saved_parameters == True:
        for reg in regularization_terms:
            LogReg = Logistic_Regression(learning_rate, 
                                        feature_size, 
                                        max_iteration, 
                                        epsilon, 
                                        theta_start, 
                                        reg)
            LogReg.fit(train_set, train_labels)

    # Make the prodiction
    for reg in regularization_terms:
        LogReg = Logistic_Regression(learning_rate, 
                                        feature_size, 
                                        max_iteration, 
                                        epsilon, 
                                        theta_start, 
                                        reg)
        predictions, error_rates, thetas, percentages = LogReg.predict(val_set, val_labels) #use either Validation or Testset with saved weights
        error_rate, false_negative, false_positive = (error_rates[0], error_rates[1], error_rates[2])
        plot_roc_curve(val_labels, percentages, "H:/Master Stanford/CS 229/project/Machine Learning algorithms/results/Linear_Regression/", "Linear_Regression_with_our_bert_cos_similarity") #Change Validation or Testing
        plot_confusion_matrix(val_labels, predictions, f"results/Linear_Regression/Linear_Regression_confusion_with_our_bert_cos_similarity.png") #Change Validation or Testing



class Logistic_Regression:

    def __init__(self, learning_rate, feature_size, max_iteration= 100000, epsilon= 1e-3, theta_start= False, reg=0):
        self.learning_rate = learning_rate
        self.feature_size = feature_size
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.reg = reg

        if theta_start == None:
            self.theta = np.zeros(self.feature_size +1) #plus intercept
        else:
            self.theta = theta_start


    def fit(self, x, labels):

        i = 0
        theta_old = np.zeros(self.feature_size)
        len_x = len(x)

        while i==0 or (i<self.max_iteration and np.linalg.norm(self.theta-theta_old,ord=1)>= self.epsilon):
            theta_old = self.theta.copy()

            z = np.clip(self.theta.dot(x.T), -500, 500)
            h_x = 1/(1+np.exp(-z)) #clip is used to prevent overflow
            
            #gradient assent:
            gradient = (labels-h_x).dot(x)/len_x
            regularization = self.reg/len_x* self.theta
            regularization[0] = 0
            self.theta += self.learning_rate*(gradient-regularization)
            i+=1
            if i%10000 == 0:
                    print("iteration: ",i)
                    print(np.linalg.norm(self.theta-theta_old,ord=1))
        np.save(f"saved_weights/weights_with_reg{self.reg}our_bert_cos_similarity.npy", self.theta)


    def predict(self, x, labels):

        self.theta = np.load(f"saved_weights/weights_with_reg{self.reg}our_bert_cos_similarity.npy") #change to get the correct weights, correct path????????

        percentages = 1/(1+np.exp(-self.theta.dot(x.T)))
        prediction_list = [0 if percentage<0.5 else 1 for percentage in percentages]

        errorlist = [[0, 0, 0] if labels[i]==prediction_list[i] else [1, labels[i], 1-labels[i]] for i in range(len(x))]
        error_rates = np.sum(errorlist, axis= 0)/len(x)
        return prediction_list, error_rates, self.theta, percentages #error_rate, false_negative, false_positive


def plot_confusion_matrix(labels, predictions, save_path):
    """
    Plots and saves a confusion matrix heatmap.
    :param labels: one-hot ground-truth labels, shape (N,2).
    :param predictions: predicted class indices (0 or 1), length N.
    :param save_path: path to save the confusion matrix plot.
    """

    #Ground truth ie correct labels
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
    :param labels: one-hot ground-truth labels, shape (N,2).
    :param soft_probs: predicted probabilities for each class, shape (N,2).
                      soft_probs[:,1] is the probability of class '1' (positive class).
    :param save_path: path to save the ROC plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create folder if needed

    # Ground truth: directly use labels[:, 0] (first column is '1' for positive class)
    true_classes = labels  

    # Correct class-1 probabilities
    pred_prob_class1 = soft_probs 

    # Compute ROC and AUC
    fpr, tpr, _ = roc_curve(true_classes, pred_prob_class1)
    roc_auc = auc(fpr, tpr)

    # Save ROC values to CSV (same path as save_path but with .csv extension)
    csv_path = os.path.splitext(save_path)[0] + f"_{model_name}.csv"
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'model': [model_name] * len(fpr), 'auc': [roc_auc] * len(fpr)})
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)  # Append without headers
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)  # Create new file with headers

    # Plot
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