import matplotlib.pyplot as plt
import numpy as np

def clean_trainset(dataset, labels):
    if len(labels.shape)>1:
        inequality = np.sum(labels[:, 0])/len(labels)
        print(inequality)
        while inequality != 0.5:
            int_rand = np.random.randint(len(labels))
            #print(int_rand)
            print(inequality)
            if inequality>0.5 and labels[int_rand][0]==1:
                    dataset = np.delete(dataset, int_rand, 0)
                    labels = np.delete(labels, int_rand, 0)
            elif inequality<0.5 and labels[int_rand][0]==0:
                dataset = np.delete(dataset, int_rand, 0)
                labels = np.delete(labels, int_rand, 0)
            inequality = np.sum(labels[:, 0])/len(labels)
    else:
        inequality = np.sum(labels)/len(labels)
        print(inequality)
        while inequality != 0.5:
            int_rand = np.random.randint(len(labels))
            #print(int_rand)
            print(inequality)
            if inequality>0.5 and labels[int_rand]==1:
                    dataset = np.delete(dataset, int_rand, 0)
                    labels = np.delete(labels, int_rand, 0)
            elif inequality<0.5 and labels[int_rand]==0:
                dataset = np.delete(dataset, int_rand, 0)
                labels = np.delete(labels, int_rand, 0)
            inequality = np.sum(labels)/len(labels)
    return dataset, labels

def loaddataset(data_path):
    with open(data_path, "r") as csv_file:
        feature_words = csv_file.readline().strip().split(',')

    label_column = [i for i in range(len(feature_words)) if feature_words[i].startswith('Y')]
    featurecolumns = [i for i in range(len(feature_words)) if not feature_words[i].startswith('Y')]
    vector_words = [[i, feature_words[i]] for i in range(len(feature_words)) if not feature_words[i].startswith('Y')]
    
    labels = np.loadtxt(data_path, delimiter = ",", skiprows=1, usecols = label_column)
    data = np.loadtxt(data_path, delimiter = ",", skiprows=1, usecols = featurecolumns)

    return labels, data, vector_words

def make_intercept(x):

    with_intercept = np.zeros((len(x), len(x[0])+1), dtype = x.dtype)
    with_intercept[:, 0] = 1
    with_intercept[:, 1:] = x

    return with_intercept

def plot_training(list1, name_of_list1, list2, name_of_list2, save_path):
    x_values = np.linspace(1, len(list1), len(list1))
    plt.plot(x_values, list1, label=name_of_list1)
    plt.plot(x_values, list2, label=name_of_list2)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

#error_rates: Tuple(error_rate, false_negative, false_positive)
#reg: regularization value no List!
#weights: None doesn't include them to the output (needs vector_words)
#vector_words: None doesn't include them to the output (needs weights)
def print_results(error_rates, reg, result_path, vector_words=None, weights=None):

    ## writing the results into the file
    with open(result_path, "w") as text_file: #correct path?????
            
            error_rate, false_negative, false_positive = (error_rates[0], error_rates[1], error_rates[2])

            # regularization and error rates
            print(f"Regularization = {reg}, error_rate, false_negative, false_positive: {error_rate}, {false_negative}, {false_positive}")

            #printing out weights
            if weights!=None and vector_words!=None:

                #sorting the weights
                max_indizes = np.argsort(weights)

                #printing most positive weights
                print("Max_values and words:")
                print("bias: ", weights[0])
                for i in range(1, 6):
                    print(vector_words[max_indizes[-i]], weights[max_indizes[-i]])

                #printing most negative weights
                print("Min_values and words:")
                for i in range(0, 5):
                    print(vector_words[max_indizes[i]], weights[max_indizes[i]])
                text_file.write(f"Regularization = {reg}, error_rate, false_negative, false_positive: {error_rate}, {false_negative}, {false_positive}")
                text_file.write("\n")
    text_file.close()

    


