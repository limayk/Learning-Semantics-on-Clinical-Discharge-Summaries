import numpy as np
import helperfunctions



def main():
    data_path = 'datasets/our_bert_features.csv' #correct path???????

    # list of labels: (whole dataset)
    # list of feature values: (whole dataset)
    # vector_words: Tuples(index, key_words), starting with index: 1
    labels, data, vector_words = helperfunctions.loaddataset(data_path)
    data = helperfunctions.make_intercept(data) #intercept at index: 0
    vector_words.insert(0, [0, 1]) #add index 0 as the intercept

    Datazise = 10000
    #make datasets:
    train_set, train_labels = (data[0:Datazise*8//10], labels[0:Datazise*8//10]) #current distribution: 80% train
    val_set, val_labels = (data[Datazise*8//10:Datazise*9//10], labels[Datazise*8//10:Datazise*9//10]) #current distribution: 10% train
    test_set, test_labels = (data[Datazise*9//10:Datazise], labels[Datazise*9//10:Datazise]) #current distribution: 10% train
    #printout the shapes
    print("trainshapes:", train_set.shape, train_labels.shape)
    print("trainshapes:", val_set.shape, val_labels.shape)
    print("trainshapes:", test_set.shape, test_labels.shape)

    #define hyperparameters
    learning_rate = 1e-4
    max_iteration = 1e6
    epsilon = 1e-6
    theta_start = None #None means zero vector
    feature_size = len(vector_words)-1
    regularization_terms = [0.1, 0.5, 2, 5, 20, 100] #list of regularizations

    saved_parameters = False #use saved parameters?

    # Make the Logistic Regrassion
    if saved_parameters == False:
        for reg in regularization_terms:
            LogReg = Logistic_Regression(learning_rate, 
                                        feature_size, 
                                        max_iteration, 
                                        epsilon, 
                                        theta_start, 
                                        reg)
            LogReg.fit(train_set, train_labels)

    # Make the prodiction
    with open(f"results/results_semantix.txt", "w") as text_file: #correct path?????
        for reg in regularization_terms:
            LogReg = Logistic_Regression(learning_rate, 
                                            feature_size, 
                                            max_iteration, 
                                            epsilon, 
                                            theta_start, 
                                            reg)
            predictions, error_rates, thetas = LogReg.predict(val_set, val_labels, saved_parameters)
            error_rate, false_negative, false_positive = (error_rates[0], error_rates[1], error_rates[2])

            ## writing the results into the file
            print(f"Regularization = {reg}, error_rate, false_negative, false_positive: {error_rate}, {false_negative}, {false_positive}")
            print("Max_values and words:")
            max_indizes = np.argsort(thetas)
            print("bias: ", thetas[0])
            for i in range(1, 6):
                print(vector_words[max_indizes[-i]], thetas[max_indizes[-i]])
            print("Min_values and words:")
            for i in range(0, 5):
                print(vector_words[max_indizes[i]], thetas[max_indizes[i]])
            text_file.write(f"Regularization = {reg}, error_rate, false_negative, false_positive: {error_rate}, {false_negative}, {false_positive}")
            text_file.write("\n")
            text_file.write("Max_values and words:")
            text_file.write("\n")
            max_indizes = np.argsort(thetas)
            text_file.write(f"bias: {thetas[0]}")
            text_file.write("\n")
            for i in range(1, 6):
                text_file.write(f"{vector_words[max_indizes[-i]]}, {thetas[max_indizes[-i]]}")
                text_file.write("\n")
            text_file.write("Min_values and words:")
            text_file.write("\n")
            for i in range(0, 5):
                text_file.write(f"{vector_words[max_indizes[i]]}, {thetas[max_indizes[i]]}")
                text_file.write("\n")
    text_file.close()

    helperfunctions.print_results()


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
        np.save(f"saved_weights/weights_with_reg{self.reg}.npy", self.theta) #correct path????????


    def predict(self, x, labels, saved_parameters = False):

        if saved_parameters == True:
            self.theta = np.load(f"saved_weights/weights_with_reg{self.reg}.npy") #change to get the correct weights, correct path????????

        percentages = 1/(1+np.exp(-self.theta.dot(x.T)))
        prediction_list = [0 if percentage<0.5 else 1 for percentage in percentages]

        errorlist = [[0, 0, 0] if labels[i]==prediction_list[i] else [1, labels[i], 1-labels[i]] for i in range(len(x))]
        error_rates = np.sum(errorlist, axis= 0)/len(x)
        return prediction_list, error_rates, self.theta #error_rate, false_negative, false_positive


if __name__ == '__main__':
    main()

























if __name__ == "__main__":
    #path of set
    trainpath = ""
    validationpath = ""
    testpath = ""

    