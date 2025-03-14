import numpy as np
import helperfunctions
import xgboost as xgb
import matplotlib.pyplot as plt

def main():
    data_path = 'datasets/semantic_BertHier_FullText.csv' #correct path???????

    # list of labels: (whole dataset)
    # list of feature values: (whole dataset)
    # vector_words: Tuples(index, key_words), starting with index: 1
    labels, data, vector_words = helperfunctions.loaddataset(data_path)
    print(labels.shape)
    print(data.shape)
    feature_names = list(np.array(vector_words)[:, 1])
    #make datasets:
    train_set, train_labels = (data[0:len(data)*8//10], labels[0:len(data)*8//10]) #current distribution: 80% train
    val_set, val_labels = (data[len(data)*8//10:len(data)*9//10], labels[len(data)*8//10:len(data)*9//10]) #current distribution: 10% train
    test_set, test_labels = (data[len(data)*9//10:len(data)], labels[len(data)*9//10:len(data)]) #current distribution: 10% train
    
    if True: #makes the training set 50/50 True, False
        train_set, train_labels = helperfunctions.clean_trainset(train_set, train_labels) #makes the training set 50/50 True, False
        val_set, val_labels = helperfunctions.clean_trainset(val_set, val_labels)
    
    #printout the shapes
    print("trainshapes:", train_set.shape, train_labels.shape)
    print("trainshapes:", val_set.shape, val_labels.shape)
    #print("trainshapes:", test_set.shape, test_labels.shape)

    #hyperparameters
    number_of_rounds = 1 #Random Forests =1
    early_stopping = None #Not needed in Random Forests



    dtrain = xgb.DMatrix(train_set, label=train_labels, feature_names=feature_names)
    dval = xgb.DMatrix(val_set, label=val_labels, feature_names=feature_names)
    dtest = xgb.DMatrix(test_set, label=test_labels, feature_names=feature_names)
    evallist = [(dtrain, 'train'), (dval, 'eval')]

    if False: #use olf model?
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(f"/saved_weights/{number_of_rounds}_rounds.model")  # load model data

    else:
        param = {
            'max_depth': 8,           # Reduced max depth (typical for boosting)
            'eta': 0.1,               # Learning rate (increased slightly)
            'objective': 'binary:logistic',
            'lambda': 1.0,            # L2 regularization
            'alpha': 0.0,             # L1 regularization
            'subsample': 0.8,         # Use 80% of data per tree to prevent overfitting
            'colsample_bytree': 0.8,  # Use 80% of features per tree
            'colsample_bynode': 0.8,  # Random feature selection at each split
            'nthread': -1,            # Use all available CPU cores
            'eval_metric': ['logloss', 'error'],
            'booster': 'gbtree',      # Use tree-based booster
            'num_parallel_tree': 100,    # IMPORTANT: Set to 1 for boosting (>1 would make it a random forest)
            'seed': 42                # For reproducibility
        }

        #training
        evals_result = {}
        bst = xgb.train(param, dtrain, number_of_rounds, evallist, evals_result=evals_result) #validation needs to improve at least every 10 rounds

    #saving the model
    bst.save_model(f"saved_weights/{number_of_rounds}_rounds.model")

    #plotting
    #make_plot(evals_result) #Random Forest doesn't have a number of rounds

    #evaluation
    y_pred = bst.predict(dval)
    y_pred_binary = (y_pred > 0.5).astype(int)
    error_rate = np.mean(val_labels != y_pred_binary)
    false_negative = np.sum(val_labels[val_labels==1] != y_pred_binary[val_labels==1])/np.sum(val_labels ==1) if np.sum(val_labels==1) > 0 else 0
    false_positive = np.sum(val_labels[val_labels==0] != y_pred_binary[val_labels==0])/np.sum(val_labels==0) if np.sum(val_labels==0) > 0 else 0

    print(f"Final Val Accuracy: errorrate: {error_rate:.4f}, false_negative: {false_negative:.4f}, false_positive: {false_positive:.4f}")

    #Test evaluation
    y_pred_test = bst.predict(dtest)
    y_pred_binary_test = (y_pred_test > 0.5).astype(int)
    error_rate = np.mean(test_labels != y_pred_binary_test)
    false_negative = np.sum(test_labels[test_labels==1] != y_pred_binary_test[test_labels==1])/np.sum(test_labels ==1) if np.sum(test_labels==1) > 0 else 0
    false_positive = np.sum(test_labels[test_labels==0] != y_pred_binary_test[test_labels==0])/np.sum(test_labels ==0) if np.sum(test_labels==0) > 0 else 0

    print(f"Final Test Accuracy: errorrate: {error_rate:.4f}, false_negative: {false_negative:.4f}, false_positive: {false_positive:.4f}")

    xgb.plot_importance(bst, importance_type='gain', max_num_features=20, height=0.5)
    plt.title('Feature Importance (by gain)')
    plt.show()
        

def make_plot(result):
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






if __name__ == '__main__':
    main()
