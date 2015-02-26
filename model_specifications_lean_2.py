import numpy as np
from sklearn import * 

# dictionary specifying models and parameters to include in model library
models = {
    linear_model.LogisticRegression: {
        'parameters': {
            'penalty' : ['l1', 'l2'], # norm used in penalty term
            'C': np.logspace(-4, 8, 13, True) # inverse regularization strength
            },
        
        # 'standardized' means that the data set should be standardized 
        # along each feature
        'dataset_type': 'standardized' 
        },
    
    # Classifies points using the values of its K nearest neighbors
    neighbors.KNeighborsClassifier: {
        'parameters': {
            'n_neighbors': [1, 5, 10, 30], # number of nearest neighbors used in classification
            'weights': ['uniform', 'distance'] # 'distance' weights each NN by 1/distance
            },
        'dataset_type': 'standardized'
        },
    
    # estimates likelihood by counting data
    naive_bayes.MultinomialNB: {
        'parameters': {
            'alpha': np.append(np.linspace(0, 3, 10), 1) # smoothing parameter 
            },
        'dataset_type': 'unstandardized'
        },
    
    # assumes Gaussian likelihood
    naive_bayes.GaussianNB : {
        'parameters': { },
        'dataset_type': 'standardized'
        },
    
    # SVM classifier
    svm.SVC: {
        'parameters': [
            {
                'kernel': ['rbf'], 
                'C': np.logspace(-7, 3, 11, True),  # missclassification penalty
                'probability': [True]
                },
        ],
        'dataset_type': 'standardized'
        },

    ensemble.AdaBoostClassifier: {
        'parameters': [
            {
                'base_estimator': [tree.DecisionTreeClassifier(max_depth = 1, max_features = 'sqrt')],
                'n_estimators': np.logspace(1, 10, 4, True, 2, np.int)
                }
            ],
        'dataset_type': 'standardized'
        },

    ensemble.RandomForestClassifier: {
        'parameters': {
            'n_estimators': [1024],
            'max_features': ['sqrt'],
            'min_samples_leaf': [1, 5, 10, 20],
            'n_jobs': [-1]
            },
        'dataset_type': 'standardized'
        }
    }
