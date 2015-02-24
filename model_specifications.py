import numpy as np
from sklearn import * 

models = {
    linear_model.LogisticRegression: {
        'parameters': {
            'penalty' : ['l1', 'l2'],
            'C': np.logspace(-4, 8, 13, True)
            },
        'dataset_type': 'standardized'
        },

    naive_bayes.MultinomialNB: {
        'parameters': {
            'alpha': np.append(np.linspace(0, 3, 10), 1)
            },
        'dataset_type': 'unstandardized'
        },

    naive_bayes.GaussianNB : {
        'parameters': { },
        'dataset_type': 'standardized'
        },

    neighbors.KNeighborsClassifier: {
        'parameters': {
            'n_neighbors': [1, 5, 10, 15, 20, 25, 50, 80, 100, 500],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
            },
        'dataset_type': 'standardized'
    
        },

    svm.SVC: {
        'parameters': [
            {
                'kernel': ['rbf'], 
                'C': np.logspace(-7, 3, 11, True), 
                'gamma': np.logspace(-4, 0, 7, True), 
                'probability': [True]
                },
            {
                'kernel': ['linear'],
                'C': np.logspace(-7, 3, 11, True), 
                'probability': [True]
                },
            {
                'kernel': ['poly'],
                'C': np.logspace(-7, 3, 11, True),
                'degree': [2, 3],
                'probability': [True]
                }
            ],
        'dataset_type': 'standardized'

        },

    ensemble.AdaBoostClassifier: {
        'parameters': [
            {
                'base_estimator': [tree.DecisionTreeClassifier(max_depth = 1, max_features = None)],
                'n_estimators': np.logspace(1, 10, 10, True, 2, np.int)
                },
            {
                'base_estimator': [tree.DecisionTreeClassifier(max_depth = 1, max_features = 'sqrt')],
                'n_estimators': np.logspace(1, 10, 10, True, 2, np.int)
                }
            ],
        'dataset_type': 'standardized'

        },

    ensemble.RandomForestClassifier: {
        'parameters': {
            'n_estimators': [1024],
            'max_features': ['sqrt', 'log2', 1, 2, 4, 20],
            'min_samples_leaf': [1, 5, 10, 20],
            'n_jobs': [-1]
            },
        'dataset_type': 'standardized'

        }
    }
