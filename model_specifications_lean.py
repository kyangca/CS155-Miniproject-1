import numpy as np
from sklearn import * 

models = {
    linear_model.LogisticRegression: {
        'parameters': {
            'penalty' : ['l1', 'l2'],
            'C': np.logspace(-4, 8, 8, True)
            },
        'dataset_type': 'standardized'
        },
        
    neighbors.KNeighborsClassifier: {
        'parameters': {
            'n_neighbors': [1, 5, 10],
            'weights': ['uniform', 'distance']
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

    svm.SVC: {
        'parameters': [
            {
                'kernel': ['rbf'], 
                'C': np.logspace(-7, 3, 6, True),  
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
            'min_samples_leaf': [1, 5, 10],
            'n_jobs': [-1]
            },
        'dataset_type': 'standardized'
        }
    }
