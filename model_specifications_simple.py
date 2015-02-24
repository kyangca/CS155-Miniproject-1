import numpy as np
from sklearn import * 

models = {
    linear_model.LogisticRegression: {
        'parameters': {
            'C': [1e-3,1]
            },
        'dataset_type': 'standardized'
        },

    naive_bayes.MultinomialNB: {
        'parameters': {
            'alpha': [1]
            },
        'dataset_type': 'unstandardized'
        },

    naive_bayes.GaussianNB : {
        'parameters': { },
        'dataset_type': 'standardized'
        },

    neighbors.KNeighborsClassifier: {
        'parameters': {
            'n_neighbors': [5],
            },
        'dataset_type': 'standardized'
        },

    svm.SVC: {
        'parameters': [
            {
                'kernel': ['rbf'], 
                'C': [1], 
                'gamma': [1], 
                'probability': [True]
                },
            {
                'kernel': ['linear'],
                'C': [1], 
                'probability': [True]
                },
            {
                'kernel': ['poly'],
                'C': [1],
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
                'n_estimators': [100]
                },
            ],
        'dataset_type': 'standardized'
        },

    ensemble.RandomForestClassifier: {
        'parameters': {
            'n_estimators': [1024],
            'max_features': ['sqrt'],
            'min_samples_leaf': [5],
            'n_jobs': [-1]
            },
        'dataset_type': 'standardized'
        }
    }
