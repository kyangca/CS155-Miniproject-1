import numpy as np
from sklearn import tree 

models = {
    linear_model.LogisticRegression: {
        'parameters': {
            'penalty' : ['l1', 'l2'],
            'C': np.logspace(-4, 8, 13, True)
            }
        },

    naive_bayes.MultinomialNB: {
        'parameters': {
            'alpha': np.append(np.linspace(0, 3, 10), 1)
            }
        },

    naive_bayes.GaussianNB : {
        'parameters': { }
        },

    neighbors.KNeighborsClassifier: {
        'parameters': {
            'n_neighbors': [1, 5, 10, 15, 20, 25, 50, 80, 100, 500],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
            }
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
            ]
        },

    ensemble.AdaBoostClassifier: {
        'parameters': [
            {
                'base_estimator': [tree.DecisionTreeClassifier(max_depth = 1, max_features = None)],
                'n_estimators': np.logspace(1, 13, 1, True, 2)
                },
            {
                'base_estimator': [tree.DecisionTreeClassifier(max_depth = 1, max_features = 'sqrt')],
                'n_estimators': np.logspace(1, 13, 1, True, 2)
                }
            ]
        },

    ensemble.RandomForestClassifier: {
        'parameters': {
            'n_estimators': [1024],
            'max_features': ['sqrt', 'log2', 1, 2, 4, 20],
            'min_samples_leaf': [1, 5, 10, 20],
            'n_jobs': [-1]
            }
        }
    }
