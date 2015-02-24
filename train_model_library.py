import util
from model_specifications import models
from sklearn import preprocessing, grid_search, cross_validation

TRAIN_PATH = 'kaggle_train_tf_idf.csv'

def train_model_library(n_folds = 5, n_folds_to_compute = 5):
    ids, features, labels = util.load_train(TRAIN_PATH)
    kf_cv = cross_validation.KFold(features.shape[0], n_folds = n_folds, shuffle = True)
    scaler = preprocessing.StandardScaler()
    
    model_grid = _generate_model_grid()
    
    tot_v_size = 0
    i = 1
    for train_idx, validate_idx in kf_cv:
        training_features = scaler.fit_transform(features[train_idx, :])
        training_labels = labels[train_idx]
        validation_features = scaler.transform(features[validate_idx, :])
        validation_labels = labels[validate_idx]
        _train_fold(model_grid, training_features, training_labels, validation_features)
        
        tot_v_size += validate_idx.size
        if i >= n_folds_to_compute:
            break
        i += 1
    for 
    
    ensemble_library_pred = np.zeros((tot_v_size, len(model_grid)))
    for model in model_grid:
        ensemble_library_pred[:, model[0]] = np.vstack(model[2])
    
    return ensemble_library_pred, model_grid
    

def _train_fold(model_grid, training_features, training_labels, validation_features):
    for model in model_grid:
        m = model[3](**model[4])
        model[1].append(m.fit(training_features, training_labels))
        model[2].append(m.predict_proba(validation_features)[:, 1:2])

def _generate_model_grid():
    mg = []
    idx = 0
    for key in models.iterkeys():
        for p in grid_search.ParameterGrid(models[key]['parameters']):
            mg.append([idx, [], [], key, p])
            idx += 1
    return mg
