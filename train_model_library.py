import util
from model_specifications import models
from sklearn import preprocessing, grid_search, cross_validation

TRAIN_PATH = 'kaggle_train_tf_idf.csv'

def train_model_library(n_folds = 5, n_folds_to_compute = 5):
    # ensemble_library_pred is an array of predictions made by the individual models.
    # each row is an obesrvation in the validation set, and each column is the
    # prediction of a cross validation model.
    # validation_labels is a column vector corresponding to the labels of the
    # observations in the validation set.
    # model_grid is a list of lists. each element corresponds to a single model.
    # m = model_grid[i]. m[0] is the model index (corresponding to a column in
    # ensemble_library_pred. m[1] is a list of n_folds_to_compute model objects.
    # m[2] is empty and holds the predictions of each model until the end
    # m[3] is the constructor for that model i.
    # m[4] is a dictionary specifying the model parameters for model i.

    ids, features, labels = util.load_train(TRAIN_PATH)
    kf_cv = cross_validation.KFold(features.shape[0], n_folds = n_folds, shuffle = True)
    scaler = preprocessing.StandardScaler()
    
    model_grid = _generate_model_grid()
    
    tot_v_size = 0
    i = 1
    validation_labels = []
    for train_idx, validate_idx in kf_cv:
        training_features = scaler.fit_transform(features[train_idx, :])
        training_labels = labels[train_idx]
        validation_features = scaler.transform(features[validate_idx, :])
        validation_labels.append(labels[validate_idx][:, np.newaxis])
        _train_fold(model_grid, training_features, training_labels, validation_features)
        
        tot_v_size += validate_idx.size
        if i >= n_folds_to_compute:
            break
        i += 1
    
    # calibrate scaler to entire training set for subsequent testing
    scaler.fit(features)
    # stack individual validation folds
    validation_labels = np.vstack(validation_labels)
    # populate the ensemble library pred and empty the model store to reduce memory
    ensemble_library_pred = np.zeros((tot_v_size, len(model_grid)))
    for model in model_grid:
        ensemble_library_pred[:, model[0]] = np.vstack(model[2])
        model[2] = [] 

    return ensemble_library_pred, validation_labels, scaler, model_grid
    
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
