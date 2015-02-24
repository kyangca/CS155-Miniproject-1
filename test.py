import numpy as np
import ensemble
from train_model_library import train_model_library
import util
import makePred

ensemble_library_pred, validation_labels, scaler, model_grid = train_model_library(n_folds_to_compute=1)

ensemble, acc, n = ensemble.generate_ensemble(ensemble_library_pred, validation_labels, n_init = 0)

ids, features = util.load_test("kaggle_test_tf_idf.csv")
labels=makePred.makePrediction(ensemble,model_grid,features,scaler)

util.write_predictions(labels,"idflabels.csv")

print done
