import numpy as np
import ensemble
from train_model_library import train_model_library
import util

ensemble_library_pred, validation_labels, scaler, model_grid = train_model_library(n_folds_to_compute=1)

ensemble, acc, n = ensemble.generate_ensemble(ensemble_library_pred, validation_labels, n_init = 5)

print done
