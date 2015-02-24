import sklearn
import numpy as np

DTYPE = np.float64

def generate_ensemble(library, classes, tolerance = .001, max_iter = float('inf'), n_init = 5, bag_p = .5, n_be = 20):
    # Generates a vector of weights to dot product with individual model predictions. 
    # library is a matrix with the prediction of each model for each validation point. library[i, j] is the probability of the ith validation point being 1 according to the jth model
    # classes is a vector of 0/1 values specifying the clases of the validation points
    # tolerance sets the goal tolerance of the ensemble; models are added until the difference in validation error is less than tolerance
    # max_iter is the maximum number of models to add to each ensemble
    # n_init is the number of top models to use in the initialization step
    # bag_p is the proportion of models to pick for each ensemble build
    # n_be is the number of ensembles to build. The returned vector is the average of all ensembles.
    
    # bag_size is the number of models to select in each bagged ensemble
    bag_size = int(bag_p * library.shape[1])
    # an array that accumulates each bagged ensemble
    ensemble = np.zeros((library.shape[1], 1), dtype = DTYPE)
    
    # make sure classes is a column vector
    classes = classes.reshape((classes.size, 1)) 

    for i in range(n_be):
        # randomly choose bag_size models without replacement
        bag_idx = np.random.choice(library.shape[1], bag_size, False)
        bag = library[:, bag_idx]
        
        # create the initial weight vector by taking the n_init best performing models
        n_ensemble = n_init
        weights = np.zeros((bag_size, 1), dtype = DTYPE)
        if n_init > 0:
            init_idx = ((bag > .5) == classes).sum(axis = 0).argsort()[-n_init:] 
            weights[init_idx, 0] += 1
        
        # at each step, find the model that most improves the accuracy of the ensemble
        # classifier. This is done with replacement, as suggested in the literature.
        prev_acc = 0
        curr_acc = 0
        while True:
            n_ensemble += 1.
            candidate_acc = (((np.tile(bag.dot(weights), (1, bag_size)) + bag) / 
                    n_ensemble > .5) == classes).sum(axis = 0) / float(classes.size) 
            next_idx = candidate_acc.argmax()
            weights[next_idx, 0] += 1
            prev_acc = curr_acc
            curr_acc = candidate_acc[next_idx]
            
            if n_ensemble >= max_iter or curr_acc - prev_acc < tolerance:
                break

        ensemble[bag_idx, :] += weights / n_ensemble
    
    ensemble = ensemble / n_be
    
    ensemble_val_acc = ((library.dot(ensemble) > .5) == classes).sum() / float(classes.size)
    
    return ensemble.transpose() / n_be, ensemble_val_acc, n_ensemble
