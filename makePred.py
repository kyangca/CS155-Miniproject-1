import sklearn as sk
import numpy as np
from sklearn import preprocessing

def makePrediction(weights,models,features,scalar):
	transFeatures=scalar.transform(features)
	labels=[0 for x in range(len(features))]
	numCVFolds=len(models[0][1])
	
	wInd=0
	for w in weights.flatten():
		if(w!=0):
			model=models[wInd][1]
			for m in model:
					if(models[wInd][5]=="standardized"):
						labels+=w*m.predict_proba(transFeatures)[:, 1]
					
					else:
						labels+=w*m.predict_proba(features)[:, 1]
		wInd+=1
	
	labels=labels/numCVFolds
	
	labels=labels>=0.5*np.ones(len(labels))
	
	return labels
	
	

def evaluatePerformance(labels,predictions):
	norm=len(labels)
	fraction=0
	for i in range(norm):
		if(labels[i]==predictions[i]):
			fraction+=1
	
	fraction/=norm
	
	return fraction
		
