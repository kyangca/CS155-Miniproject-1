import sklearn as sk
import numpy as np
from sklearn import preprocessing

def makePrediction(weights,models,features,scalar,numCVFolds=5):
	transFeatures=scalar.transform(features)
	labels=[]
	fInd=0
	
	for x in transFeatures:
		labels.append(0)
		wInd=0
		for w in weights.flatten():
			if(w!=0):
				model=models[wInd][1]
				for m in model:
					if(models[wInd][5]=="standardized"):
						labels[fInd]+=w*m.predict_proba(x)[0, 1]
					
					else:
						labels[fInd]+=w*m.predict_proba(features[fInd])[0, 1]
				
			
			wInd+=1
			
		labels[fInd]/=numCVFolds
		if(labels[fInd]>=0.5):
			labels[fInd]=1
		
		else:
			labels[fInd]=0
		
		fInd+=1

	return labels

def evaluatePerformance(labels,predictions):
	norm=len(labels)
	fraction=0
	for i in range(norm):
		if(labels[i]==predictions[i]):
			fraction+=1
	
	fraction/=norm
	
	return fraction
		
