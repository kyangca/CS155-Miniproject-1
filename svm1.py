import sklearn as sk
import numpy as np
from sklearn import cross_validation as cval
from sklearn.svm import SVC

	
def trainSVM(xTrain,yTrain,err=1.0,kern='linear'):
	alg=SVC(C=err,kernel=kern,probability=True)
	alg.fit(xTrain,yTrain)
	return alg
	
def crossValidateSVM(X,Y,kernel,err=1.0):
	fiveFold = cval.KFold(len(Y), n_folds=5)
	algArr=[]
	valArr=[]
	for trainInd, testInd in fiveFold:
		xTrain, yTrain = X[trainInd] , Y[trainInd]
		xTest, yTest = X[testInd], Y[testInd]
		alg=trainSVM(xTrain,yTrain,err,kernel)
		valArr.append(alg.predict_proba(xTest))
		algArr.append(alg)
		
	return algArr, valArr
	
def populateMatrix(X,Y):
	probMat=np.array([[]])
	modelMat=np.array([[]])
	for i in range(1,11):
		err=i/4.0
		[algArr, valArr]=crossValidateSVM(X,Y,"linear",err)
		valArr=np.array([valArr])
		algArr=np.array([algArr])
		if(i==1):
			probMat=valArr
			modelMat=algArr
			
		else:
			probMat=np.append(probMat,valArr,axis=0)
			modelMat=np.append(modelMat,algArr,axis=0)
	
	return probMat, modelMat
		
		
