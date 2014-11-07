from sklearn.externals import joblib
from sklearn import svm
import scipy.io
from numpy import *

model = joblib.load('../data/models/trained_model.pkl')
feature_file=scipy.io.loadmat('../data/features/test_features.mat')
X = feature_file['data']
Y = feature_file['labels']
#print X.shape, Y.shape
#print Y
predicted_labels = model.predict(X)
accuracy = 0
#print unique(predicted_labels)

for i in range(len(predicted_labels)):
	#print predicted_labels[i],Y[i]
	if predicted_labels[i] == Y[i][0]:
		#print predicted_labels[i],Y[i][0]
		accuracy+=1
#print accuracy
#print len(predicted_labels)
print accuracy/float(len(predicted_labels))