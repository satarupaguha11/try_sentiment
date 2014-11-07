from sklearn import svm
from sklearn.svm import LinearSVC
import scipy.io
from sklearn.externals import joblib
from numpy import *
from timeit import default_timer

start = default_timer()
clf = svm.LinearSVC(C=0.4,class_weight='auto',dual=False)
train_features = scipy.io.loadmat('../data/features/train_features.mat')
X = train_features['data']
Y = train_features['labels']
#print Y
n_samples = Y.size
print n_samples
Y = reshape(Y,(n_samples,))
clf.fit(X, Y)
joblib.dump(clf, '../data/models/trained_model.pkl')
end = default_timer()
print 'time taken to run classifier:'+str(end-start)