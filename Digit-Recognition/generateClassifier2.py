
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter
import sys


 datasets = np.empty(shape = (28, 0), dtype = 'int16')#datasets = np.array([])#np.load('./training_data/' + sys.stdin[0][0])
labels = np.array([])#np.zeros(dataset.size // (28*28), 'int') + int(sys.stdin[0][0])



for line in sys.stdin:


	dataset = np.load('./training_data/' + line.strip())
	label = np.zeros(dataset.size // (28*28), 'int') + int(line[0])

	datasets = np.concatenate((datasets, dataset))
	labels = np.concatenate((labels, label))


print(labels)

features = dataset.reshape((-1, 28*28))

# Extract the features and labels
# features = np.array(dataset.data, 'int16') 

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "digits_cls1.pkl", compress=3)