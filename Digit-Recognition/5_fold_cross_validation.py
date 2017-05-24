from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter

confusion = np.zeros((10,10))

for testnum in range(5):
    datasets = np.empty(shape = (0, 28), dtype = 'int16')
    labels = np.array([], 'int')
    for digit in range(10):
        for sheetnum in range(10):
            if sheetnum != testnum:
                dataset = np.load('./training_data/' + str(digit) + '_' + str(sheetnum) + '.npy')
                label = np.zeros(dataset.size // (28*28), 'int') + digit
                datasets = np.concatenate((datasets, dataset))
                labels = np.concatenate((labels, label))
	
    features = datasets.reshape((-1, 28*28))
    list_hog_fd = []
    for feature in features:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')

    # Create an linear SVM object
    clf = LinearSVC()

    # Perform the training
    clf.fit(hog_features, labels)
    
    for digit in range(10):
        dataset = np.load('./training_data2/' + str(digit) + '_' + str(testnum) + '.npy')
        features = dataset.reshape((-1, 28*28))
        for feature in features:
            roi_hog_fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
            confusion[digit][nbr] = confusion[digit][nbr] + 1
            
print (confusion)