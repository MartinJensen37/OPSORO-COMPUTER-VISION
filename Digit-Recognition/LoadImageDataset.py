import numpy as np
from time import sleep
from sklearn import datasets
import cv2



# Load the dataset
dataset = datasets.fetch_mldata("MNIST Original", data_home = 'C:\Python36\Lib\site-packages\Datasets')


#frame = []
#frame[:28][:28] = dataset.data[:784]


frame =  (dataset.data[0][:784])
frame.resize((28,28))


for i in range(1,10):

    for j in range (70000):
        if dataset.target[j] == i:
            print (j)
            stuff = (dataset.data[j][:784])
            stuff.resize((28,28))
            frame = np.concatenate((stuff, frame))
            print (dataset.target[j])

            break


# Extract the features and labels
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')



#cv2.namedWindow('image')
frame = cv2.resize(frame, (50, 500))
cv2.imshow('frame', frame)



#for feature in features:
#    feature = feature.reshape(28,28)
#
#    print(feature)
#    sleep(10)


    


