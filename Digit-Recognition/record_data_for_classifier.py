#Videocap for RPi including decision function and contour stuff.
import numpy as np
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
from time import sleep
import time

from picamera.array import PiRGBArray
from picamera import PiCamera
import math

# Load the classifier
#clf = joblib.load("digits_cls.pkl")

# Default camera has index 0 and externally(USB) connected cameras have
# indexes ranging from 1 to 3
#cap = cv2.VideoCapture(0)


# Defining the camera parameters: frame rate and resolution
cap = PiCamera()
cap.resolution = (640, 480)
cap.framerate = 32
im = PiRGBArray(cap, size=(640, 480))


def nothing(x):
    pass

cap.capture(im, format="bgr", use_video_port=True)
sleep(0.1)
_, frame = cap.capture(im, format="bgr", use_video_port=True)

    
#for frame in cap.capture_continuous(im, format="bgr", use_video_port=True):
image = frame.array

# Capture frame-by-frame
#ret, frame = cap.read()
#frame = cap.capture_continuous(im, format="bgr", use_video_port=True)[0]

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)


# Threshold the image
# ret, im_th = cv2.threshold(im_gray.copy(), 120, 255, cv2.THRESH_BINARY_INV)

im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

im_th = cv2.medianBlur(im_th, 7)

# Find contours in the binary image 'im_th'

im_th = cv2.dilate(im_th, np.ones((3,3), np.uint8))
im_th = cv2.erode(im_th, np.ones((3,3), np.uint8))

im_th = cv2.morphologyEx(im_th, cv2.MORPH_ELLIPSE, (7,7))
im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))



im_th, contours0, hierarchy  = cv2.findContours(im_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# area = [cv2.contourArea(cnt) for cnt in contours0]

# hull = [cv2.convexHull(cnt) for cnt in contours0]

# Draw contours in the original image 'im' with contours0 as input
#stretch = cv2.getTrackbarPos('stretch','frame') /100+1
stretch = 1.5



newpersistence = []

for i in range(len(contours0)):
    dims = cv2.boundingRect(contours0[i])
    if hierarchy[0][i][3]==-1 and dims[3] < 200 and dims[3] > 20 and dims[2] < 250 and dims[2] > 3:
        
        im1 = np.zeros((int(dims[3]), int(dims[2])), np.uint8)
        
        cont = contours0[i] -[[dims[0], dims[1]]]
            #cv2.fillPoly(im, cont, (255))
        cv2.drawContours(im1, contours0, i, (255), cv2.FILLED, cv2.LINE_AA, hierarchy, 1, (-dims[0], -dims[1]))

            
        #if i == 0:
        #    cv2.imshow('im', im1)
        
        #dims[2] =int(dims[2]*stretch)
        
        dims = (dims[:2]+ (int(dims[2]*stretch),) + dims[3:])
        
        
        im1 = cv2.resize(im1, (dims[2], dims[3]), cv2.INTER_AREA)
        maxsize = max(dims[2], dims[3])
        
        im2 = cv2.copyMakeBorder(im1 ,maxsize - dims[3],maxsize - dims[3],maxsize - dims[2],maxsize - dims[2],cv2.BORDER_CONSTANT,value=[0])
        
        im2 = cv2.dilate(im2, np.ones((maxsize//20,maxsize//20), np.uint8))
        
        #im2 = cv2.GaussianBlur(im2, (5, 5), 0)
        
        im2 = cv2.resize(im2, (24, 24), cv2.INTER_AREA)
        im2 = cv2.copyMakeBorder(im2, 2, 2, 2, 2, cv2.BORDER_CONSTANT,value=[0])
        
        im2 = cv2.GaussianBlur(im2, (3, 3), 0)
        
        #if i == 0:
        #    cv2.imshow('im2', im2)
        
        """
        roi_hog_fd = hog(im2, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        
        
        #nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        #cv2.putText(frame, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)
        test = clf.decision_function(np.array([roi_hog_fd], 'float64'))
        prob = test.tolist()[0]
        prob.append(dims)
        newpersistence.append(prob)
        """
        
        cv2.rectangle(image, (dims[0], dims[1]), (dims[0] + dims[2], dims[1] + dims[3]), (0, 255, 0), 3)
        
        newpersistence.append(im2)
        
        

        
        
        
        
        
        
cv2.imwrite("../OPSORO/OS/src/opsoro/apps/testapp/static/images/example.JPEG", image)
result = newpersistence[0]
for digits in newpersistence[1:]:
    result = np.concatenate((result, digits))
    
    
cv2.imwrite("../OPSORO/OS/src/opsoro/apps/testapp/static/images/data.JPEG", result)
im.truncate(0)