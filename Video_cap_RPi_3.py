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
clf = joblib.load("digits_cls.pkl")

# Default camera has index 0 and externally(USB) connected cameras have
# indexes ranging from 1 to 3
#cap = cv2.VideoCapture(0)


# Defining the camera parameters: frame rate and resolution
cap = PiCamera()
cap.resolution = (640, 480)
cap.framerate = 32
im = PiRGBArray(cap, size=(640, 480))



persistence = []

#_, testframe = cap.read()
#cv2.imshow('frame', testframe)

def nothing(x):
    pass
"""
cv2.createTrackbar('slack','frame',30,100,nothing)
cv2.createTrackbar('old_weight','frame',80,100,nothing)
cv2.createTrackbar('new_weight','frame',20,100,nothing)
cv2.createTrackbar('time_to_live','frame',10,50,nothing)
cv2.createTrackbar('new_penalty','frame',0,1000,nothing)
cv2.createTrackbar('invis_penalty','frame',0,1000,nothing)
cv2.createTrackbar('min_confidence','frame',300,1000,nothing)
cv2.createTrackbar('stretch','frame',50,100,nothing)
"""
noofframes = 0
endtime = time.time()+60


#while(True):
for frame in cap.capture_continuous(im, format="bgr", use_video_port=True):
    noofframes =+ 1
    if endtime =< time.time():
        break
    
    
    
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
            
            
            roi_hog_fd = hog(im2, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            
            
            #nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
            #cv2.putText(frame, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)
            test = clf.decision_function(np.array([roi_hog_fd], 'float64'))
            prob = test.tolist()[0]
            prob.append(dims)
            newpersistence.append(prob)
            
            
    
    """
    #for cnt in contours0:
    
        #dims = cv2.boundingRect(cnt)
        #roi = im_th[dims[0]:dims[0]+dims[2], dims[1]:dims[1]+dims[3]]
        #cv2.drawContours(im, cnt, -1, (0,0,255), 2, cv2.LINE_AA, 0, abs(-1))
        
        #dims = cv2.boundingRect(cnt)
        #im = np.zeros((int(dims[2] * 1.6), int(dims[3]*1.6),3), np.uint8)
        #cv2.fillPoly(im, cnt-[[dims[0], dims[1]]], (255,0,255))
        #cv2.imshow('im', im)
        #print(cnt)
        #sleep(5)
    
    
    
    #cv2.fillPoly(frame, contours0, (255,0,0))
    
    # print (len(contours0[0]))
    """
    """
    # Rectangular bounding box around each number/contour
    rects = [cv2.boundingRect(cnt) for cnt in contours0]
    
    
    
    
    
    
    # Draw the bounding box around the numbers(Making it visual)
    for rect, i in zip(rects, range(0, len(rects))):
                   
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            
        
        # moments = [cv2.moments(cnt) for cnt in contours0]
        
           
            
        # Check if any regions were found
        if roi.any() and rect[3] < 200 and rect[3] > 30 and rect[2] < 250 and rect[2] > 5:
            # Draw rectangles
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            # Resize the image
            roi = cv2.resize(roi, (28, 28), im_th, interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            roi = cv2.erode(roi, (3, 3))
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            
            
            #nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
            #cv2.putText(frame, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)
            test = clf.decision_function(np.array([roi_hog_fd], 'float64'))
            prob = test.tolist()[0]
            #print (test)
            #for j in range(9):
                #print(str(prob[j]))
                
            #    cv2.putText(frame, str(int(prob[j]*100)), (rect[0]-20,rect[1]+10*j), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
            prob.append(rect)
            newpersistence.append(prob)
            
            
    #print (newpersistence)
    """
    
    ##############
    #Tracking And Decision
    ##############
    slack = 30
    #how many pixels the contur can have moved where it will still be recocnised as the same contour
    old_weight = 0.8
    #proportion of the old confidence value to reuse
    new_weight = 0.2
    #proportion of the new confidence value to include
    time_to_live = 5
    #how many frames may pass without a stored contour being found before we delete it
    new_penalty = 2
    #penalty for new contours
    invis_penalty = 0.2
    #penalty if no new contour matches up with a stored one
    min_confidence = -2.5
    #confidence required to recognise a digit
  
    #new[:10] contains the probabilities
    #new[10] is the position
    #old[11] is the time to live
    """
    slack          = cv2.getTrackbarPos('slack','frame')        
    old_weight     = cv2.getTrackbarPos('old_weight','frame')   /100
    new_weight     = cv2.getTrackbarPos('new_weight','frame')   /100
    time_to_live   = cv2.getTrackbarPos('time_to_live','frame') 
    new_penalty    = cv2.getTrackbarPos('new_penalty','frame')  /100
    invis_penalty  = cv2.getTrackbarPos('invis_penalty','frame')/100
    min_confidence = cv2.getTrackbarPos('min_confidence','frame')/100 - 5
    """
    
    
    
    for new in newpersistence:
        #go through each of hte contours captured this frame
        inserted = 0
        for old in persistence:
            #see if they match up with any of the existing conturs
            #it stops at the first one that is wihtin the slack
            #instead of finding the closest one
            if new[10][0] < old[10][0] +slack and new[10][0] > old[10][0] -slack and new[10][1] < old[10][1]+slack and new[10][1] > old[10][1] -slack:
                #if it matches, update it
                inserted = 1
                old[:10] = [x * old_weight + y * new_weight for x,y in zip(old[:10], new[:10])]
                old[10] = new[10]#position
                old[11] = time_to_live
                break #break so we don't update the time to live on multiple contours 
        #otherwise, append it
        if inserted == 0:
            #persistence.append(new + [time_to_live])
            persistence.append([x - new_penalty for x in new[:10]]+ [new[10]] + [time_to_live])
            #penalty for new contours
            #print([x - new_penalty for x in new[:10]]+ [new[10]] + [time_to_live])
            
    #print (len(persistence))
    
    for old in persistence:
        old[11] -= 1
        #decrement time to live
        if old[11] < 9:
            old[:10] = [x - invis_penalty for x in old[:10]]
        
    persistence = [x for x in persistence if x[11] > 0]
        #remove if time to live is 0
    
    for old in persistence:
        #print confidence
        #for j in range(10):
            #cv2.putText(image, str(int(old[j]*100)), (old[10][0]-20,old[10][1]+10*j), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
        
        #print result if certain
        if max(old[:10])> min_confidence:
            cv2.putText(image, str(int(old.index(max(old[:10])))), (old[10][0], old[10][1]),cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)
        #draw numbers on the image
    
    
    
    # Display the resulting frame
    #cv2.imshow('frame2', frame)
    #cv2.imshow('Threshold', im_th)
    
    
    # Save image to folder that connected to the OPSORO website
    cv2.imwrite("../OPSORO/OS/src/opsoro/apps/testapp/static/images/example.JPEG", image)

    im.truncate(0)


    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print (noofframes)