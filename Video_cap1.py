import numpy as np
import cv2
from sklearn.externals import joblib
from skimage.feature import hog


# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Default camera has index 0 and externally(USB) connected cameras have
# indexes ranging from 1 to 3
cap = cv2.VideoCapture(0)
persistence = []

while(True):

    
    # Capture frame-by-frame
    ret, frame = cap.read()
  
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
   

    
    im_th, contours0, hierarchy  = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    area = [cv2.contourArea(cnt) for cnt in contours0]

    # hull = [cv2.convexHull(cnt) for cnt in contours0]

    # Draw contours in the original image 'im' with contours0 as input

    # cv2.drawContours(frame, contours0, -1, (0,0,255), 2, cv2.LINE_AA, hierarchy, abs(-1))
    

    # Rectangular bounding box around each number/contour
    rects = [cv2.boundingRect(cnt) for cnt in contours0]
    
    newpersistence = []
    # Draw the bounding box around the numbers(Making it visual)
    for rect, i in zip(rects, range(0, len(rects))):
                   
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            
        
        moments = [cv2.moments(cnt) for cnt in contours0]
        
        #Check if any regions were found
        if roi.any() and rect[3] < 200 and rect[3] > 30 and rect[2] < 250 and rect[2] > 5:
            # Draw rectangles
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            # Resize the image
            roi = cv2.resize(roi, (28, 28), im_th, interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            roi = cv2.erode(roi, (3, 3))
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            
            
            nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
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
    
    
    ##############
    #Tracking And Decision
    ##############
    slack = 30
    #how many pixels the contur can have moved where it will still be recocnised as the same contour
    old_weight = 0.8
    #proportion of the old confidence value to reuse
    new_weight = 0.2
    #proportion of the new confidence value to include
    time_to_live = 10
    #how many frames may pass without a stored contour being found before we delete it
    penalty = 2
    #penalty for new contours
    
    #new[:9] contains the probabilities
    #new[10] is the position
    #old[11] is the time to live
    
    
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
                old[10] = new[10]
                old[11] = time_to_live
                break #break so we don't update the time to live on multiple contours 
        #otherwise, append it
        if inserted == 0:
            #persistence.append(new + [time_to_live])
            persistence.append([x - penalty for x in new[:10]]+ [new[10]] + [time_to_live])
            print([x - penalty for x in new[:10]]+ [new[10]] + [time_to_live])
            print (new + [time_to_live])
    
    print (len(persistence))
    
    for old in persistence:
        old[11] -= 1
        #decrement time to live
    persistence = [x for x in persistence if x[11] > 0]
        #remove if time to live is 0
    
    for old in persistence:
        for j in range(10):
            cv2.putText(frame, str(int(old[j]*100)), (old[10][0]-20,old[10][1]+10*j), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
        if max(old[:10])> 0:
            cv2.putText(frame, str(int(old.index(max(old[:10])))), (old[10][0], old[10][1]),cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)
        #draw numbers on the image
    
    
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('Threshold', im_th)
    
    



    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
    
