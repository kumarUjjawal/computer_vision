
# coding: utf-8

# ### Getting Started With Videos

# In[1]:

import numpy as np
import cv2


#Videos from camera

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# playing Videos from file

Videos from file

cap = cv2.VideoCapture('',)

while (True):
    ret, frame = cv2.read()

    gray = cv2.cvtColor(frame, cv2,COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# capture and save video file

# Drawing shapes

# create black image
img = np.zeros((512,512,3), np.int8)

#line
line = cv2.line(img, (0,0), (511,511),(255,0,0), 5)
#rectangle
rect = cv2.rectangle(img, (384,0), (512,128), (0,255,0), 2)
#circle
circle = cv2.circle(img,(447,63), 63, (0,0,255), -1)

# adding fonts
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)


cv2.imshow('line',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
