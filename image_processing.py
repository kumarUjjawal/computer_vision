import cv2
import numpy as np

# object tracking

# Take each frame of the video
# Convert from BGR to HSV color-space
# threshold the HSV image for a range of color
# Now extract the blue object alone, we can do whatever on that image we want.

# find HSV values to track
# green = np.uint8([[[0,255,0 ]]])                                                      
# hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
# print hsv_green

cap = cv2.VideoCapture(0)

while(1):
    _, frame = cap.read()
    # convert bgr to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in hsv
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # threshold the HSV image to get the only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #Bitwise AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
