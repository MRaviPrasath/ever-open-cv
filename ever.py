#!/usr/bin/env python
# coding: utf-8

# In[1]: scanar


import cv2
import numpy as np
# cap = cv2.VideoCapture(0)               # 0 for webcam
# cap.set(3,640)                          # id 3 is width and 4 is height    # to set brightness, id is 10
# cap.set(4,480)

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres


def getContours(img):
    biggest = np.array([])
    maxArea=0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
                # cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                if area>maxArea and len(approx)==4:
                    biggest = approx
                    maxArea=area
    cv2.drawContours(imgContour,biggest,-1,(255,0,0),20)
    return biggest


def getWarp(img,biggest):
    pass



while True:
    # success, img= cap.read()
    img = cv2.imread(r"C:\raviprasath\Desktop\opencv\opencv try.png")
    # cv2.resize(img,(640,480))
    img = cv2.resize(img,(640,480))
    imgContour = img.copy()
    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    print(biggest)
    getWarp(img,biggest)
    cv2.imshow("Video",imgContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


# In[2]: Reading an image in OpenCV


import cv2
 
# To read image from disk, we use
# cv2.imread function, in below method,
img = cv2.imread("geeksforgeeks.png", cv2.IMREAD_COLOR)
 
# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
cv2.imshow("image", img)
 
# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.waitKey(0)
 
# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()


# In[3]: Display an image in OpenCV using Python


import cv2
  
# path
path = r'Users\raviprasath\Desktop\geeksforgeeks.png'
  
# Reading an image in default mode
image = cv2.imread(path)
  
# Window name in which image is displayed
window_name = 'image'
  
# Using cv2.imshow() method
# Displaying the image
cv2.imshow(window_name, image)
  
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows(


# In[ ]: Resizing image using Open CV


import cv2
import numpy as np
import matplotlib.pyplot as plt
 
image = cv2.imread(r"D:\sims\eb\sim21\EB-ML-06-10-2022-Test-Output-15\PERFORATION\Overkill\Fail\Blister 1 2022-03-12 12-59-43.859 T0 M0 G0 3 PERFORATION Mono.bmp", 1)
# Loading the image
 
half = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
bigger = cv2.resize(image, (1050, 1610))
 
stretch_near = cv2.resize(image, (780, 540),
               interpolation = cv2.INTER_LINEAR)
 
 
Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
images =[image, half, bigger, stretch_near]
count = 4
 
for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(images[i])
 
plt.show()
Output: 

