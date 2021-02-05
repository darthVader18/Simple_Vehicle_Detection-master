# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV 
import cv2
 
# Capture frames from a video
cap = cv2.VideoCapture('video.avi')
# cap = cv2.VideoCapture('video_2.avi')
# cap = cv2.VideoCapture('video_3.avi')
 
# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')
 
# Loop runs if capturing has been initialized.
while True:
    # Reads frames from a video
    ret, frames = cap.read()
     
    # Convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) # Converting BGR to grayscale 

    # cv2.cvtColor() method is used to convert an image from one color space to another

    # RGB stands for Red Green Blue. Most often, an RGB color is stored in a structure or 
    # unsigned integer with Blue occupying the least significant "area" (a byte in 32-bit 
    # and 24-bit formats), Green the second least, and Red the third least.

    # BGR is the same, except the order of areas is reversed. Red occupies the least significant
    #  area, Green the second (still), and Blue the third.
     
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
     
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
    
    # cv2.rectangle() method is used to draw a rectangle on any image
    # Here in line 35 this - (0,0,255) is for red colour rectangles
 
   # Display frames in a window 
    cv2.imshow('video2', frames)

    # cv2.imshow() method is used to display an image in a window 
    # The window automatically fits to the image size
     
    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break
 
# De-allocate any associated memory usage
cv2.destroyAllWindows()
