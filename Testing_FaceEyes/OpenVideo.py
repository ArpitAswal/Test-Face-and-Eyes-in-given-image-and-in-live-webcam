# Here we playing our saved video from any folder using OpenCV  
import numpy
import cv2

capture=cv2.VideoCapture(0) # if we give device index then it capture a video from live webcam and then we can perform various operation
 #                    OR we can give our saved video path
#r'C:\Users\Acer\Downloads\Telegram Desktop\blackclover.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480)) 
while(capture.isOpened()):
    ret,frame=capture.read()
    #for flip the output window screen
    
    if ret==True:
      frame = cv2.flip(frame,1)  # we can flip the scree by passing 0 instead of 1
      # write the flipped frame
      out.write(frame)
      cv2.imshow('frame',frame)
      gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # convert into gray scale image
      cv2.imshow('frame',gray)
     # while displaying the frame we should use appropriate time for waitKey(),as for too less video will be very fast or vice-versa
      if(cv2.waitKey(4) & 0xFF ==ord('q')):
        break
    else:
        break 
    
capture.release()
out.release()
cv2.destroyAllWindows()