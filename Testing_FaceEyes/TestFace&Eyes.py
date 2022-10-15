import cv2
import os
#cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
#cascPatheyes = os.path.dirname(cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(r'C:\Users\Acer\Documents\Testing_FaceEyes\haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier(r'C:\Users\Acer\Documents\Testing_FaceEyes\haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier(r'C:\Users\Acer\Documents\Testing_FaceEyes\haarcascade_smile.xml')
# above we give face, eyes and smile haarcascade files by passing path in CascadeClassifier

# create a function to detect face and it draw the rectangle in co-ordinates of face on  given image
def detect_face(img):
     
    face_img = img.copy()
     
    face_rect = faceCascade.detectMultiScale(face_img,
                                              scaleFactor = 1.2,
                                              minNeighbors = 5)
     
    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y),
                      (x + w, y + h), (255, 255, 255), 10)
         
    return face_img

# create a function to detect eyes and it draw the circle in co-ordinates of eyes on  given image
def detect_eyes(img):
    eye_rect = eyeCascade.detectMultiScale(img,
                                            scaleFactor = 1.2,
                                            minNeighbors = 5)   
    for (x, y, w, h) in eye_rect:
        
        radius = int(round((w + h) * 0.20))
        cv2.circle(img, (x+w//2,y+h//2), radius, (255, 255, 255), 7)       
    return img  


img=cv2.imread(r'C:\Users\Acer\Documents\Resume_pic.jpg',1)
# give the path of image in which you want to detect face and eyes
img_resize=cv2.resize(img,(780,540),interpolation=cv2.INTER_NEAREST)
#eyes = detect_eyes(img_resize)
#cv2.imshow('eyes',eyes)
#face = detect_face(img_resize)
#cv2.imshow('face',face)

face_eyes=detect_face(img_resize)
face_eyes=detect_eyes(face_eyes)
# by above two operations it draw the rectangle on face and circle on eyes on given image
cv2.imshow('face & eyes',face_eyes)
cv2.waitKey()
def detect(gray,frame):
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 18)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        '''smiles = smile_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)'''    
    
video_capture = cv2.VideoCapture(0) #it open the webcam
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()  # read frame by frame from webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert into grayscale 
    detect(gray, frame)
    
    '''for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        
        eyes = eyeCascade.detectMultiScale(frame,
                                            scaleFactor = 1.3,
                                            minNeighbors = 5)
        for (x2, y2, w2, h2) in eyes:
            #eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            
             radius = int(round((w2 + h2) * 0.3))
             cv2.circle(frame, (x2+20,y2+25), radius, (255, 255, 255), 5)'''
     
      # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()