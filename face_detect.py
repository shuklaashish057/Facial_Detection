import cv2
import numpy as np
import os 
import pyttsx3
import engineio

engineio = pyttsx3.init()
voices = engineio.getProperty('voices')
engineio.setProperty('rate', 130)  
voice_id = "hindi"
engineio.setProperty('voice',voice_id)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None','Ashish', 'Apurva', 'Ankit'] 
cam = cv2.VideoCapture(0)
cam.set(3, 1250) 
cam.set(4, 1250) 

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w]) 
        if True:
            id = names[id]
            cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
            break
        else:
            id = "unknown"
            cv2.putText(
                        img, 
                        str(id), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                    )
        
      
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break
    
engineio.say('Thank you {0}'.format(id))
engineio.runAndWait()
print("\n Thank you for the test.")
cam.release()
cv2.destroyAllWindows()








