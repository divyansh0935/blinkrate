# blinkrate
import cv2 

import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


blink_count = 0
eyes_open = True
consecutive_closed_frames = 0
consecutive_open_frames = 0
threshclose = 2
threshopen = 2


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detect = False
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 3)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)
        
        if len(eyes) >= 2:
            detect = True
    
    if detect:
        consecutive_open_frames += 1
        consecutive_closed_frames = 0
        if consecutive_open_frames > threshopen and not eyes_open:
            eyes_open = True
    else:
        consecutive_closed_frames += 1
        consecutive_open_frames = 0
        if consecutive_closed_frames >= threshclose and eyes_open:
            eyes_open = False
            blink_count += 1
    
    cv2.putText(frame, f"Blinks: {blink_count}", (40,80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
    
    cv2.imshow('blink-counter', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
