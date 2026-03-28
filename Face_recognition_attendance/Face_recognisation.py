import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Load images from dataset
path = 'dataset'
images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Mark attendance in CSV
def markAttendance(name):
    file_name = 'attendance.csv'
    
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(file_name, index=False)

    df = pd.read_csv(file_name)

    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    # Check if already marked today
    if not ((df['Name'] == name) & (df['Date'] == date)).any():
        new_entry = pd.DataFrame([[name, date, time]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(file_name, index=False)
        print(f"Attendance marked for {name}")

# Encode known faces
encodeListKnown = findEncodings(images)
print("Encoding Complete")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, name, (x1,y2+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

            markAttendance(name)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()