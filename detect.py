import numpy as np
import cv2
import face_recognition as fr
import pandas as pd

cam = cv2.VideoCapture(0)

employee_data = pd.read_csv("employee_data.csv", index_col=False) #reading our employee face database

#gathering encodings for all employees in a list to compare 
encodings_list = []
for employee in range(len(employee_data)):
    data = employee_data.iloc[employee,1:]
    data = np.asarray(data, dtype="float64")
    encodings_list.append(data)

known_face_encodings = encodings_list
known_face_names = [encoding for encoding in employee_data.iloc[:,0]] #first column of dataframe is name from where we are capturing employee names in a list

while True:
    _, frame = cam.read()

    face_locations = fr.face_locations(frame) #Face co-ordinates of person currently in front of webcam
    face_encodings = fr.face_encodings(frame, face_locations) #Generated encodings for the person

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face_encodings, face_encoding, tolerance=0.5) #compares encodings, whether found or not
        name = "Unknown person" #If the model is unable to find the name in database

        face_distances = fr.face_distance(known_face_encodings, face_encoding) #Finds the euclidean distance between the person and known encodings

        best_match_index = np.argmin(face_distances) #Matching the person with minimum euclidean distance with known encodings
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2) #Bounding box for the face

        # cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv2.FILLED) #Bounding box for the name tag
        font = cv2.FONT_HERSHEY_COMPLEX

        cv2.putText(frame, name, (left +6, bottom-6), font, 1.0, (255,255,255),1)

    cv2.imshow('face_recogniser', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'): 
        break

cam.release()
cv2.destroyAllWindows()