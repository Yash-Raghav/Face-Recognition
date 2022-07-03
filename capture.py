import cv2
import face_recognition as fr  #Primary library for face recognition
import os
import pandas as pd

def capture(name):
    '''This function captures the face, generates encodings for it and saves it into a csv file'''

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    file_name_path = name + '.jpg'

    count = 0 #for terminating the loop if this exceeds 0

    new_encoding = [] #generating encodings for the image
    while True:
        _ , img = cam.read() 
        faces = fr.face_locations(img) #Finding the co-ordinates of face in the frame

        for (top, right, bottom, left) in faces:
            cv2.rectangle(img, (left,top), (right,bottom), (255,0,0), 2) #Creates a bounding box on the face

            cv2.imshow('image', img)
            cv2.imwrite(file_name_path, img[top:bottom,left:right]) #Saving the image in our hard drive
            new_image = fr.load_image_file(file_name_path) #Loading the saved face image from above
            new_encoding = fr.face_encodings(new_image) #Generating encodings for the image
            if len(new_encoding)>0: #Sometimes face_recognition module is unable to get the encodings from an image hence we will run the code till it gets
                new_encoding=new_encoding[0] 
                count +=1

        if cv2.waitKey(1) & 0xFF==ord('q') or count==1: #Pressing "q" quits the frame capture
            break

    cam.release()
    cv2.destroyAllWindows()      
    print("Sample collected")

    data = pd.concat([pd.DataFrame([name]),pd.DataFrame([new_encoding])], axis=1) #Saving name and encodings into dataframe to later add to our csv database file

    data_path = "employee_data.csv"

    add_to_data = input("Add face data to database? Y/n \n") #Confirms whether we want to save the name and corresponding encodings in our database
    if add_to_data.lower() == "y":
        if not os.path.exists(data_path):
            data.to_csv(data_path, index=False)
        else:
            data.to_csv(data_path, mode='a', index=False, header=False)

capture(name = input("Enter your name: \n"))

