# Face-Recognition

Aim of the project is to create a rudimentary face detection and recognition model which detects when a face is in the view and recognises the identity of the user based on the face encodings saved in the database.

The project utilises [face-recognition](https://pypi.org/project/face-recognition/) library to encode the dacial features.

1.	“Capture” file detects the face and saves the image into the hard-drive while also creating a csv file (as database to be compared for recognising) 
    with the name and encodings updated of the facial image captured.
2.	“Detect” file detects and recognises the face and displays the name on-screen.
