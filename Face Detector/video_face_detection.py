import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv ( haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('Face Detector\haarcascade_frontalface_default.xml')

# to capture video from webcam
#webcam = cv2.VideoCapture(0)

webcam = cv2.VideoCapture(r'C:\Users\Manish\Complete AI\OpenCV\Face Detector\video.mp4')

### Iterate forever over frames
while True:

    ## Read from the current frame
    successful_frame_read, frame = webcam.read()

    # Must Convert to Grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # Draw rectangle aroung the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)),2)
  
    # Displaying video
    cv2.imshow(' Face Detector', frame ) 
    key = cv2.waitKey(1)

    ### Stop if Q is pressed
    if key ==81 or key ==113:
        break

### Release videoCapture object
webcam.release()

print("Code Completed")



