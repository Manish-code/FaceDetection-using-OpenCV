import cv2

# face and smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Grab webcam feed
webcam = cv2.VideoCapture(0)

while True:

    #Read current frame from webcam
    successful_frame_read, frame = webcam.read()

    # if there's an error, abort
    if not successful_frame_read:
        break

    # Change to grayscale
    frame_grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     # Show the current frame
    cv2.imshow('Why So Serious?', frame)

   
       

        # Display
        key = cv2.waitKey(1)

         #stop if Q key is pressed
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            break

   
        


    



#clean up
webcam.release()
cv2.destroyallWindows()