import cv2

# Our Video
video = cv2.VideoCapture('Pedestrian.mp4')

# our Pre trained pedestrian classifier
pedestrian_tracker_file = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# create pedestrian classifier
pedestrian_tracker = cv2.CascadeClassifier('pedestrian_tracker_file')


#Run forever until car stops or something

while True:

    #Read the current frame
    (read_successful, frame) = video.read()

    # Safe coding
    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)

     # Draw rectangles around the cars
    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Display the image with the faces spotted
    cv2.imshow('Pedestrian Tracker', frame)

    # Dont autoclose ( wait here in the code and listen for a key to press )
    key = cv2.waitKey(3)

     ### Stop if Q is pressed
    if key ==81 or key ==113:
        break


print('Code Complete')