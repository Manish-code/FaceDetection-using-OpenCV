import cv2

# Our Image
video = cv2.VideoCapture('car video.mp4 ')

# Our pre-trained car classifier
classifier_file = 'car_detector.xml'

#create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

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
    cars = car_tracker.detectMultiScale(grayscaled_frame)

     # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Display the image with the faces spotted
    cv2.imshow('Car Detector', frame)

    # Dont autoclose ( wait here in the code and listen for a key to press )
    cv2.waitKey(3)


    

















print('Code Complete')