import cv2

# Our Image
video = cv2.VideoCapture('Pedestrian.mp4 ')

# Our pre-trained car classifier
pedestrian_classifier_file = 'haarcascade_fullbody.xml'
car_classifier_file = 'car_detector.xml'
#create classifier
pedestrian_classifier = cv2.CascadeClassifier(pedestrian_classifier_file)
car_classifier = cv2.CascadeClassifier(car_classifier_file)
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

    # detect cars and pedestrians
    cars = car_classifier.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_classifier.detectMultiScale(grayscaled_frame)

     # Draw rectangles around the cars and pedestrian
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Display the image with the faces spotted
    cv2.imshow('Car and Pedestrian Detector', frame)

    # Dont autoclose ( wait here in the code and listen for a key to press )
    key = cv2.waitKey(1)

     ### Stop if Q is pressed
    if key ==81 or key ==113:
        break


    

















print('Code Complete')