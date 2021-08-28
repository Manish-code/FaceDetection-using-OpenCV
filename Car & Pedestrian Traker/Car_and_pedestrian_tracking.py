import cv2

# Our Image
img_file = r'C:\Users\Manish\Complete AI\OpenCV\Car & Pedestrian Traker\car5.png'

# Our pre-trained car classifier
classifier_file = 'car_detector.xml'

# create opencv image
img = cv2.imread(img_file)

#create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# convert to grayscale ( needed for cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)



#Display thge image with faces spotted
cv2.imshow('Car Detector', img)

# Dont autoclose ( wait here in the code and listen for a key to press )
cv2.waitKey()



print('Code Complete')