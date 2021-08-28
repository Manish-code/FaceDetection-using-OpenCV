import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv ( haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('Face Detector\haarcascade_frontalface_default.xml')

# Choose an image to detect face in
img = cv2.imread(r"C:\Users\Manish\Complete AI\OpenCV\Face Detector\modi.png")

# Must Convert to Grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


# Draw rectangle aroung the faces

(x,y,w,h) = face_coordinates[0]
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)),2)
 
# Displaying image

cv2.imshow(' Face Detector', img ) 
cv2.waitKey()

print("Code Completed")