import face_recognition  
import cv2
  
image = face_recognition.load_image_file("Parth/7.jpg")  
  
face_locations = face_recognition.face_locations(image)  

for face_location in face_locations:  
    top, right, bottom, left = face_location  
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  
  
cv2.imshow("Faces", image)  
cv2.waitKey(0)  