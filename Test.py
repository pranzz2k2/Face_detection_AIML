import cv2
import numpy as np
import face_recognition
# Load and convert the reference image
reference_image = face_recognition.load_image_file('Testcase/Hrithik1.jpg')
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
# Load and convert the test image
test_image = face_recognition.load_image_file('Testcase/Hrithik3.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# Locate and encode the face in the reference image
reference_face_location = face_recognition.face_locations(reference_image)[0]
reference_face_encoding = face_recognition.face_encodings(reference_image)[0]
cv2.rectangle(reference_image, (reference_face_location[3], reference_face_location[0]), 
               (reference_face_location[1], reference_face_location[2]), (255, 0, 255), 10)
# Locate and encode the face in the test image
test_face_location = face_recognition.face_locations(test_image)[0]
test_face_encoding = face_recognition.face_encodings(test_image)[0]
cv2.rectangle(test_image, (test_face_location[3], test_face_location[0]), 
               (test_face_location[1], test_face_location[2]), (255, 0, 255), 10)
# Compare the faces and calculate the distance
comparison_result = face_recognition.compare_faces([reference_face_encoding], test_face_encoding)
face_distance = face_recognition.face_distance([reference_face_encoding], test_face_encoding)
print(comparison_result, face_distance)
# Display the results on the test image
cv2.putText(test_image, f'{comparison_result} {(100 - round(face_distance[0] * 100, 2))}%', 
            (50, 200), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 5)
# Show images
cv2.imshow('Reference Image', reference_image)
cv2.imshow('Test Image', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
