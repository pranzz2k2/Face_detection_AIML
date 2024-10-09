import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime, timedelta
import time
import platform
import subprocess

# Path to the folder containing student images
image_directory = 'Students'
image_data = []
student_names = []

# Load all student images and their names
image_files = os.listdir(image_directory)
print(image_files)

for img_file in image_files:
    img = cv2.imread(f'{image_directory}/{img_file}')
    if img is None:
        print(f"Image {img_file} not loaded correctly.")
        continue
    image_data.append(img)
    student_names.append(os.path.splitext(img_file)[0])
print(student_names)

# Function to encode faces in the images
def generate_face_encodings(images):
    encoding_list = []
    for image in images:
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(image)[0]
            encoding_list.append(encoding)
    return encoding_list

# Function to record attendance
def record_attendance(name):
    with open('Attendance.csv', 'r+') as file:
        data = file.readlines()
        names_list = [line.split(',')[0] for line in data]

        if name not in names_list:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            date_string = now.strftime('%d/%m/%Y')
            day_string = now.strftime('%A')
            file.writelines(f'\n{name},{date_string},{day_string},{time_string}')
            convert_csv_to_excel()

# Function to convert CSV file to Excel
def convert_csv_to_excel():
    df = pd.read_csv('Attendance.csv', names=['Name', 'Date', 'Day', 'Time'])
    unique_names = df['Name'].nunique()
    if unique_names >= 3:
        df.to_excel('Attendance.xlsx', index=False)
        print('Attendance.xlsx created.')
        open_excel_file('Attendance.xlsx')
        cv2.destroyAllWindows()
        video_capture.release()
        exit()

# Function to open the Excel file
def open_excel_file(filename):
    if platform.system() == "Darwin":       # macOS
        subprocess.call(('open', filename))
    elif platform.system() == "Windows":    # Windows
        os.startfile(filename)
    else:                                   # linux variants
        subprocess.call(('xdg-open', filename))
        
# Generate face encodings for known images
known_encodings = generate_face_encodings(image_data)
print('Encoding Complete')

# Start the webcam
video_capture = cv2.VideoCapture(0)
recognition_time_log = {}
session_start_time = time.time()

while True:
    success, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    scaled_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    scaled_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(scaled_frame)
    current_encodings = face_recognition.face_encodings(scaled_frame, face_locations)

    for encoding, face_location in zip(current_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index] and face_distances[best_match_index] < 0.5:
            student_name = student_names[best_match_index].upper()
        else:
            student_name = 'UNKNOWN'

        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.putText(frame, student_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 255, 255), 2)

        if student_name != 'UNKNOWN':
            current_time = datetime.now()
            if student_name in recognition_time_log:
                if current_time - recognition_time_log[student_name] >= timedelta(seconds=3):
                    record_attendance(student_name)
                    del recognition_time_log[student_name]
            else:
                recognition_time_log[student_name] = current_time

        current_time = datetime.now()
        recognition_time_log = {name: time for name, time in recognition_time_log.items() if
                                current_time - time < timedelta(seconds=3)}

    cv2.imshow('Webcam', frame)
    cv2.waitKey(1)

    if time.time() - session_start_time > 50:
        print("Time's up! Ending live capture.")
        break

# Release resources
cv2.destroyAllWindows()
video_capture.release()

# Generate Excel file if not done already
convert_csv_to_excel()



