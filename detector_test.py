import threading
import cv2
import os
from deepface import DeepFace

#variables
face_counter = 0
counter = 0
face_match = False
reference_img = cv2.imread("reference.jpg")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Create the Video Capture using the Webcam
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Directory to save new faces
output_dir = "detected_faces"
os.makedirs(output_dir, exist_ok=True)

#Cgecks the frame to see if its reference
def check_frame(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())["verified"]:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


# Draw bounding boxes around detected faces
def draw_detection_box():
    for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Save the detected face to a file
            face_img = frame[y:y + h, x:x + w]
            save_face(face_img)


#save new faces
def save_face(face_img):
    global face_counter
    face_filename = os.path.join(output_dir, f"face_{face_counter}.jpg")
    print(f"Saving face to {face_filename}")  # Debug statement
    success = cv2.imwrite(face_filename, face_img)
    if success:
        print(f"Successfully saved {face_filename}")
    else:
        print(f"Failed to save {face_filename}")
    face_counter += 1

#Main loop
while True:
    revalue, frame = capture.read()
    if revalue:
        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check the frame for face match every 30 frames
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_frame, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        # Draw bounding boxes around detected faces and save new faces
        draw_detection_box()

        # Display match status
        if face_match:
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Match!", (20, 450), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
