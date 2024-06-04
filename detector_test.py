import threading
import cv2
import os
from deepface import DeepFace
import pickle
from pathlib import Path
from imutils import paths
import face_recognition
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
##################################################CODE TO CAPTURE THE FACES AND SAVE THEM################################################
#variables
face_counter = 0
counter = 0
face_match = False
reference_img = cv2.imread("reference.jpg")
image_dir  = Path("faces_dir")

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
def draw_detection_box(faces,frame):
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
def video_detection():
	global counter
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
			draw_detection_box(faces,frame)

			# Display match status
			if face_match:
				cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 255, 0), 3)
			else:
				cv2.putText(frame, "No Match!", (20, 450), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 3)

			cv2.imshow("video", frame)

		key = cv2.waitKey(1)
		if key == ord("q"):
			break
	
	capture.release()
	cv2.destroyAllWindows()
##########################################################################################################################################


##########################################################CODE to Encode the detected and saved faces#####################################
def encode_faces():
	imagePaths = list(paths.list_images("detected_faces"))
	data = []
	# loop over the image paths
	for (i, imagePaths) in enumerate(imagePaths):
		# load the input image and convert it from RGB (OpenCV ordering)
		# to dlib ordering (RGB)
		print("[INFO] processing image {}/{}".format(i + 1,
			len(imagePaths)))
		print(imagePaths)
		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		image = cv2.imread(imagePaths)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		boxes = face_recognition.face_locations(rgb,model="HOG")    
		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(rgb, boxes)
		# build a dictionary of the image path, bounding box location,
		# and facial encodings for the current image
		d = [{"imagePath": imagePaths, "loc": box, "encoding": enc}
			for (box, enc) in zip(boxes, encodings)]
		data.extend(d)

	# dump the facial encodings data to disk
	print("[INFO] serializing encodings...")
	with open("encodings/encodings.pickle", "wb") as f:
		f.write(pickle.dumps(data))
########################################################################################################################################	

def cluster_faces():
	# load the serialized face encodings + bounding box locations from
	# disk, then extract the set of encodings to so we can cluster on
	# them
	print("[INFO] loading encodings...")
	data = pickle.loads(open("encodings/encodings.pickle", "rb").read())
	data = np.array(data)
	encodings = [d["encoding"] for d in data]

	# cluster the embeddings
	print("[INFO] clustering...")
	clt = DBSCAN(metric="euclidean", n_jobs=-1)
	clt.fit(encodings)
	# determine the total number of unique faces found in the dataset
	labelIDs = np.unique(clt.labels_)
	numUniqueFaces = len(np.where(labelIDs > -1)[0])
	print("[INFO] # unique faces: {}".format(numUniqueFaces))

	# loop over the unique face integers
	for labelID in labelIDs:
		# find all indexes into the `data` array that belong to the
		# current label ID, then randomly sample a maximum of 25 indexes
		# from the set
		print("[INFO] faces for face ID: {}".format(labelID))
		idxs = np.where(clt.labels_ == labelID)[0]
		idxs = np.random.choice(idxs, size=min(25, len(idxs)),
			replace=False)
		# initialize the list of faces to include in the montage
		faces = []

		# loop over the sampled indexes
		for i in idxs:
			# load the input image and extract the face ROI
			image = cv2.imread(data[i]["imagePath"])
			(top, right, bottom, left) = data[i]["loc"]
			face = image[top:bottom, left:right]
			# force resize the face ROI to 96x96 and then add it to the
			# faces montage list
			face = cv2.resize(face, (96, 96))
			faces.append(face)

		# create a montage using 96x96 "tiles" with 5 rows and 5 columns
		montage = build_montages(faces, (96, 96), (5, 5))[0]

		# show the output montage
		title = "Face ID #{}".format(labelID)
		title = "Unknown Faces" if labelID == -1 else title
		cv2.imshow(title, montage)
		cv2.waitKey(0)

if __name__ == "__main__":
	video_detection()
	encode_faces()
	cluster_faces()
	