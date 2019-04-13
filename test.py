import imutils
import pickle
import time
import face_recognition
import pickle
import cv2
import dlib
import numpy as np
import pandas as pd
import numpy as np
import dlib

dlib.DLIB_USE_CUDA = True

encode_file="encodings.pickle"
#detection_method='cnn'
detection_method='hog'

#reading the encoded training data
data = pickle.loads(open(encode_file, "rb").read())

# We turn the webcam on.
video_capture = cv2.VideoCapture(0) 
print("[INFO] Camera Capturing...")
time.sleep(2.0)

while True:
    ret, frame = video_capture.read() # We get the last frame.
    if(ret):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        continue
    rgb = imutils.resize(frame, width=1000)
    r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input frame, then compute the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,model=detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    for encoding in encodings:
        # attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=0.48)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
			# find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                    
            # determine the recognized face with the largest number of votes (note: in the event of an unlikely tie Python will select first entry in the dictionary)
            name = max(counts, key=counts.get)
		# update the list of names
        names.append(name)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

	# draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        if name in names: 
            if name == "Unknown":
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            else:
                cv2.putText(frame, name+" Marked", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.

print("[INFO] Recognising")

UID = list(set(data['names']))
Status = []
for x in UID:
    Status.append('A') 
for x in names:
    if x in UID:
        Status[UID.index(x)] = 'P'

print("[INFO] UID: ",UID)
print("[INFO] Status: ",Status)


#write data to excel file
import pandas as pd
df = pd.read_csv("Attendance.csv")
df["UID"] = UID
df["Status"] = Status
df.to_csv("Attendance.csv", index=False)

print("[Success] Database Updated")


from sklearn.neighbors import KNeighborsClassifier

x=data['encodings']
y=np.array(data['names'])

from sklearn.model_selection import  train_test_split 

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

classifier=KNeighborsClassifier()

classifier.fit(xtrain,ytrain)

print("[Info] Accuracy Score: ",classifier.score(xtest,ytest)*100)

