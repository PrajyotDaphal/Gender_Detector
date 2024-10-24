import cv2 as cv
import json
import os
import numpy as np

# Load address data from JSON
with open("E:\\Projects\\Python\\Advanced Voice Assistant\\Data\\json\\Address.json", 'r') as R:
    address = json.load(R)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
padding = 20

# Load DNN models
if not all(os.path.isfile(path) for path in [address['faceProto'], address['faceModel'], address['genderProto'], address['genderModel']]):
    raise FileNotFoundError("One or more model files are missing. Please check the file paths.")

# Load the gender detection model and proto file
genderNet = cv.dnn.readNet(address['genderModel'], address['genderProto'])
faceNet = cv.dnn.readNet(address['faceModel'], address['faceProto'])

def getFaceBox(net, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    
    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    
    return bboxes

# Define the function for gender detection
def gender_detector(img, genderNet, faceNet):
    # Preprocess the image for face detection
    blob = cv.dnn.blobFromImage(img, 1.0, (227, 227), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Iterate over detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI and prepare it for gender detection
            face = img[startY:endY, startX:endX]
            face_blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0))
            genderNet.setInput(face_blob)
            gender_preds = genderNet.forward()
            gender = "Male" if gender_preds[0][0] > 0.5 else "Female"
            return [gender]  # Return the detected gender
    return []
