
import cv2
import glob2
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os
from os import listdir
import matplotlib
from Algorithm.main import *
import argparse
import time
import requests
import yaml
import tqdm
import torchvision
from turtle import color
import pandas as pd
import seaborn
import numpy as np
from PIL import Image
import PIL
import png


images = []

warnings.filterwarnings("ignore")
data = pd.read_csv("updatedTruth.csv")  #Training Classifier
X = data.iloc[:, 1:12].values
y = data.iloc[:, 12].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
joblib.dump(classifier, "./random_forest.joblib")
loaded_rf = joblib.load("./random_forest.joblib")
#End of classifier training


for file in glob2.glob(r'C:\Users\nydan\PycharmProjects\C2Smart\cctv266\*.jpg'):
    images.append(file)                                                                #Loop through folder to get all images
# Model
model = get_model()
font = cv2.FONT_ITALIC


counter=0
for img in images:
    counter+=1
    results = model(img)  # includes NMS
    #print(img)
    dataframe = results.pandas().xyxy[0]
    #print('\n', dataframe)
    isempty = dataframe.empty
    classifierInput = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #resets classifier input


    if (isempty == False):

        frameHeight = dataframe.shape[0] #gets height of dataframe to loop through
        writtenImg = cv2.imread(img)
        for i in range(frameHeight):

            confidence = dataframe.iat[i, 4]
            workzoneObject = dataframe.iat[i, 6]
            workzoneObjectID = dataframe.iat[i, 5]
            xLeft = dataframe.iat[i, 0] #lines 75 to 82 get the coordinates of the object if you wish to map the objects on your own with matplot
            xRight = dataframe.iat[i, 2]
            yLower = dataframe.iat[i, 1]
            yUpper = dataframe.iat[i, 3]
            xCoor = (xRight + xLeft) / 2
            yCoor = (yUpper + yLower) / 2
            x = dataframe.iat[i, 0]
            y = dataframe.iat[i, 1]
            imageHeight, imageWidth, c = writtenImg.shape
            width = xRight - xLeft
            height = yLower - yUpper

            if (confidence >= 0.1):  # the following "switch cases" can be achieved also with dictionary mapping and using the numerical ID of the objects
                if (workzoneObject == "barricade"):
                    classifierInput[2] += 1
                elif (workzoneObject == "worker"):
                    classifierInput[6] += 1
                elif (workzoneObject == "construction vehicle"):
                    classifierInput[7] += 1
                #elif (workzoneObject == "delineator"):
                #    classifierInput[4] += 1
                elif (workzoneObject == "cone"):
                    classifierInput[0] += 1
                elif (workzoneObject == "barrel"):
                    classifierInput[1] += 1
                #elif (workzoneObject == "trench cover"):
                #    classifierInput[9] += 1
                elif (workzoneObject == "sign"):
                    classifierInput[8] += 1
                #elif (workzoneObject == "fence"):
                #    classifierInput[3] += 1
                elif (workzoneObject == "vent"):
                    classifierInput[5] += 1
                else:
                    classifierInput[10] += 1
                #cv2.rectangle(writtenImg, (int(xLeft), int(yUpper)), (int(xRight), int(yLower)), (0, 255, 0), 2) #draws the lines by hand
                #cv2.putText(writtenImg, workzoneObject, (int(xLeft), int(yUpper+height)), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA) #labels the object above the bounding box


        cv2.imshow('YOLO', writtenImg)

        print(classifierInput)
        #isWorker = classifierInput[6]
        classifierInput = np.array(classifierInput).reshape(-1, 11)
        classifierInput = sc.transform(classifierInput)
        y_pred = loaded_rf.predict(classifierInput)
        print(y_pred)

        #if ((y_pred[0] == 1) or (isWorker != 0)): an example of what a future work could look like, combining the classifier with hand-set rules
        if (y_pred[0] == 1):
            cv2.putText(writtenImg, "Workzone Detected", (int(0), int(imageHeight-5)), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
        else:
            cv2.putText(writtenImg, "No Workzone", (int(0), int(imageHeight-5)), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('YOLO', writtenImg)

        cv2.imwrite("c266-"+str(counter)+".jpg", writtenImg)
        cv2.waitKey(1)
