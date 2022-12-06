# import important packages
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
import cv2
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

matplotlib.use('TKAgg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(_argv):

    warnings.filterwarnings("ignore") #train random forest classifier
    data = pd.read_csv("updatedTruth.csv")
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
    testList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    testList = np.array(testList).reshape(-1, 11)
    testList = sc.transform(testList)
    y_pred = loaded_rf.predict(testList)
    #print(y_pred)
    #print(type(y_pred))


    #Computer vision model
    model = get_model() #gets model from


    while True:


        cap = cv2.VideoCapture(0)
        #cap = cv2.imRead
        while cap.isOpened():
            ret, frame = cap.read()

            results = model(frame)


            dataframe = results.pandas().xyxy[0]



            print('\n', dataframe)
            #print('\n', dataframe.shape)

            isempty = dataframe.empty


            if (isempty == False):

                frameHeight = dataframe.shape[0]
                for i in range(frameHeight):

                    confidence = dataframe.iat[i, 4]
                    workzoneObject = dataframe.iat[i, 6]
                    workzoneObjectID = dataframe.iat[i, 5]
                    if(confidence >= 0.1):                             # the following "switch cases" can be achieved also with dictionary mapping and using the numerical ID of the objects
                        if (workzoneObject == "barricade"):
                            testList[2] += 1
                        elif (workzoneObject == "worker"):
                            testList[6] += 1
                        elif (workzoneObject == "construction vehicle"):
                            testList[7] += 1
                        elif (workzoneObject == "delineator"):
                            testList[4] += 1
                        elif (workzoneObject == "cone"):
                            testList[0] += 1
                        elif (workzoneObject == "barrel"):
                            testList[1] += 1
                        elif (workzoneObject == "trench cover"):
                            testList[9] += 1
                        elif (workzoneObject == "sign"):
                            testList[8] += 1
                        elif (workzoneObject == "fence"):
                            testList[3] += 1
                        elif (workzoneObject == "vent"):
                            testList[5] += 1
                        else:
                            testList[10] += 1

            cv2.imshow('YOLO', np.squeeze(results.render()))




            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])