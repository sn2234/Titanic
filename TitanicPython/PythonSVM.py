import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn import preprocessing

import csv

import LogisticRegression

def getLearnError(trainData, cvData, Cx):
    svmFit = svm.SVC(C = Cx)
    svmFit.fit(trainData[0], trainData[1])

    errTrain = LogisticRegression.predictionError(trainData[1], svmFit.predict(trainData[0]))
    errCv = LogisticRegression.predictionError(cvData[1], svmFit.predict(cvData[0]))
    
    return (errTrain, errCv)

def testSvm():
    featureColumns = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

    df = pd.io.parsers.read_csv("..\\train_pure_train.csv")
    std_scale = preprocessing.StandardScaler().fit(df[featureColumns])

    x = std_scale.transform(df[featureColumns])
    y = df['Survived'] #.reshape((df.shape[0], 1))

    df_cv = pd.io.parsers.read_csv("..\\train_pure_cv.csv")
    x_cv = std_scale.transform(df_cv[featureColumns])
    y_cv = df_cv['Survived']

    (errTrain, errCv) = getLearnError((x,y), (x_cv, y_cv), 1E-10)

    print("Err Train: {0}, Err CV: {1}".format(errTrain, errCv))

def svmRegularization():
    featureColumns = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

    df = pd.io.parsers.read_csv("..\\train_pure_train.csv")
    std_scale = preprocessing.StandardScaler().fit(df[featureColumns])

    x = std_scale.transform(df[featureColumns])
    y = df['Survived'] #.reshape((df.shape[0], 1))

    df_cv = pd.io.parsers.read_csv("..\\train_pure_cv.csv")
    x_cv = std_scale.transform(df_cv[featureColumns])
    y_cv = df_cv['Survived']

    cspace = np.linspace(0.1, 10, num = 50, dtype = float)
    errList = []
    for c in cspace:
        (errTrain, errCv) = getLearnError((x,y), (x_cv, y_cv), c)
        errList.append((errTrain, errCv))

    plt.plot(cspace, [x for (x,y) in errList], "r", label="Train")
    plt.plot(cspace, [y for (x,y) in errList], "g", label="CV")
    plt.legend()
    plt.show()

def doTest():
    df = pd.io.parsers.read_csv("..\\train_full.csv")
    featureColumns = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    std_scale = preprocessing.StandardScaler().fit(df[featureColumns])
    x = std_scale.transform(df[featureColumns])
    y = df['Survived'] #.reshape((df.shape[0], 1))

    svmFit = svm.SVC()
    svmFit.fit(x, y)

    df_test = pd.io.parsers.read_csv("..\\test_full.csv")
    x_test = std_scale.transform(df_test[featureColumns])

    y_pred = svmFit.predict(x_test)

    with open("..\\result_svm.csv", 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['PassengerId', 'Survived'])

        writer.writerows(zip(df_test['PassengerId'], y_pred))

svmRegularization()
