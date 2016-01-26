import numpy as np
from matplotlib import pyplot as plt
import math
import sys
import csv

import DataModel
import LogisticRegression

def checkRegressionHypothesis(theta, x, y, threshold):
    (cost, _) = LogisticRegression.computeCostGrad(theta, x, y, 5)
    prediction = LogisticRegression.predict(theta, x, threshold)
    predictionError = LogisticRegression.predictionError(prediction, y)
    print("Predicted values: {0}\nPrediction error: {1}\nCost: {2}"
          .format(prediction[:10], predictionError,cost))

def test1():
    dl = DataModel.DataLoader()
    trainData = dl.loadTrainingSet("..\\train_pure_train.csv")
    cvData = dl.loadDataSet("..\\train_pure_cv.csv")
    testData = dl.loadDataSet("..\\train_pure_test.csv")

    theta = LogisticRegression.runRegression(0, trainData[0], trainData[1])

    print("Theta: {0}".format(theta))

    print("Checking hypothesis on train set")
    checkRegressionHypothesis(theta, trainData[0], trainData[1], 0.5)

    print("Checking hypothesis on CV set")
    checkRegressionHypothesis(theta, cvData[0], cvData[1], 0.5)

def makeSubSet(data, subSetLen):
    return (data[0][:subSetLen,:], data[1][:subSetLen,:].reshape(subSetLen,1))

def getLearnError(trainData, testData, threshold, regLambda):
    (x, y, _) = trainData
    (x_, y_, _) = testData
    theta = LogisticRegression.runRegression(regLambda, x, y)
    predictionTrain = LogisticRegression.predict(theta, x, threshold)
    predictionErrorTrain = LogisticRegression.predictionError(predictionTrain, y)

    predictionTest = LogisticRegression.predict(theta, x_, threshold)
    predictionErrorTest= LogisticRegression.predictionError(predictionTest, y_)
    return (predictionErrorTrain, predictionErrorTest)

def getSubsetLearnError(trainData, testData, subSetLen, threshold, regLambda):
    return getLearnError(makeSubSet(trainData, subSetLen), makeSubSet(testData, subSetLen), threshold, regLambda)

def plotLearningCurve():
    dl = DataModel.DataLoader()
    trainData = dl.loadTrainingSet("..\\train_pure_train.csv")
    cvData = dl.loadDataSet("..\\train_pure_cv.csv")
    testData = dl.loadDataSet("..\\train_pure_test.csv")

    lenRange = np.linspace(1, min(trainData[0].shape[0], cvData[0].shape[0]), num = 10, dtype=int)
    for i in lenRange:
        (errTrain,errTest) = getSubsetLearnError(trainData, cvData, i, 0.5, 0)
        print("Len: {0} Err Train: {1} Err Test: {1}".format(i, errTrain, errTest))

def fitRegParam(trainData,cvData ):
    minErr = sys.float_info.max
    optimalReg = 0
    rspace = np.linspace(0, 5, num = 10, dtype = float)
    for i in rspace:
        (errTrain,errTest) = getLearnError(trainData, cvData, 0.5, i)
        print("Lambda: {0} Train error: {1} Test error: {2}".format(i, errTrain, errTest))
        if errTest < minErr:
            minErr = errTest
            optimalReg = i

    print("Optimal lambda value: {0} with error: {1}".format(optimalReg, minErr))
    return optimalReg

def doTest():
    dl = DataModel.DataLoader()
    trainData = dl.loadTrainingSet("..\\train_pure_train.csv")
    cvData = dl.loadDataSet("..\\train_pure_cv.csv")

    optimalReg = fitRegParam(trainData, cvData)

    del dl
    dl = DataModel.DataLoader()
    trainFull = dl.loadTrainingSet("..\\train_full.csv")
    testFull = dl.loadDataSet("..\\test_full.csv")

    theta = LogisticRegression.runRegression(optimalReg, trainFull[0], trainFull[1])
    prediction = LogisticRegression.predict(theta, testFull[0], 0.5)

    outData = np.hstack((testFull[2], prediction.reshape((prediction.shape[0], 1))))
    
    with open("..\\result.csv", 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['PassengerId', 'Survived'])

        for r in outData:
            writer.writerow([int(r[0]), int(r[1])])
#test1()
#plotLearningCurve()
#fitRegParam()
doTest()