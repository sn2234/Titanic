import numpy as np

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
    testData = dl.loadDataSet("..\\train_pure_cv.csv")

    theta = LogisticRegression.runRegression(0, trainData[0], trainData[1])

    print("Theta: {0}".format(theta))

    print("Checking hypothesis on train set")
    checkRegressionHypothesis(theta, trainData[0], trainData[1], 0.5)

    print("Checking hypothesis on CV set")
    checkRegressionHypothesis(theta, cvData[0], cvData[1], 0.5)

def test2LoadData():
    data = DataModel.readData("ex2data1.txt")
    tmp = np.array(data)
    x = tmp[:,:2].astype(float)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    y = tmp[:,2].astype(float).reshape(tmp.shape[0], 1)
    return (x, y)

def test2():
    (x, y) = test2LoadData()
    theta = np.zeros((x.shape[1],1))
    (cost, grad) = LogisticRegression.computeCostGrad(theta, x, y, 0)
    print("Initial Cost: {0}\nGrad: {1}".format(cost, grad))
    theta = LogisticRegression.runRegression(0, x, y)
    (cost, grad) = LogisticRegression.computeCostGrad(theta, x, y, 0)
    print("Optimized Cost: {0}\nGrad: {1}\nTheta: {2}".format(cost, grad,theta))

test1()
