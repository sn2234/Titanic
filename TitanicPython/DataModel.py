import numpy as np
import csv

class DataLoader:
    def initNormalize(self, data):
        self.normalization = [(data[:,i].mean(), data[:,i].std()) for i in range(data.shape[1])]

    def normalizeDataSet(self, data):
        for i in range(data.shape[1]):
            data[:,i] = (data[:,i] - self.normalization[i][0])/self.normalization[i][1]
        return data

    def loadTrainingSet(self, fileName):
        rawData = readData(fileName)
        validateHeader(rawData)
        (x, y) = populateModel(rawData)
        self.initNormalize(x)
        x = self.normalizeDataSet(x)
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        return (x,y)

    def loadDataSet(self, fileName):
        rawData = readData(fileName)
        validateHeader(rawData)
        (x, y) = populateModel(rawData)
        x = self.normalizeDataSet(x)
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        return (x,y)

def readData(fileName):
    data = []
    with open(fileName, 'r') as inp:
        csvReader = csv.reader(inp)
        for row in csvReader:
            data.append(row[0:])
    return data

def validateHeader(data):
    if data[0] != ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']:
        print("Bad header")
        raise Exception("Wrong header")

def mapAndNomalizeFeatures(x):
    # Notmalize fetures values
    # Add polinomial features
    # Add y-intercept
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    return x

def populateModel(data):
    tmp = np.array(data)
    x = tmp[1:,2:].astype(float)
    y = tmp[1:,1].astype(float)
    return (x, y.reshape(len(data)-1,1))

def loadData(fileName):
    rawData = readData(fileName)
    validateHeader(rawData)
    return populateModel(rawData)
