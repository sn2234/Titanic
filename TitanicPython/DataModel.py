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
        (x, y, id) = populateModel(rawData)
        x = addPolynomialFeatures(x)
        self.initNormalize(x)
        x = self.normalizeDataSet(x)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return (x, y, id)

    def loadDataSet(self, fileName):
        rawData = readData(fileName)
        validateHeader(rawData)
        (x, y, id) = populateModel(rawData)
        x = addPolynomialFeatures(x)
        x = self.normalizeDataSet(x)
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        return (x, y, id)

def polyMult(c, x):
    return np.hstack((c, x*c))

def addPolynomialFeatures(data):
    if data.shape[1] == 0:
        return np.zeros((data.shape[0], 0))
    else:
        return np.hstack((polyMult(data[:,0].reshape((data.shape[0], 1)), data),
                          addPolynomialFeatures(data[:,1:])))

def polynomialMultiply(c, x):
    return [c] + [c+i for i in x]

def polySq(x):
    if len(x) == 0:
        return []
    else:
        return polynomialMultiply(x[0], x) + polySq(x[1:])

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
    id = tmp[1:,0].astype(float)
    return (x, y.reshape(len(data)-1,1), id.reshape(len(data)-1,1))

def loadData(fileName):
    rawData = readData(fileName)
    validateHeader(rawData)
    return populateModel(rawData)
