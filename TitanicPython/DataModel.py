import numpy as np
import csv

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

def populateModel(data):
    tmp = np.array(data)
    x = tmp[1:,2:].astype(float)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    y = tmp[1:,1].astype(float)
    return (x, y.reshape(len(data)-1,1))

def loadData(fileName):
    rawData = readData(fileName)
    validateHeader(rawData)
    return populateModel(rawData)
