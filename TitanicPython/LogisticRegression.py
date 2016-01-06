import numpy as np
import math as m

vexp = np.vectorize(m.exp, otypes=[float])

def sigmoid(x):
    return 1/(1 + vexp(-x))

xlog = np.vectorize(lambda x: m.log(1e-20) if x == 0 else m.log(x))

def hypothesis(theta, x):
    return sigmoid(np.dot(x, theta))

def computeCostGrad(theta, x, y, lmb):
    (m, n) = x.shape
    sig = hypothesis(theta, x)
    theta1 = theta.copy()
    theta1[0] = 0
    regVal = (lmb/(2*m)) * (theta1*theta1).sum()
    forOnes = np.dot((-y).transpose(), xlog(sig))
    forZeroes = np.dot((1-y).transpose(), xlog(1 - sig))

    cost = ((forOnes - forZeroes)[0,0])/m + regVal
    grad = (np.dot((sig - y).transpose(), x)/m).transpose() + (lmb/m)*theta1

    return (cost, grad)

def gradDescentOptimize(theta, alpha, fx, iterations):
    for i in range(iterations):
        (cost, grad) = fx(theta)
        theta -= grad*alpha

    return theta

def runRegression(alpha, lmb, x, y, iterations):
    (nr, nc) = x.shape
    theta = np.zeros((nc, 1))
    return gradDescentOptimize(theta, alpha, lambda p: computeCostGrad(p, x, y, lmb), iterations)

x = np.random.random((1000,7))

y = np.random.random((1000,1))

runRegression(0.01, 0, x, y, 20)