import numpy as np
import math as m
from scipy.optimize import minimize

vexp = np.vectorize(m.exp, otypes=[float])

def sigmoid(x):
    return 1/(1 + vexp(-x))

xlog = np.vectorize(lambda x: m.log(1e-17) if x == 0 else m.log(x))

def hypothesis(theta, x):
    return sigmoid(np.dot(x, theta))

def computeCostGrad(theta, x, y, lmb):
    (m, n) = x.shape
    thetaR = theta.reshape((n,1))
    sig = hypothesis(thetaR, x)
    theta1 = thetaR.copy()
    theta1[0] = 0
    regVal = (lmb/(2*m)) * (theta1*theta1).sum()
    forOnes = np.dot((-y).transpose(), xlog(sig))
    forZeroes = np.dot((1-y).transpose(), xlog(1 - sig))

    cost = ((forOnes - forZeroes).flat[0])/m + regVal
    grad = (np.dot((sig - y).transpose(), x)/m).transpose() + (lmb/m)*theta1

    return (cost, grad)

def costFun(theta, x, y, lmb):
    (cost, _) = computeCostGrad(theta, x, y, lmb)
    return cost

def gradFun(theta, x, y, lmb):
    (_, grad) = computeCostGrad(theta, x, y, lmb)
    return grad.reshape((grad.size,))

def runRegression(lmb, x, y):
    (nr, nc) = x.shape
    theta = np.zeros((nc, 1))
    thetaOpt = minimize(lambda p: costFun(p, x, y, lmb),
                    theta,
                    method='BFGS',
                    jac=lambda p: gradFun(p, x, y, lmb),
                    options={'disp': True})
    return thetaOpt.x.reshape(nc,1)

def squareError(x, y):
    return ((x-y)**2).sum()/(2*x.size)

def predictionError(x, y):
    return np.mean([1 if i[0] != i[1] else 0 for i in zip(x,y)])

def predict(theta, x, theshold):
    estimatedProbabilities = hypothesis(theta, x)
    return np.array([1.0 if p >= theshold else 0.0 for p in estimatedProbabilities])

# Do some tests
def doTests():
    x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=float)
    y = np.array([[1],[0],[1],[0]], dtype=float).reshape(4, 1)

    theta = np.ones((3,1))

    l = 1.5

    h = hypothesis(theta, x)
    print("Hypothesis: {0}".format(h))

    (cost, grad) = computeCostGrad(theta, x, y, l)
    print("Cost: {0}\nGrad: {1}".format(cost, grad))

    opt = gradDescentOptimize(theta, 0.01, lambda p: computeCostGrad(p, x, y, l), 3000)
    print("Optimized theta: {0}".format(opt))

    (newCost, _) = computeCostGrad(opt, x, y, l)
    print("New cost: {0}".format(newCost))

    rr = runRegression(0.01, l, x, y, 1000)

    print("runRegression: {0}".format(rr))

    pp = predict(rr, x, 0.5)
    print("Predicted results: {0}".format(pp));
