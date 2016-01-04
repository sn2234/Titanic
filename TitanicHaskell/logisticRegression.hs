module LogisticRegression (
	CostGrad,
	cost,
	grad,
	
	hypothesis,
	gradDescentOptimize,
	runRegression,
	predict
)
where

import Data.Matrix
import Control.Monad
import Debug.Trace

{-
	Definitions
	
	x,y - training set
	theta - parameters vector
	
	m - number of samples in training set
	n - number of features
		
	x     :: R[m*n]
	y     :: R[m*1]
	theta :: R[n*1]
-}

data CostGrad = CostGrad { cost :: Double, grad :: Matrix Double } deriving (Show)

sigmoid x = 1/(1 + exp(-x))

hypothesis :: Matrix Double -> Matrix Double -> Matrix Double
hypothesis theta x =
	fmap sigmoid $ x * theta

xlog x =
	if x == 0.0 then
		log 1e-17
	else
		log x

computeCostGrad :: Matrix Double -> Matrix Double -> Matrix Double -> Double -> CostGrad
computeCostGrad theta x y lambda =
	let
		m = fromIntegral $ nrows x
		n = fromIntegral $ ncols x
		sig = hypothesis theta x
		ones = matrix (nrows sig) (ncols sig) (\_ -> 1)
		
		theta1 = setElem 0 (1, 1) theta
		regVal = (lambda / (2.0 * m))*(sum $ elementwise (*) theta1 theta1)
		
		forOnes = (transpose $ fmap (* (-1)) y) * (fmap xlog sig)
		forZeroes = (transpose $ fmap (\z -> 1-z) y) * (fmap xlog $ fmap (\z -> 1-z) sig)
	in
		CostGrad {
			cost = ((forOnes - forZeroes) ! (1, 1))/m + regVal,
			grad = transpose (fmap (/ m) $ ((transpose (sig - y)) * x)) + (fmap (\i -> i*(lambda/m)) theta1)
		}

gradDescentOptimize :: Matrix Double -> Double -> (Matrix Double -> CostGrad) -> Double -> Integer -> Matrix Double
gradDescentOptimize theta alpha fx epsilon iterations =
	let
		c1 = fx theta
		newTheta = theta - (fmap (* alpha) $ grad c1)
		c2 = fx newTheta
		diff = ((cost c1 - cost c2) ** 2)
		diff1 = {-Debug.Trace.trace ("diff: " ++ show diff ++ " cost: " ++ (show $ cost c2) ++ " newTheta: " ++ show newTheta)$-} diff
	in
		if diff1 > epsilon && iterations > 0  then
			gradDescentOptimize newTheta alpha fx epsilon (iterations - 1)
		else
			newTheta

runRegression alpha lambda x y iterations =
	let
		theta = zero (ncols x) 1
	in
		gradDescentOptimize theta alpha (\z -> computeCostGrad z x y lambda) 0 iterations

predict theta x theshold =
	let
		estimatedProbabilities = hypothesis theta x
	in
		fmap (\x -> if x >= theshold then 1.0 else 0.0) estimatedProbabilities


testCost = do
	let x = fromLists [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	let y = fromLists [[1],[0],[1],[0]]
	let theta = fmap (+1) $ zero 3 1
	let lambda = 1.5
	print "X:"
	print x
	print "Y:"
	print y
	print "Theta:"
	print theta
	print "Hypothesis"
	print $ hypothesis theta x
	let result = computeCostGrad theta x y lambda
	print "Cost:"
	print $ cost result
	print "Gradient:"
	print $ grad result
	let opt = gradDescentOptimize theta 0.01 (\z -> computeCostGrad z x y lambda) 0 3000
	print "Optimized theta:"
	print opt
	let newCost = cost $ computeCostGrad opt x y lambda
	print "New cost:"
	print newCost
	let rr = runRegression 0.01 lambda x y 1000
	print "runRegression:"
	print rr
