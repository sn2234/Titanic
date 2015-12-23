
import Data.Matrix
import Control.Monad
import Debug.Trace

import DataModel

{-
mapMat :: (a -> a) -> Matrix a -> Matrix a
mapMat f m =
	let
		n = nrows m
		processRow f n m =
			if n == 0 then
				mapRow (\_ x -> f x) n m
			else
				mapRow (\_ x -> f x) n (processRow f (n-1) m)
	in
		processRow f n m
-}

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
		log 1e-10
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

gradDescentOptimize :: Matrix Double -> Double -> (Matrix Double -> CostGrad) -> Double -> Matrix Double
gradDescentOptimize theta alpha fx epsilon =
	let
		c1 = fx theta
		newTheta = theta - (fmap (* alpha) $ grad c1)
		c2 = fx newTheta
		diff = ((cost c1 - cost c2) ** 2)
		diff1 = Debug.Trace.trace ("diff: " ++ show diff) $ diff
	in
		if diff1 > epsilon then
			gradDescentOptimize newTheta alpha fx epsilon
		else
			newTheta

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
	let opt = gradDescentOptimize theta 0.01 (\z -> computeCostGrad z x y lambda) 0.000001
	print "Optimized theta:"
	print opt
	let newCost = cost $ computeCostGrad opt x y lambda
	print "New cost:"
	print newCost
