
import Data.Matrix
import Control.Monad

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

data CostGrad = CostGrad { cost :: Float, grad :: Matrix Float } deriving (Show)

sigmoid x = 1/(1 + exp(-x))

hypothesis :: Matrix Float -> Matrix Float -> Matrix Float
hypothesis theta x =
	fmap sigmoid $ x * theta

xlog x =
	if x == 0.0 then
		log 1e-10
	else
		log x

computeCostGrad :: Matrix Float -> Matrix Float -> Matrix Float -> Float -> CostGrad
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
			grad = (fmap (/ m) $ ((transpose (sig - y)) * x)) + (fmap (\i -> i*(lambda/m)) theta1)
		}

testCost = do
	let x = fromLists [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	let y = fromLists [[1],[0],[1],[0]]
	let theta = fmap (+1) $ zero 3 1
	print "X:"
	print x
	print "Y:"
	print y
	print "Theta:"
	print theta
	print "Hypothesis"
	print $ hypothesis theta x
	let result = computeCostGrad theta x y 0.0
	print "Cost:"
	print $ cost result
	print "Gradient:"
	print $ grad result
