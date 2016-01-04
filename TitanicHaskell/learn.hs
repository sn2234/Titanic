
import Data.Matrix
import Control.Monad
import Debug.Trace
import Control.Monad.Except
import Control.Applicative

import DataModel
import LogisticRegression

-- Process data set and produce X and Y for logistic regression
processDataModel :: [DataRec] -> (Matrix Double, Matrix Double)
processDataModel dta =
	(
		matrixFromLists $ map dataRecToList dta,
		matrixFromLists $ map (\ x -> [survived x]) dta
	)

-- X vector only
dataRecToList :: DataRec -> [Double]
dataRecToList r =
		[pclass r, sex r, age r, sibSp r, parch r, fare r, embarked r]

matrixFromLists :: [[a]] -> Matrix a
matrixFromLists x =
	matrix (length x) (length (x !! 0)) (\(i, j) -> (x !! (i - 1)) !! (j - 1))

squareError :: Matrix Double -> Matrix Double -> Double
squareError a b =
	(sum $ elementwise (\x y -> (x - y)^^2) a b)/(fromIntegral $ 2*(ncols a)*(nrows a))

test = do
	rawData <- parseDataFile "train_pure_cv.csv"
	let res = processDataModel <$> rawData
	print res

reg x = runRegression 0.01 0.0 (fst x) (snd x) 100

main = do
	print "Training logistic regression using trainig data set"
	rawData <- parseDataFile "train_pure_train.csv"
	let processedData = processDataModel <$> rawData
	let theta = reg <$> processedData
	print "Theta:"
	print theta
	
	cvRawData <- parseDataFile "train_pure_cv.csv"
	let cvProcessedData = processDataModel <$> cvRawData
	let predictionCv = (\t z -> predict t (fst z) 0.5) <$> theta <*> cvProcessedData
	let errorCv = squareError <$> predictionCv <*> (snd <$> cvProcessedData)
	print "errorCv"
	print errorCv
	
