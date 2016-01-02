
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

test = do
	rawData <- parseDataFile "train_pure_cv.csv"
	let res = processDataModel <$> rawData
	print res

reg x = runRegression 0.01 0.0 (fst x) (snd x) 10000

main = do
	rawData <- parseDataFile "train_pure_cv.csv"
	let processedData = processDataModel <$> rawData
	let theta = reg <$> processedData
	print theta
