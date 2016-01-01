
import Data.Matrix
import Control.Monad
import Debug.Trace

import DataModel
import LogisticRegression

-- Process data set and produce X and Y for logistic regression
processDataModel :: [DataRec] -> (Matrix Double, Matrix Double)
processDataModel dta =
	(
		matrixFromLists $ map dataRecToList dta,
		transpose $ matrixFromLists $ map (\ x -> [survived x]) dta
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
	case rawData of
		Right d -> print $ processDataModel d
		Left err -> print $ "Error: " ++ err
