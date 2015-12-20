module DataModel
(
	DataRec,
	passengerId,
	survived,
	pclass,
	sex,
	age,
	sibSp,
	parch,
	fare,
	embarked,
	
	parseDataFile,
	parseDataRec
)
where

import Text.CSV
import Text.Parsec.Error

data DataRec = DataRec {
				 passengerId :: Float
				,survived :: Float
				,pclass :: Float
				,sex :: Float
				,age :: Float
				,sibSp :: Float
				,parch :: Float
				,fare :: Float
				,embarked :: Float
				} deriving(Show)

parseDataFile :: String -> IO (Either String [DataRec])
parseDataFile fileName = do
	content <- readFile fileName
	return $ parseDataRec fileName content

parseDataRec :: String -> String -> Either String [DataRec]
parseDataRec fileName dataText =
	case parseCSV fileName dataText of
		Left err -> Left (show err)
		Right csv -> processParsed csv

processParsed :: CSV -> Either String [DataRec]
processParsed [] = Left "Empty file"
processParsed (_:[]) = Left "No Data"
processParsed (headerRecord:dataRecords)
	| headerRecord /= ["PassengerId","Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"] = Left "Bad CSV Header"
	| otherwise = parseData [] dataRecords
	
parseData :: [DataRec] -> [Record] -> Either String [DataRec]
parseData acc [] = Right acc
parseData acc [_:[]] = Right acc
parseData acc (r:rs) =
	case parseRecord r of
		Right result -> parseData (acc ++ [result]) rs
		Left x -> Left x

parseRecord :: [Field] -> Either String DataRec
parseRecord (p1:p2:p3:p4:p5:p6:p7:p8:p9:[]) =
	Right $ DataRec { passengerId = read p1, survived = read p2, pclass = read p3, sex = read p4, age = read p5, sibSp = read p6, parch = read p7, fare = read p8, embarked = read p9 }
parseRecord _ = Left "Bad CSV record"
