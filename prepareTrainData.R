library(caret)

# Load training samples
train <- read.csv("train.csv")

# Extract and convert
trainOut <- data.frame(PassengerId = train$PassengerId,
                       Survived = train$Survived,
                       Pclass = train$Pclass,
                       Sex = as.numeric(train$Sex),
                       Age = train$Age,
                       SibSp = train$SibSp,
                       Parch = train$Parch,
                       Fare = train$Fare,
                       Embarked = as.numeric(train$Embarked))
trainOut <- trainOut[complete.cases(trainOut), ]

# Split training and cross-validation sets
set.seed(12345)
inTrain = createDataPartition(trainOut$PassengerId, p = 2/3, list = FALSE)

trainOutTrain <- trainOut[inTrain, ]
trainOutX <- trainOut[-inTrain,]

inTrain2 = createDataPartition(trainOutX$PassengerId, p = 1/2, list = FALSE)
trainOutCV = trainOutX[inTrain2, ]
trainOutTest = trainOutX[-inTrain2, ]

write.csv(trainOutTrain, file = "train_pure_train.csv", row.names = FALSE)
write.csv(trainOutCV, file = "train_pure_cv.csv", row.names = FALSE)
write.csv(trainOutTest, file = "train_pure_test.csv", row.names = FALSE)

