#----------------------------------------
# Practical Machine Learning Project Code
#----------------------------------------
# Author: Jeremy Sidwell
# Date: November 5, 2018
#----------------------------------------
#
# This code supports the PML Project.RMD file
#
# Libaries applied in project:
library(AppliedPredictiveModeling)
library(caret)
library(ElemStatLearn)
library(pgmm)
library(rpart)
#install.packages("gbm")
library(gbm)
library(lubridate)
#install.packages("forecast")
library(forecast)
library(e1071)
library(randomForest)


#Load Test and Training Data
# Data has been downloaded from course website and saved to the following locations:
training = read.csv("~/Documents/DataScience/Course 8/Programming Assignment/DATA/pml-training.csv")
testing  = read.csv("~/Documents/DataScience/Course 8/Programming Assignment/DATA/pml-testing.csv")
head(training$classe)
head(training)
dim(training);dim(testing)
# Note only 20 rows available for Testing File 

# Dataset Review
# Variable to include in the model:
View(training,"PML Training Data")
# Keep only those columns that hold the sensor data (Exclude Username, Entry Time, etc.)
sensorData = grep(pattern = "belt|arm|dumbbell|forearm|classe",names(training))
head(sensorData)
training = training[,c(sensorData)]
dim(training)
View(training,"PML Adjusted Training Data")

# Keep columns where at least 75% of rows are not NA (or #DIV/0!)
## Convert Blank cells to NA
train.na<-apply(training,2,function(x) gsub("^$|^ $", NA, x))
training.clean = training[,which(colSums(!is.na(train.na))>14716)] #75% of 19622
dim(training.clean)
table(complete.cases(training.clean))
View(training.clean,"PML Cleaned Training Data")
# 19622 rows and 53 columns

table(sapply(training.clean[1,],class))
# Classe is factor, all other columns are numeric.

# How many entries by class?
library(plyr)
count(training.clean,"classe")
# 5,580 entries in Classe A, between 3k and 4k in Classe B-E.

# Correlation Matrix to observe columns: 
library(corrplot)
# Calculate Correlations, omit the Classe Factor Variable
cor.train <- cor(training.clean[,-53])
corrplot(cor.train, type = "upper", order="hclust",
                    method = "color",
                    labels = "none",
        sig.level = 0.05)
# Appears very few variables are strongly correlated

# Model Build
#Fit 1) RF, 2) Boosted "gmb", 3) LDA.  
set.seed(12345)
#Ensure Classe Variable is a factor
inTrain <- createDataPartition(training.clean$classe, p = 3/4)[[1]]
training.train = training.clean[ inTrain,]
training.validate = training.clean[-inTrain,]

control <- trainControl(method = "cv", number = 5)
modFitRF <- train(classe~., data=training.train, method="rf", trControl = control)
modFitGBM <- train(classe~.,data=training.train, method="gbm", trControl = control)
modFitLDA <- train(classe~.,data=training.train, method="lda", trControl = control)

# Model Testing on Validation Set
predRF <- predict(modFitRF,training.validate)
predGBM<-predict(modFitGBM,training.validate)
predLDA <- predict(modFitLDA,training.validate)
cmRF <- confusionMatrix(predRF, training.validate$classe)
cmGBM <- confusionMatrix(predGBM, training.validate$classe)
cmLDA <- confusionMatrix(predLDA, training.validate$classe)

overallRF <- cmRF$overall
overallGBM <- cmGBM$overall
overallLDA <- cmLDA$overall

overallRF
Model.Names <- names(overallRF[1:6])
Model.Names <- c("Model_Type",Model.Names)
Model.Names

# Create Percent function to format output for write-up
percent <- function(x, digits = 2, format = "f", ...) {
  paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}
Model_Type <- c('RF','GBM','LDA')
Model.Output <- cbind(Model_Type,rbind(percent(overallRF[1:6]),percent(overallGBM[1:6]),percent(overallLDA[1:6])))
colnames(Model.Output)<-Model.Names
# Model Accuracy Results
Model.Output 


#Best Fit Model: Random Forest
# In Sample Error
predTrainRF <- predict(modFitRF, training.train)
Accuracy.Train <- sum(predTrainRF == training.train$classe)/length(predTrainRF)
InSampleError <- (1-Accuracy.Train)*100
round(InSampleError, digits= 2) # In Sample Error Estimate: 0.0%

# Out of Sample Error appyling the best fit model:
# Accuracy of the model on Test Data
Accuracy.Test <- sum(predRF == training.test$classe)/length(predRF)
# Out of Sample Error
OutOfSampleError <- (1-Accuracy.Test)*100
round(OutOfSampleError, digits= 2) # Out of Sample Error Estimate: 0.63%


# Predict on the Test Set
testRF <- predict(modFitRF,testing)
testRF
