#Loading libraries
library(jsonlite)
library(tm)
library(xgboost)
library(Matrix)
library(rpart)
library(caret)

#Loading the train and test dataset
train <- fromJSON("train.json")
test <- fromJSON("test.json")

#Adding the cuisine to the test set
test$cuisine <- NA

#Combine train and test set for preprocess
full_data <- rbind(train,test)

corpus <- Corpus(VectorSource(full_data$ingredients))

#Processing the text
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, c(stopwords('english')))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)


#Counting the frequencies
frequencies <- DocumentTermMatrix(corpus)

freq <- colSums(as.matrix(freq))
ord <- order(freq)

sparse <- removeSparseTerms(frequencies, 0.99)

final_train <- as.data.frame(as.matrix(sparse[1:nrow(train),]))
final_test <- as.data.frame(as.matrix(sparse[-(1:nrow(train)),]))

final_train$cuisine <- as.factor(train$cuisine)
train$cuisine <- as.factor(train$cuisine)


#CART Model

cart_model <- rpart(cuisine ~ . , data=final_train, method="class")

confusionMatrix(predict(cart_model,type="class"),train$cuisine)

test$cuisine <- predict(cart_model, newdata = final_test, type="class")

write.csv(test[,c("id","cuisine")],file = "cart_op.csv", row.names = FALSE)

#Random Forest
library(ggplot2)
library(randomForest)
random_for <- randomForest(cuisine ~., data=final_train,importance=TRUE,ntree=20)

confusionMatrix(predict(random_for,type = "class"),train$cuisine)

test$cuisine <- predict(random_for, newdata = final_test, type="class")

write.csv(test[,c("id","cuisine")],file = "rf_op.csv", row.names = FALSE)
