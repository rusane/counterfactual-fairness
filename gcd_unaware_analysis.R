library(caret)
library(sm)

# Baseline evaluation
load("gcd_data.Rdata")
N <- dim(data)[1]
data$credit_risk <- as.factor(data$credit_risk)

set.seed(0)
trainIndex <- createDataPartition(data$credit_risk, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- data[trainIndex,]
test <- data[-trainIndex,]
N_train <- dim(train)[1]
N_test <- dim(test)[1]
cv <- trainControl(method = "cv", number = 10)

unaware <- train(credit_risk ~ . - sex, method="glm", data=train, family="binomial", trControl=cv)
#unaware <- glm(credit_risk ~ . - sex, family=binomial("logit"), data=train)

pred <- predict(unaware, newdata=test)
confusionMatrix(data=pred, test$credit_risk, positive='1')



# Samples with original sex
load("data_samples_og.Rdata")
data_og$credit_risk <- as.factor(data_og$credit_risk)

pred <- predict(unaware, newdata=data_og)
confusionMatrix(data=pred, data_og$credit_risk, positive='1')
pred_raw <- predict(unaware, newdata=data_og, type="prob")[,'1']

# Samples with counterfactual sex
load("data_samples_cf.Rdata")
data_cf$credit_risk <- as.factor(data_cf$credit_risk)

pred_CF <- predict(unaware, newdata=data_cf)
confusionMatrix(data=pred_CF, data_cf$credit_risk, positive='1')
pred_raw_CF <- predict(unaware, newdata=data_cf, type="prob")[,'1']





# Original sampled data
load("data_samples_og.Rdata")
N <- dim(data_og)[1]
data_og$credit_risk <- as.factor(data_og$credit_risk)

set.seed(0)
trainIndex <- createDataPartition(data_og$credit_risk, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- data_og[trainIndex,]
test <- data_og[-trainIndex,]
N_train <- dim(train)[1]
N_test <- dim(test)[1]
male_test_idx <- which(test$sex %in% '0')
cv <- trainControl(method = "cv", number = 10)

#unaware <- glm(credit_risk ~ . - sex - u, family=binomial("logit"), data=train)
unaware <- train(credit_risk ~ . - sex - u, method="glm", data=train, family="binomial", trControl=cv)

pred <- predict(unaware, newdata=test)
confusionMatrix(data=pred, test$credit_risk, positive='1')

#pred_raw <- predict(unaware, newdata=test, type='response')
#pred <- ifelse(pred_raw > 0.5, 1, 0)
#error <- mean(pred != test$credit_risk)
#print(paste('Accuracy:', 1-error))

pred_raw <- predict(unaware, newdata=test, type="prob")[,'1']


# Counterfactual sampled data
load("data_samples_cf.Rdata")
N_CF <- dim(data_cf)[1]
data_cf$credit_risk <- as.factor(data_cf$credit_risk)


set.seed(0)
#trainIndex_CF <- createDataPartition(data_cf$credit_risk, p = .8, 
#                                     list = FALSE, 
#                                     times = 1)
train_CF <- data_cf[trainIndex,]
test_CF <- data_cf[-trainIndex,]
N_train_CF <- dim(train_CF)[1]
N_test_CF <- dim(test_CF)[1]
male_test_idx_CF <- which(test_CF$sex %in% '0')
cv <- trainControl(method = "cv", number = 10)

#unaware_CF <- glm(credit_risk ~ . - sex - u, family=binomial("logit"), data=train_CF)
unaware_CF <- train(credit_risk ~ . - sex - u, method="glm", data=train_CF, family="binomial", trControl=cv)

pred_CF <- predict(unaware_CF, newdata=test_CF)
confusionMatrix(data=pred_CF, test_CF$credit_risk, positive='1')

#pred_raw_CF <- predict(unaware_CF, newdata=test_CF, type='response')
#pred_CF <- ifelse(pred_raw_CF > 0.5, 1, 0)
#error_CF <- mean(pred_CF != test$credit_risk)
#print(paste('Accuracy:', 1-error_CF))

pred_raw_CF <- predict(unaware_CF, newdata=test_CF, type="prob")[,'1']



# Plot
male <- pred_raw[male_test_idx]
male_data <- data.frame(pred=male, type=as.factor(rep("original", length(male))))
male_CF <- pred_raw_CF[male_test_idx] 
male_data_CF <- data.frame(pred=male_CF, type=as.factor(rep("counterfactual", length(male_CF))))
male_compare <- rbind(male_data, male_data_CF)

female <- pred_raw[-male_test_idx]
female_data <- data.frame(pred=female, type=as.factor(rep("original", length(female))))
female_CF <- pred_raw_CF[-male_test_idx] 
female_data_CF <- data.frame(pred=female_CF, type=as.factor(rep("counterfactual", length(female_CF))))
female_compare <- rbind(female_data, female_data_CF)


# Comparison
cols = c("black", "red")
sm.options(col=cols, lty=c(1,2), lwd=2)

sm.density.compare(male_compare$pred, male_compare$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (M)")
legend("topright", legend=levels(male_compare$type), fill=cols)

sm.density.compare(female_compare$pred, female_compare$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (F)")
legend("topright", legend=levels(female_compare$type), fill=cols)

# Joined
original <- data.frame(pred=pred_raw, type=as.factor(rep("original", N_test)))
counterfactual <- data.frame(pred=pred_raw_CF, type=as.factor(rep("counterfactual", N_test)))
compare <- rbind(original, counterfactual)
sm.density.compare(compare$pred, compare$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (unaware)")
legend("topright", legend=levels(compare$type), fill=cols)

