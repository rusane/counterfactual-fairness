library(caret)
library(sm)

# Original sampled data
load("data_y_og.Rdata")
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

fair <- train(credit_risk ~ u + age, method="glm", data=train, family="binomial", trControl=cv)

pred <- predict(fair, newdata=test)
confusionMatrix(data=pred, test$credit_risk, positive='1')

pred_raw <- predict(fair, newdata=test, type="prob")[,'1']

#pred_raw <- predict(fair, newdata=test, type='response')
#pred <- ifelse(pred_raw > 0.5, 1, 0)
#error <- mean(pred != test$credit_risk)
#print(paste('Accuracy:', 1-error))

error <- mean(pred != test$credit_risk)
print(paste('Accuracy:', 1-error))

TN <- sum(pred[test$credit_risk == 0] == test$credit_risk[test$credit_risk == 0])
TP <- sum(pred[test$credit_risk == 1] == test$credit_risk[test$credit_risk == 1])
FN <- sum(pred[test$credit_risk == 1] != test$credit_risk[test$credit_risk == 1])
FP <- sum(pred[test$credit_risk == 0] != test$credit_risk[test$credit_risk == 0])



# Counterfactual sampled data
load("data_y_cf.Rdata")
N_CF <- dim(data_cf)[1]
data_cf$credit_risk <- as.factor(data_cf$credit_risk)

set.seed(0)
trainIndex_CF <- createDataPartition(data_cf$credit_risk, p = .8, 
                                     list = FALSE, 
                                     times = 1)
train_CF <- data_cf[trainIndex_CF,]
test_CF <- data_cf[-trainIndex_CF,]
N_train_CF <- dim(train_CF)[1]
N_test_CF <- dim(test_CF)[1]
male_test_idx_CF <- which(test_CF$sex %in% '0')
cv <- trainControl(method = "cv", number = 10)

fair_CF <- train(credit_risk ~ u + age, method="glm", data=train_CF, family="binomial", trControl=cv)

pred_CF <- predict(fair_CF, newdata=test_CF)
confusionMatrix(data=pred_CF, test_CF$credit_risk, positive='1')

pred_raw_CF <- predict(fair_CF, newdata=test_CF, type="prob")[,'1']

#pred_raw_CF <- predict(fair_CF, newdata=test_CF, type='response')
#pred_CF <- ifelse(pred_raw_CF > 0.5, 1, 0)
#error_CF <- mean(pred_CF != test$credit_risk)
#print(paste('Accuracy:', 1-error_CF))

error_CF <- mean(pred_CF != test_CF$credit_risk)
print(paste('Accuracy:', 1-error_CF))




# Statistical fairness
male_pred <- pred[male_test_idx]
male_te <- test$credit_risk[male_test_idx]
female_pred <- pred[-male_test_idx]
female_te <- test$credit_risk[-male_test_idx]

demographic_parity_m <- length(male_pred[male_pred==1])/length(male_pred); demographic_parity_m
demographic_parity_f <- length(female_pred[female_pred==1])/length(female_pred); demographic_parity_f

# TPR = TP / (TP + FN)
male_TP <- sum(male_pred[male_te == 1] == male_te[male_te == 1])
male_TPR <- male_TP / length(male_te[male_te==1]); male_TPR
female_TP <- sum(female_pred[female_te == 1] == female_te[female_te == 1])
female_TPR <- female_TP / length(female_te[female_te==1]); female_TPR

# FPR = FP / (FP + TN)
male_FP <- sum(male_pred[male_te == 0] != male_te[male_te == 0])
male_FPR <- male_FP / length(male_te[male_te==0]); male_FPR
female_FP <- sum(female_pred[female_te == 0] != female_te[female_te == 0])
female_FPR <- female_FP / length(female_te[female_te==0]); female_FPR


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
