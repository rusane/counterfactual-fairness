library(caret)
library(sm)

# Baseline evaluation
load("gcd_data_norm.Rdata")
N <- dim(data)[1]
#data$credit_risk <- as.factor(data$credit_risk)

set.seed(0)
trainIndex <- createDataPartition(data$credit_risk, p = .8, 
                                  list = FALSE, 
                                  times = 1)
data$credit_risk <- as.factor(data$credit_risk)
#load("trainIndex.Rdata")
train <- data[trainIndex,]
test <- data[-trainIndex,]
N_train <- dim(train)[1]
N_test <- dim(test)[1]
cv <- trainControl(method = "cv", number = 10)

#full <- glm(credit_risk ~ ., family=binomial("logit"), data=train)
full <- train(credit_risk ~ ., method="glm", data=train, family="binomial", trControl=cv)

pred <- predict(full, newdata=test)
#save(pred, file="pred_full.Rdata")
confusionMatrix(data=pred, test$credit_risk, positive='1')



# Samples with original sex
load("data_samples_og.Rdata")
data_og$credit_risk <- as.factor(data_og$credit_risk)

pred <- predict(full, newdata=data_og)
confusionMatrix(data=pred, data_og$credit_risk, positive='1')
pred_raw <- predict(full, newdata=data_og, type="prob")[,'1']

# Samples with counterfactual sex
load("data_samples_cf.Rdata")
data_cf$credit_risk <- as.factor(data_cf$credit_risk)

pred_CF <- predict(full, newdata=data_cf)
confusionMatrix(data=pred_CF, data_cf$credit_risk, positive='1')
pred_raw_CF <- predict(full, newdata=data_cf, type="prob")[,'1']



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

#full <- glm(credit_risk ~ . - u, family=binomial("logit"), data=train)
full <- train(credit_risk ~ . - u, method="glm", data=train, family="binomial", trControl=cv)

pred <- predict(full, newdata=test)
confusionMatrix(data=pred, test$credit_risk, positive='1')

pred_raw <- predict(full, newdata=test, type="prob")[,'1']

# Counterfactual sampled data
load("data_samples_cf.Rdata")
N_CF <- dim(data_cf)[1]
data_cf$credit_risk <- as.factor(data_cf$credit_risk)

set.seed(0)
#trainIndex_CF <- createDataPartition(data_cf$credit_risk, p = .8, 
#                                  list = FALSE, 
#                                  times = 1)
train_CF <- data_cf[trainIndex,]
test_CF <- data_cf[-trainIndex,]
N_train_CF <- dim(train_CF)[1]
N_test_CF <- dim(test_CF)[1]
male_test_idx_CF <- which(test_CF$sex %in% '0')
cv <- trainControl(method = "cv", number = 10)

#full_CF <- glm(credit_risk ~ . - u, family=binomial("logit"), data=train_CF)
full_CF <- train(credit_risk ~ . - u, method="glm", data=train_CF, family="binomial", trControl=cv)

pred_CF <- predict(full_CF, newdata=test_CF)
confusionMatrix(data=pred_CF, test$credit_risk, positive='1')

pred_raw_CF <- predict(full_CF, newdata=test_CF, type="prob")[,'1']


# Plot
pred_m_full <- pred_raw[male_test_idx]
m_compare_full <- data.frame(pred=pred_m_full, type=as.factor(rep("original", length(pred_m_full))))
pred_m_full_CF <- pred_raw_CF[male_test_idx] 
m_compare_full_CF <- data.frame(pred=pred_m_full_CF, type=as.factor(rep("counterfactual", length(pred_m_full_CF))))
m_compare_unfair <- rbind(m_compare_full, m_compare_full_CF)

pred_f_full <- pred_raw[-male_test_idx]
f_compare_full <- data.frame(pred=pred_f_full, type=as.factor(rep("original", length(pred_f_full))))
pred_f_full_CF <- pred_raw_CF[-male_test_idx] 
f_compare_full_CF <- data.frame(pred=pred_f_full_CF, type=as.factor(rep("counterfactual", length(pred_f_full_CF))))
f_compare_unfair <- rbind(f_compare_full, f_compare_full_CF)


#orig_pred <- data.frame(pred=pred_raw, type=as.factor(rep("original", length(pred_raw))))
#cf_pred <- data.frame(pred=pred_raw_CF, type=as.factor(rep("counterfactual", length(pred_raw_CF))))
#compare_distr <- rbind(orig_pred, cf_pred)

# Comparison
cols = c("black", "red")
sm.options(col=cols, lty=c(1,2), lwd=2)

sm.density.compare(m_compare_unfair$pred, m_compare_unfair$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (M)")
legend("topright", legend=levels(m_compare_unfair$type), fill=cols)

sm.density.compare(f_compare_unfair$pred, f_compare_unfair$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (F)")
legend("topright", legend=levels(f_compare_unfair$type), fill=cols)

original <- data.frame(pred=pred_raw, type=as.factor(rep("original", N_test)))
counterfactual <- data.frame(pred=pred_raw_CF, type=as.factor(rep("counterfactual", N_test)))
compare <- rbind(original, counterfactual)
sm.density.compare(compare$pred, compare$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (full)")
legend("topright", legend=levels(compare$type), fill=cols)


#sm.density.compare(compare_distr$pred, compare_distr$type, xlab="Credit risk probability", model="equal")
#title("Density plot comparison of sex")
#legend("topright", legend=levels(compare_distr$type), fill=2+(0:nlevels(compare_distr$type)))
