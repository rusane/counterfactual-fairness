library(caret)
library(sm)

# Original sampled data
load("data_samples_og.Rdata")
N <- dim(data_og)[1]

set.seed(10)
trainIndex <- createDataPartition(data_og$credit_risk, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- data_og[trainIndex,]
test <- data_og[-trainIndex,]
N_train <- dim(train)[1]
N_test <- dim(test)[1]
male_test_idx <- which(test$sex %in% '0')

full <- glm(credit_risk ~ . - u, family=binomial("logit"), data=train)

pred_raw <- predict(full, newdata=test, type='response')
pred <- ifelse(pred_raw > 0.5, 1, 0)
error <- mean(pred != test$credit_risk)
print(paste('Accuracy:', 1-error))


# Counterfactual sampled data
load("data_samples_cf.Rdata")
N_CF <- dim(data_cf)[1]

set.seed(0)
trainIndex_CF <- createDataPartition(data_cf$credit_risk, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train_CF <- data_cf[trainIndex_CF,]
test_CF <- data_cf[-trainIndex_CF,]
N_train_CF <- dim(train_CF)[1]
N_test_CF <- dim(test_CF)[1]
male_test_idx_CF <- which(test_CF$sex %in% '0')

full_CF <- glm(credit_risk ~ . - u, family=binomial("logit"), data=train_CF)

pred_raw_CF <- predict(full_CF, newdata=test_CF, type='response')
pred_CF <- ifelse(pred_raw_CF > 0.5, 1, 0)
error_CF <- mean(pred_CF != test$credit_risk)
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
sm.options(col=cols, lty=c(1,1), lwd=2)

sm.density.compare(m_compare_unfair$pred, m_compare_unfair$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (M)")
legend("topright", legend=levels(m_compare_unfair$type), fill=cols)

sm.density.compare(f_compare_unfair$pred, f_compare_unfair$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (F)")
legend("topright", legend=levels(f_compare_unfair$type), fill=cols)


#sm.density.compare(compare_distr$pred, compare_distr$type, xlab="Credit risk probability", model="equal")
#title("Density plot comparison of sex")
#legend("topright", legend=levels(compare_distr$type), fill=2+(0:nlevels(compare_distr$type)))
