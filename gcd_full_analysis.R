library(caret)
library(sm)

load("gcd_data_num.Rdata")
N <- dim(data)[1]

set.seed(0)
trainIndex <- createDataPartition(data$credit_risk, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- data[trainIndex,]
test <- data[-trainIndex,]

N_train <- dim(train)[1]
N_test <- dim(test)[1]

# Create counterfactual test set
test_CF <- test
male_test_idx <- which(test_CF$sex %in% '0')
test_CF$sex[male_test_idx] <- '1'
test_CF$sex[-male_test_idx] <- '0'

# Full model
full <- glm(credit_risk ~ . - housing - status - savings, family=binomial("logit"), data=train)

predictions_raw_full <- predict(full, newdata=test, type='response')
predictions_full <- ifelse(predictions_raw_full > 0.5, 1, 0)
error_full <- mean(predictions_full != test$credit_risk)
print(paste('Accuracy:', 1-error_full))

predictions_raw_full_CF <- predict(full, newdata=test_CF, type='response')
predictions_full_CF <- ifelse(predictions_raw_full_CF > 0.5, 1, 0)
error_full_CF <- mean(predictions_full_CF != test$credit_risk)
print(paste('Accuracy:', 1-error_full_CF))

# Plot
pred_m_full <- predictions_raw_full[male_test_idx]
m_compare_full <- data.frame(pred=pred_m_full, type=as.factor(rep("original", length(pred_m_full))))
pred_m_full_CF <- predictions_raw_full_CF[male_test_idx] 
m_compare_full_CF <- data.frame(pred=pred_m_full_CF, type=as.factor(rep("counterfactual", length(pred_m_full_CF))))
m_compare_unfair <- rbind(m_compare_full, m_compare_full_CF)

pred_f_full <- predictions_raw_full[-male_test_idx]
f_compare_full <- data.frame(pred=pred_f_full, type=as.factor(rep("original", length(pred_f_full))))
pred_f_full_CF <- predictions_raw_full_CF[-male_test_idx] 
f_compare_full_CF <- data.frame(pred=pred_f_full_CF, type=as.factor(rep("counterfactual", length(pred_f_full_CF))))
f_compare_unfair <- rbind(f_compare_full, f_compare_full_CF)

#orig_pred <- data.frame(pred=predictions_raw_full, type=as.factor(rep("original", length(predictions_raw_full))))
#cf_pred <- data.frame(pred=predictions_raw_full_CF, type=as.factor(rep("counterfactual", length(predictions_raw_full_CF))))
#compare_distr <- rbind(orig_pred, cf_pred)

# Comparison
sm.density.compare(m_compare_unfair$pred, m_compare_unfair$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (M)")
legend("topright", legend=levels(m_compare_unfair$type), fill=2+(0:nlevels(m_compare_unfair$type)))

sm.density.compare(f_compare_unfair$pred, f_compare_unfair$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (F)")
legend("topright", legend=levels(f_compare_unfair$type), fill=2+(0:nlevels(f_compare_unfair$type)))

#sm.density.compare(compare_distr$pred, compare_distr$type, xlab="Credit risk probability", model="equal")
#title("Density plot comparison of sex")
#legend("topright", legend=levels(compare_distr$type), fill=2+(0:nlevels(compare_distr$type)))
