library(caret)
library(sm)
library(doBy)

# Baseline evaluation
load("gcd_data_bin.Rdata")
N <- dim(data)[1]

set.seed(0)
trainIndex <- createDataPartition(data$credit_risk, p = .8, 
                                  list = FALSE, 
                                  times = 1)
data$credit_risk <- as.factor(data$credit_risk)
train <- data[trainIndex,]
test <- data[-trainIndex,]
N_train <- dim(train)[1]
N_test <- dim(test)[1]

unaware <- glm(credit_risk ~ . - sex, family=binomial("logit"), data=train)

pred_raw <- predict(unaware, newdata=test, type='response')
max_risk_idx <- which.maxn(pred_raw, 60)
pred <- pred_raw
pred[max_risk_idx] <- 1
pred[-max_risk_idx] <- 0
pred <- as.factor(pred)
#save(pred, file="pred_unaware.Rdata")
confusionMatrix(data=pred, test$credit_risk, positive='1')



# Samples with original sex
load("data_samples_og.Rdata")
pred_raw <- predict(unaware, newdata=data_og, type="response")

# Samples with counterfactual sex
load("data_samples_cf.Rdata")
pred_raw_CF <- predict(unaware, newdata=data_cf, type="response")

ks.test(pred_raw, pred_raw_CF)

# Density plot for counterfactual fairness
cols = c("black", "red")
sm.options(col=cols, lty=c(1,2), lwd=2)

original <- data.frame(pred=pred_raw, type=as.factor(rep("original", N_test)))
counterfactual <- data.frame(pred=pred_raw_CF, type=as.factor(rep("counterfactual", N_test)))
compare <- rbind(original, counterfactual)
sm.density.compare(compare$pred, compare$type, xlab="Credit risk probability")
title("Density plot comparison of sex (unaware)")
legend("topright", legend=levels(compare$type), fill=cols)
