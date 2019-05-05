library(caret)
library(sm)

load("gcd_data_num.Rdata")
N <- dim(data)[1]

set.seed(42)
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

# unaware model
unaware <- glm(credit_risk ~ . - sex - housing - savings - status, family=binomial("logit"), data=train)

pred_raw <- predict(unaware, newdata=test, type='response')
pred <- ifelse(pred_raw > 0.5, 1, 0)
error <- mean(pred != test$credit_risk)
print(paste('Accuracy:', 1-error))

pred_raw_CF <- predict(unaware, newdata=test_CF, type='response')
pred_CF <- ifelse(pred_raw_CF > 0.5, 1, 0)
error_CF <- mean(pred_CF != test$credit_risk)
print(paste('Accuracy:', 1-error_CF))

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

