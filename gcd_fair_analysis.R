library(caret)
library(sm)

load("data_train_u.Rdata")
N <- dim(data_train_u)[1]
data_train_u$credit_risk <- as.factor(data_train_u$credit_risk)
train <- data_train_u

fair <- glm(credit_risk ~ u + age, family=binomial("logit"), data=train)

# Samples with original sex
load("data_samples_og.Rdata")
pred_raw <- predict(fair, newdata=data_og, type="response")

# Samples with counterfactual sex
load("data_samples_cf.Rdata")
pred_raw_CF <- predict(fair, newdata=data_cf, type="response")

ks.test(pred_raw, pred_raw_CF)

# Density plot for counterfactual fairness
cols = c("black", "red")
sm.options(col=cols, lty=c(1,2), lwd=2)

original <- data.frame(pred=pred_raw, type=as.factor(rep("original", 1000)))
counterfactual <- data.frame(pred=pred_raw_CF, type=as.factor(rep("counterfactual", 1000)))
compare <- rbind(original, counterfactual)
sm.density.compare(compare$pred, compare$type, xlab="Credit risk probability")
title("Density plot comparison of sex (fair)")
legend("topright", legend=levels(compare$type), fill=cols)
