library(caret)

load("gcd_data_bin.Rdata")
N <- dim(data)[1]

set.seed(0)
trainIndex <- createDataPartition(data$credit_risk, p = .8, list = FALSE, times = 1)
train <- data[trainIndex,]
test <- data[-trainIndex,]
N_train <- dim(train)[1]
N_test <- dim(test)[1]

# Model comparison (seed=0): Accuracy
load("pred_fair.Rdata")
pred_fair <- pred
load("pred_full.Rdata")
pred_full <- pred
load("pred_unaware.Rdata")
pred_unaware <- pred
pred <- NULL # avoid issues

fair_correct <- which(pred_fair == test$credit_risk)
full_correct <- which(pred_full == test$credit_risk)
unaware_correct <- which(pred_unaware == test$credit_risk)

fair_wrong <- which(pred_fair != test$credit_risk)
full_wrong <- which(pred_full != test$credit_risk)
unaware_wrong <- which(pred_unaware != test$credit_risk)

# Fair vs Full
correct_fair_full <- length(which(fair_correct %in% full_correct)); correct_fair_full
wrong_fair_full <- length(which(fair_wrong %in% full_wrong)); wrong_fair_full
correct_fair_wrong_full <- length(which(fair_correct %in% full_wrong)); correct_fair_wrong_full
correct_full_wrong_fair <- length(which(full_correct %in% fair_wrong)); correct_full_wrong_fair

fair_full <- matrix(c(correct_fair_full, correct_full_wrong_fair, correct_fair_wrong_full, wrong_fair_full),
                    nrow = 2,
                    dimnames = list("fair" = c("correct", "wrong"),
                                    "full" = c("correct", "wrong"))); fair_full
mcnemar.test(fair_full)

# Fair vs Unaware
correct_fair_unaware <- length(which(fair_correct %in% unaware_correct)); correct_fair_unaware
wrong_fair_unaware <- length(which(fair_wrong %in% unaware_wrong)); wrong_fair_unaware
correct_fair_wrong_unaware <- length(which(fair_correct %in% unaware_wrong)); correct_fair_wrong_unaware
correct_unaware_wrong_fair <- length(which(unaware_correct %in% fair_wrong)); correct_unaware_wrong_fair

fair_unaware <- matrix(c(correct_fair_unaware, correct_unaware_wrong_fair, correct_fair_wrong_unaware, wrong_fair_unaware),
                       nrow = 2,
                       dimnames = list("fair" = c("correct", "wrong"),
                                       "unaware" = c("correct", "wrong"))); fair_unaware
mcnemar.test(fair_unaware)

# Full vs Unaware
correct_full_unaware <- length(which(full_correct %in% unaware_correct)); correct_full_unaware
wrong_full_unaware <- length(which(full_wrong %in% unaware_wrong)); wrong_full_unaware
correct_full_wrong_unaware <- length(which(full_correct %in% unaware_wrong)); correct_full_wrong_unaware
correct_unaware_wrong_full <- length(which(unaware_correct %in% full_wrong)); correct_unaware_wrong_full

full_unaware <- matrix(c(correct_full_unaware, correct_unaware_wrong_full, correct_full_wrong_unaware, wrong_full_unaware),
                       nrow = 2,
                       dimnames = list("full" = c("correct", "wrong"),
                                       "unaware" = c("correct", "wrong"))); full_unaware
mcnemar.test(full_unaware)
