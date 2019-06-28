library(caret)
library(sm)
library(doBy)

load("train_u_0.Rdata")
N <- dim(train_u_0)[1]
train_u_0$credit_risk <- as.factor(train_u_0$credit_risk)
train <- train_u_0

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



### Statistical fairness
load("train_u_0.Rdata"); load("train_u_1.Rdata"); load("train_u_2.Rdata"); load("train_u_3.Rdata"); load("train_u_4.Rdata")
load("test_u_0.Rdata"); load("test_u_1.Rdata"); load("test_u_2.Rdata"); load("test_u_3.Rdata"); load("test_u_4.Rdata")
trainList <- list(train_u_0, train_u_1, train_u_2, train_u_3, train_u_4)
testList <- list(test_u_0, test_u_1, test_u_2, test_u_3, test_u_4)

N_m <- N_f <- pos_m <- pos_f <- neg_m <- neg_f <- TP_m <- TP_f <- FP_m <- FP_f <- FN_m <- FN_f <- TN_m <- TN_f <- 0
positive_balance_m <- positive_balance_f <- negative_balance_m <- negative_balance_f <- vector()
for (i in 1:5) { # Start with 1 because of how indices work in R
  train <- trainList[[i]]
  test <- testList[[i]]
  male_test_idx <- which(test$sex %in% '0')
  
  fair <- glm(credit_risk ~ u + age, family=binomial("logit"), data=train)
  
  pred_raw <- predict(fair, newdata=test, type='response')
  max_risk_idx <- which.maxn(pred_raw, 60)
  pred <- pred_raw
  pred[max_risk_idx] <- 1
  pred[-max_risk_idx] <- 0
  pred <- as.factor(pred)
  
  male_pred <- pred[male_test_idx] # male predictions
  male_te <- test$credit_risk[male_test_idx] # male outcome
  female_pred <- pred[-male_test_idx] # female predictions
  female_te <- test$credit_risk[-male_test_idx] # female outcome
  
  N_m <- N_m + length(male_pred) # number of males
  N_f <- N_f + length(female_pred) # number of females
  
  pos_m <- pos_m + length(male_pred[male_pred==1])# number of males predicted in positive class (1)
  pos_f <- pos_f + length(female_pred[female_pred==1]) # number of females predicted in positive class (1)
  
  neg_m <- neg_m + length(male_pred[male_pred==0]); neg_m # number of males predicted in negative class (0)
  neg_f <- neg_f + length(female_pred[female_pred==0]) # number of females predicted in negative class (0)
  
  TP_m <- TP_m + sum(male_pred[male_te == 1] == male_te[male_te == 1]) # male true positives
  TP_f <- TP_f + sum(female_pred[female_te == 1] == female_te[female_te == 1]) # female true positives
  
  FP_m <- FP_m +  sum(male_pred[male_te == 0] != male_te[male_te == 0]); FP_m # male false positives
  FP_f <- FP_f + sum(female_pred[female_te == 0] != female_te[female_te == 0]) # female false positives
  
  FN_m <- FN_m + sum(male_pred[male_te == 1] != male_te[male_te == 1]); FN_m # male false negatives
  FN_f <- FN_f + sum(female_pred[female_te == 1] != female_te[female_te == 1]) # female false negatives
  
  TN_m <- TN_m + sum(male_pred[male_te == 0] == male_te[male_te == 0]); TN_m # male true negatives
  TN_f <- TN_f + sum(female_pred[female_te == 0] == female_te[female_te == 0]) # female true negatives
  
  positive_balance_m <- c(positive_balance_m, pred_raw[male_test_idx][male_te == 1])
  positive_balance_f <- c(positive_balance_f, pred_raw[-male_test_idx][female_te == 1])
  
  negative_balance_m <- c(negative_balance_m, pred_raw[male_test_idx][male_te == 0])
  negative_balance_f <- c(negative_balance_f, pred_raw[-male_test_idx][female_te == 0])
}

# Demographic Parity
demographic_parity_m <- pos_m/N_m; demographic_parity_m
demographic_parity_f <- pos_f/N_f; demographic_parity_f

# TPR = TP / (TP + FN)
TPR_m <- TP_m / (TP_m + FN_m); TPR_m
TPR_f <- TP_f / (TP_f + FN_f); TPR_f

# FPR = FP / (FP + TN)
FPR_m <- FP_m / (FP_m + TN_m); FPR_m
FPR_f <- FP_f / (FP_f + TN_f); FPR_f

# Positive Predictive Value (PPV) = TP / (TP + FP)
ppv_m <- TP_m / pos_m; ppv_m
ppv_f <- TP_f / pos_f; ppv_f

# Negative Predictive Value (NPV) = TN / (TN + FN)
npv_m <- TN_m / neg_m; npv_m
npv_f <- TN_f / neg_f; npv_f

# Overall accuracy equality
oae_m <-  (TP_m + TN_m) / N_m; oae_m
oae_f <-  (TP_f + TN_f) / N_f; oae_f

# Balance for positive class
bpc_m <- mean(pred_raw[male_test_idx][male_te == 1]); bpc_m
bpc_f <- mean(pred_raw[-male_test_idx][female_te == 1]); bpc_f

# Balance for negative class
bnc_m <- mean(pred_raw[male_test_idx][male_te == 0]); bnc_m
bnc_f <- mean(pred_raw[-male_test_idx][female_te == 0]); bnc_f


# Signficance testing (Fisher Exact test)
DP_mat <- rbind(c(neg_m, pos_m), c(neg_f, pos_f))
fisher.test(DP_mat, alternative="two.sided")

TPR_mat <- rbind(c(TP_m, FN_m), c(TP_f, FN_f))
fisher.test(TPR_mat, alternative="two.sided") 

FPR_mat <- rbind(c(TN_m, FP_m), c(TN_f, FP_f))
fisher.test(FPR_mat, alternative="two.sided") 

ppv_mat <- rbind(c(TP_m, FP_m), c(TP_f, FP_f))
fisher.test(ppv_mat, alternative="two.sided") 

npv_mat <- rbind(c(TN_m, FN_m), c(TN_f, FN_f))
fisher.test(npv_mat, alternative="two.sided") 
oae_mat <- rbind(c(TP_m+TN_m, FP_m+FN_m), c(TP_f+TN_f, FP_f+FN_f))
fisher.test(oae_mat, alternative="two.sided")

# t-test for balance for positive class or negative class
t.test(pred_raw[male_test_idx][male_te == 1], pred_raw[-male_test_idx][female_te == 1])
t.test(pred_raw[male_test_idx][male_te == 0], pred_raw[-male_test_idx][female_te == 0])


# False Negative Rate (FNR) [to show incompatibility between conditional use accuracy equality and equality in FNR and FPR]
FNR_m <- FN_m / (TP_m + FN_m); FNR_m
FNR_f <- FN_f / (TP_f + FN_f); FNR_f

FNR_mat <- rbind(c(FN_m, TP_m), c(FN_f, TP_f))
fisher.test(FNR_mat, alternative="two.sided")
