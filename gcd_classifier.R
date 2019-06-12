require(rjags)
require(coda)
library(caret)
library(sm)

load("gcd_data_bin.Rdata")
N <- dim(data)[1]

set.seed(0)
trainIndex <- createDataPartition(data$credit_risk, p = .8, list = FALSE, times = 1)
#save(trainIndex, file="trainIndex.Rdata")
train <- data[trainIndex,]
test <- data[-trainIndex,]
N_train <- dim(train)[1]
N_test <- dim(test)[1]
male_test_idx <- which(test$sex %in% '0')

### Training the model
model = jags.model('gcd_model_train.jags',
                   data = list('N' = N_train, 'y' = train$credit_risk, 'a' = train$sex, 
                               'amt' = train$amount, 'dur' = train$duration,
                               'age' = train$age,
                               'hous' = train$housing, 'sav' = train$savings, 'stat' = train$status
                               #'nhous' = 2, 'nsav' = 4, 'nstat' = 3,
                               #'cut_hous' = cut_hous, 'cut_sav' = cut_sav, 'cut_stat' = cut_stat
                               ),
                   n.chains = 1,
                   n.adapt = 1000)
update(model, 10000)
samples = coda.samples(model, c('u', 
                                'amt0', 'amt_u', 'amt_a', 'amt_tau', 'amt_c',
                                'dur0', 'dur_u', 'dur_a', 'dur_tau', 'dur_c',
                                'hous0', 'hous_u', 'hous_a', 'hous_c',
                                'sav0', 'sav_u', 'sav_a', 'sav_c',
                                'stat0', 'stat_u', 'stat_a', 'stat_c'
                                ), 
                       n.iter = 10000,
                       thin = 2)
#save(samples, file='seed0_bin.Rdata')
#load(seed0_bin.Rdata)

params <- c("u[1]", "dur_u", "amt_u")
plot(samples[,params])
#gelman.diag(samples[,params])
#gelman.plot(samples[,params])

mcmcMat = as.matrix(samples, chains=FALSE )
means <- colMeans(mcmcMat)

amt0 <- means["amt0"]
amt_u <- means["amt_u"]
amt_a <- means["amt_a"]
amt_tau <- means["amt_tau"]
amt_c <- means["amt_c"]

dur0 <- means["dur0"]
dur_u <- means["dur_u"]
dur_a <- means["dur_a"]
dur_tau <- means["dur_tau"]
dur_c <- means["dur_c"]

hous0 <- means["hous0"]
hous_u <- means["hous_u"]
hous_a <- means["hous_a"]
hous_c <- means["hous_c"]

sav0 <- means["sav0"]
sav_u <- means["sav_u"]
sav_a <- means["sav_a"]
sav_c <- means["sav_c"]

stat0 <- means["stat0"]
stat_u <- means["stat_u"]
stat_a <- means["stat_a"]
stat_c <- means["stat_c"]

u_train <- means[23:length(means)]
#data_train_u <- data.frame(train, "u" = u_train)
#save(data_train_u, file='data_train_u.Rdata')


### Learning u for test data using learned parameters
model_test = jags.model('gcd_model_u.jags',
                   data = list('N' = N_test, 'a' = test$sex, 
                               'amt' = test$amount, 'dur' = test$duration,
                               'age' = test$age,
                               'amt0' = amt0, 'amt_u' = amt_u, 'amt_a' = amt_a, 'amt_tau' = amt_tau, 'amt_c' = amt_c,
                               'dur0' = dur0, 'dur_u' = dur_u, 'dur_a' = dur_a, 'dur_tau' = dur_tau, 'dur_c' = dur_c,
                               'hous0' = hous0, 'hous_u' = hous_u, 'hous_a' = hous_a, 'hous_c' = hous_c,
                               'sav0' = sav0, 'sav_u'= sav_u, 'sav_a' = sav_a, 'sav_c' = sav_c,
                               'stat0' = stat0, 'stat_u' = stat_u, 'stat_a' = stat_a, 'stat_c' = stat_c
                               # 'nhous' = 2, 'nsav' = 4, 'nstat' = 3,
                               # 'cut_hous' = cut_hous, 'cut_sav' = cut_sav, 'cut_stat' = cut_stat
                   ),
                   n.chains = 1,
                   n.adapt = 1000)
update(model_test, 10000)
samples_u = coda.samples(model_test, c('u'), n.iter = 10000)

mcmcMat_u = as.matrix(samples_u , chains=FALSE )
u_test <- colMeans(mcmcMat_u)


# Classifier
X <- data.frame(u=u_train, age=train$age, credit_risk=train$credit_risk)
X_te <- data.frame(u=u_test, age=test$age, credit_risk=test$credit_risk)
cv <- trainControl(method = "cv", number = 10)
X$credit_risk <- as.factor(X$credit_risk)
X_te$credit_risk <- as.factor(X_te$credit_risk)

#classifier <- glm(credit_risk ~ u + age, family=binomial("logit"), data=X)
classifier <- train(credit_risk ~ u + age, method="glm", data=X, family="binomial", trControl=cv)

pred <- predict(classifier, newdata=X_te)
#save(pred, file="pred_fair.Rdata")
confusionMatrix(data=pred, X_te$credit_risk, positive='1')

pred_raw <- predict(classifier, newdata=X_te, type="prob")[,'1']


# Statistical fairness
male_pred <- pred[male_test_idx] # male predictions
male_te <- test$credit_risk[male_test_idx] # male outcome
female_pred <- pred[-male_test_idx] # female predictions
female_te <- test$credit_risk[-male_test_idx] # female outcome

N_m <- length(male_pred) # number of males
N_f <- length(female_pred) # number of females

pos_m <- length(male_pred[male_pred==1]) # number of males predicted in positive class (1)
pos_f <- length(female_pred[female_pred==1]) # number of females predicted in positive class (1)

neg_m <- length(male_pred[male_pred==0]) # number of males predicted in negative class (0)
neg_f <- length(female_pred[female_pred==0]) # number of females predicted in negative class (0)

TP_m <- sum(male_pred[male_te == 1] == male_te[male_te == 1]) # male true positives
TP_f <- sum(female_pred[female_te == 1] == female_te[female_te == 1]) # female true positives

FP_m <- sum(male_pred[male_te == 0] != male_te[male_te == 0]) # male false positives
FP_f <- sum(female_pred[female_te == 0] != female_te[female_te == 0]) # female false positives

FN_m <- sum(male_pred[male_te == 1] != male_te[male_te == 1]) # male false negatives
FN_f = sum(female_pred[female_te == 1] != female_te[female_te == 1]) # female false negatives

TN_m <- sum(male_pred[male_te == 0] == male_te[male_te == 0]) # male true negatives
TN_f = sum(female_pred[female_te == 0] == female_te[female_te == 0]) # female true negatives


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
fisher.test(DP_mat, alternative="two.sided") # p = 0.1485

TPR_mat <- rbind(c(TP_m, FN_m), c(TP_f, FN_f))
fisher.test(TPR_mat, alternative="two.sided") # p = 0.3858

FPR_mat <- rbind(c(TN_m, FP_m), c(TN_f, FP_f))
fisher.test(FPR_mat, alternative="two.sided") # p = 1

ppv_mat <- rbind(c(TP_m, FP_m), c(TP_f, FP_f))
fisher.test(ppv_mat, alternative="two.sided") # p = 0.5238

npv_mat <- rbind(c(TN_m, FN_m), c(TN_f, FN_f))
fisher.test(npv_mat, alternative="two.sided") # p = 0.1176

oae_mat <- rbind(c(TP_m+TN_m, FP_m+FN_m), c(TP_f+TN_f, FP_f+FN_f))
fisher.test(oae_mat, alternative="two.sided") # p = 0.1801
#prop.test(x=c(TP_m+TN_m, TP_f+TN_f), n=c(N_m, N_f), alternative="two.sided") # p = 0.2268

# Balance for positive class
t.test(pred_raw[male_test_idx][male_te == 1], pred_raw[-male_test_idx][female_te == 1]) # p = 0.529
t.test(pred_raw[male_test_idx][male_te == 0], pred_raw[-male_test_idx][female_te == 0]) # p = 0.01718


# Model comparison: accuracy
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
