require(rjags)
require(coda)
library(caret)
library(doBy)

load("gcd_data_bin.Rdata")
N <- dim(data)[1]

set.seed(4) # default 0 (TO-DO: 3, 4)
trainIndex <- createDataPartition(data$credit_risk, p = .8, list = FALSE, times = 1)
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
#save(samples, file='seedX.Rdata')
#load("seed0.Rdata")

#params <- c("u[1]", "dur_u", "amt_u")
#plot(samples[,params])
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
#train_u_X <- data.frame(train, "u" = u_train)
#save(train_u_X, file='train_u_X.Rdata')


### Learn U for test data using learned parameters
model_test = jags.model('gcd_model_u.jags',
                   data = list('N' = N_test, 'a' = test$sex, 
                               'amt' = test$amount, 'dur' = test$duration,
                               'age' = test$age,
                               'amt0' = amt0, 'amt_u' = amt_u, 'amt_a' = amt_a, 'amt_tau' = amt_tau, 'amt_c' = amt_c,
                               'dur0' = dur0, 'dur_u' = dur_u, 'dur_a' = dur_a, 'dur_tau' = dur_tau, 'dur_c' = dur_c,
                               'hous0' = hous0, 'hous_u' = hous_u, 'hous_a' = hous_a, 'hous_c' = hous_c,
                               'sav0' = sav0, 'sav_u'= sav_u, 'sav_a' = sav_a, 'sav_c' = sav_c,
                               'stat0' = stat0, 'stat_u' = stat_u, 'stat_a' = stat_a, 'stat_c' = stat_c
                   ),
                   n.chains = 1,
                   n.adapt = 1000)
update(model_test, 10000)
samples_u = coda.samples(model_test, c('u'), n.iter = 10000)

mcmcMat_u = as.matrix(samples_u , chains=FALSE )
u_test <- colMeans(mcmcMat_u)
#test_u_X <- data.frame(test, "u" = u_test)
#save(test_u_X, file='test_u_X.Rdata')


# Classifier
X <- data.frame(u=u_train, age=train$age, credit_risk=train$credit_risk)
X_te <- data.frame(u=u_test, age=test$age, credit_risk=test$credit_risk)
X$credit_risk <- as.factor(X$credit_risk)
X_te$credit_risk <- as.factor(X_te$credit_risk)

classifier <- glm(credit_risk ~ u + age, family=binomial("logit"), data=X)

pred_raw <- predict(classifier, newdata=X_te, type='response')
max_risk_idx <- which.maxn(pred_raw, 60)
pred <- pred_raw
pred[max_risk_idx] <- 1
pred[-max_risk_idx] <- 0
pred <- as.factor(pred)
#save(pred, file="pred_fair.Rdata")
confusionMatrix(data=pred, X_te$credit_risk, positive='1')



# Statistical fairness
male_pred <- pred[male_test_idx] # male predictions
male_te <- test$credit_risk[male_test_idx] # male outcome
female_pred <- pred[-male_test_idx] # female predictions
female_te <- test$credit_risk[-male_test_idx] # female outcome

N_m <- length(male_pred); N_m # number of males
N_f <- length(female_pred); N_f # number of females

pos_m <- length(male_pred[male_pred==1]); pos_m # number of males predicted in positive class (1)
pos_f <- length(female_pred[female_pred==1]); pos_f # number of females predicted in positive class (1)

neg_m <- length(male_pred[male_pred==0]); neg_m # number of males predicted in negative class (0)
neg_f <- length(female_pred[female_pred==0]); neg_f # number of females predicted in negative class (0)

TP_m <- sum(male_pred[male_te == 1] == male_te[male_te == 1]); TP_m # male true positives
TP_f <- sum(female_pred[female_te == 1] == female_te[female_te == 1]); TP_f # female true positives

FP_m <- sum(male_pred[male_te == 0] != male_te[male_te == 0]); FP_m # male false positives
FP_f <- sum(female_pred[female_te == 0] != female_te[female_te == 0]); FP_f # female false positives

FN_m <- sum(male_pred[male_te == 1] != male_te[male_te == 1]); FN_m # male false negatives
FN_f = sum(female_pred[female_te == 1] != female_te[female_te == 1]); FN_f # female false negatives

TN_m <- sum(male_pred[male_te == 0] == male_te[male_te == 0]); TN_m # male true negatives
TN_f <- sum(female_pred[female_te == 0] == female_te[female_te == 0]); TN_f # female true negatives


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
fisher.test(DP_mat, alternative="two.sided") # p = 0.1367

TPR_mat <- rbind(c(TP_m, FN_m), c(TP_f, FN_f))
fisher.test(TPR_mat, alternative="two.sided") # p = 0.7898

FPR_mat <- rbind(c(TN_m, FP_m), c(TN_f, FP_f))
fisher.test(FPR_mat, alternative="two.sided") # p = 2011

ppv_mat <- rbind(c(TP_m, FP_m), c(TP_f, FP_f))
fisher.test(ppv_mat, alternative="two.sided") # p = 0.5916

npv_mat <- rbind(c(TN_m, FN_m), c(TN_f, FN_f))
fisher.test(npv_mat, alternative="two.sided") # p = 0.08966

oae_mat <- rbind(c(TP_m+TN_m, FP_m+FN_m), c(TP_f+TN_f, FP_f+FN_f))
fisher.test(oae_mat, alternative="two.sided") # p = 0.2057

# Balance for positive class
t.test(pred_raw[male_test_idx][male_te == 1], pred_raw[-male_test_idx][female_te == 1]) # p = 0.5401
t.test(pred_raw[male_test_idx][male_te == 0], pred_raw[-male_test_idx][female_te == 0]) # p = 0.01695
