require(rjags)
require(coda)
library(caret)
library(sm)

load("gcd_data_bin.Rdata")
N <- dim(data)[1]

#data$sex <- as.factor(data$sex)
#data$status <- as.factor(data$status)
#data$savings <- as.factor(data$savings)
#data$housing <- as.factor(data$housing)

#polr(formula = housing ~ age + sex, data = data) # -2.260, -1.051
#polr(formula = savings ~ age + sex, data = data) # -1.472, 1.335, 2.115, 3.021
#polr(formula = status ~ age + sex, data = data) # -0.420, 0.716, 2.720


set.seed(0)
trainIndex <- createDataPartition(data$credit_risk, p = .8, list = FALSE, times = 1)
train <- data[trainIndex,]
test <- data[-trainIndex,]
N_train <- dim(train)[1]
N_test <- dim(test)[1]
male_test_idx <- which(test$sex %in% '0')

### Training the model
#load.module("glm")

#cut_hous = c(-0.1, 0.1)
#cut_sav = c(-0.3, -0.1, 0.1, 0.3)
#cut_stat = c(-0.2, 0, 0.2)
#cut_hous = c(-2.260, -1.051)
#cut_sav = c(-1.472, 1.335, 2.115, 3.021)
#cut_stat = c(-0.420, 0.716, 2.720)

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

#params_u <- c("u[10]")
#plot(samples_u[,params_u])
#gelman.diag(samples_u[,params_u])
#gelman.plot(samples_u[,params_u])

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
confusionMatrix(data=pred, X_te$credit_risk, positive='1')

pred_raw <- predict(classifier, newdata=X_te, type="prob")[,'1']

# Train accuracy
#predictions_raw <- predict(classifier, type='response')
#predictions <- ifelse(predictions_raw > 0.5, 1, 0)
#error <- mean(predictions != train$credit_risk)
#print(paste('Accuracy:', 1-error))

# Test accuracy
#predictions_raw_te <- predict(classifier, newdata=X_te, type='response')
#predictions_te <- ifelse(predictions_raw_te > 0.5, 1, 0)
#error_te <- mean(predictions_te != test$credit_risk)
#print(paste('Accuracy:', 1-error_te))


# Plot
#pred_m_te <- predictions_raw_te[male_test_idx]
#m_compare_te <- data.frame(pred=pred_m_te, type=as.factor(rep("original", length(pred_m_te))))
#pred_m_CF <- predictions_raw_CF[male_test_idx] 
#m_compare_CF <- data.frame(pred=pred_m_CF, type=as.factor(rep("counterfactual", length(pred_m_CF))))
#m_compare <- rbind(m_compare_te, m_compare_CF)

#pred_f_te <- predictions_raw_te[-male_test_idx]
#f_compare_te <- data.frame(pred=pred_f_te, type=as.factor(rep("original", length(pred_f_te))))
#pred_f_CF <- predictions_raw_CF[-male_test_idx] 
#f_compare_CF <- data.frame(pred=pred_f_CF, type=as.factor(rep("counterfactual", length(pred_f_CF))))
#f_compare <- rbind(f_compare_te, f_compare_CF)


# Comparison
#cols = c("black", "red")
#sm.options(col=cols, lty=c(1,2), lwd=2)

#sm.density.compare(m_compare$pred, m_compare$type, xlab="Credit risk probability", model="equal")
#title("Density plot comparison of sex (M)")
#legend("topright", legend=levels(m_compare$type), fill=2+(0:nlevels(m_compare$type)))

#sm.density.compare(f_compare$pred, f_compare$type, xlab="Credit risk probability", model="equal")
#title("Density plot comparison of sex (F)")
#legend("topright", legend=levels(f_compare$type), fill=2+(0:nlevels(f_compare$type)))


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

# Calibration
calibration_m <- mean(pred_raw[male_test_idx])
calibration_f <- mean(pred_raw[-male_test_idx])


#N_m <- length(male_pred)
#N_f <- length(female_pred)
good_m <- length(male_pred[male_pred==0])
good_f <- length(female_pred[female_pred==0])
bad_m <- length(male_pred[male_pred==1])
bad_f <- length(female_pred[female_pred==1])

DP_mat <- rbind(c(good_m, bad_m), c(good_f, bad_f))
fisher.test(DP_mat, alternative="two.sided") # p = 0.1485

# TP + FN = Positives
male_FN <- sum(male_pred[male_te == 1] != male_te[male_te == 1])
female_FN = sum(female_pred[female_te == 1] != female_te[female_te == 1])

TPR_mat <- rbind(c(male_TP, male_FN), c(female_TP, female_FN))
fisher.test(TPR_mat, alternative="two.sided")

# TN + FP = Negatives
male_TN <- sum(male_pred[male_te == 0] == male_te[male_te == 0])
female_TN = sum(female_pred[female_te == 0] == female_te[female_te == 0])

FPR_mat <- rbind(c(male_TN, male_FP), c(female_TN, female_FP))
fisher.test(FPR_mat, alternative="two.sided")

t.test(pred_raw[male_test_idx], pred_raw[-male_test_idx]) # 0.007466
