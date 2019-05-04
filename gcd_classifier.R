require(rjags)
require(coda)
library(caret)
library(sm)


load("gcd_data.Rdata")
N <- dim(data)[1]

# seed=0 results in only probabilities lower than 0.5, i.e. 'always' predicts good credit (bad if confusion matrix is taken into account)
# seed=1 overfits the data as the train accuracy >> test accuracy
set.seed(1)
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


### Training the model

model = jags.model('gcd_model_train.jags',
                   data = list('N' = N_train, 'y' = train$credit_risk, 'a' = train$sex, 
                               'amt' = train$amount, 'dur' = train$duration,
                               'age' = train$age
#                               'stat1' = train$status1, 'stat2' = train$status2, 'stat3' = train$status3, 'stat4' = train$status4
                               ),
                   n.chains = 4)

samples = coda.samples(model, c('u', 
                                'amt0', 'amt_u', 'amt_a', 'amt_tau', 'amt_c',
                                'dur0', 'dur_u', 'dur_a', 'dur_tau', 'dur_c'
#                                'stat1_u', 'stat1_a', 'stat2_u', 'stat2_a', 'stat3_u', 'stat3_a', 'stat4_u', 'stat4_a',
#                                'stat10', 'stat20', 'stat30', 'stat40'
#                                'y0', 'y_u', 'y_a', 'y_amt', 'y_dur', 'y_c'
                                ), 
                       n.iter = 10000)
#save(samples, file='seed2.Rdata')

mcmcMat = as.matrix(samples , chains=TRUE )
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

#y0 <- means["y0"]
#y_u <- means["y_u"]
#y_a <- means["y_a"]
#y_amt <- means["y_amt"]
#y_dur <- means["y_dur"]
#y_c <- means["y_c"]

#stat10 <- means["stat10"]
#stat20 <- means["stat20"]
#stat30 <- means["stat30"]
#stat40 <- means["stat40"]
#stat1_u <- means["stat1_u"]
#stat1_a <- means["stat1_a"]
#stat2_u <- means["stat2_u"]
#stat2_a <- means["stat2_a"]
#stat3_u <- means["stat3_u"]
#stat3_a <- means["stat3_a"]
#stat4_u <- means["stat4_u"]
#stat4_a <- means["stat4_a"]

u_train <- means[12:length(means)]
#u_train <- means[12:(length(means)-6)]


### Learning u for test data using learned parameters

model_test = jags.model('gcd_model_u.jags',
                   data = list('N' = N_test, 'a' = test$sex, 
                               'amt' = test$amount, 'dur' = test$duration,
                               'age' = test$age,
                               'amt0' = amt0, 'amt_u' = amt_u, 'amt_a' = amt_a, 'amt_tau' = amt_tau, 'amt_c' = amt_c,
                               'dur0' = dur0, 'dur_u' = dur_u, 'dur_a' = dur_a, 'dur_tau' = dur_tau, 'dur_c' = dur_c
#                               'stat1_u' = stat1_u, 'stat1_a' = stat1_a, 
#                               'stat2_u' = stat2_u, 'stat2_a' = stat2_a, 
#                               'stat3_u' = stat3_u, 'stat3_a' = stat3_a,
#                               'stat4_u' = stat4_u, 'stat4_a' = stat4_a,
#                               'stat10' = stat10, 'stat20' = stat20, 'stat30' = stat30, 'stat40' = stat40
#                               'y0' = y0, 'y_u' = y_u, 'y_a' = y_a, 'y_amt' = y_amt, 'y_dur' = y_dur, 'y_c' = y_c
                   ),
                   n.chains = 4)

samples_u = coda.samples(model_test, c('u'), n.iter = 10000)
# save(samples_u, file='mcmc_samples.Rdata')
mcmcMat_u = as.matrix(samples_u , chains=TRUE )
# u = mcmcMat[,"u"]
u_test <- colMeans(mcmcMat_u)
u_test <- u_test[2:length(u_test)]


### CF model
model_test_CF = jags.model('gcd_model_u.jags',
                        data = list('N' = N_test, 'a' = test_CF$sex, 
                                    'amt' = test_CF$amount, 'dur' = test_CF$duration,
                                    'age' = test_CF$age,
                                    'amt0' = amt0, 'amt_u' = amt_u, 'amt_a' = amt_a, 'amt_tau' = amt_tau, 'amt_c' = amt_c,
                                    'dur0' = dur0, 'dur_u' = dur_u, 'dur_a' = dur_a, 'dur_tau' = dur_tau, 'dur_c' = dur_c
#                                    'y0' = y0, 'y_u' = y_u, 'y_a' = y_a, 'y_amt' = y_amt, 'y_dur' = y_dur, 'y_c' = y_c
                        ),
                        n.chains = 4)
samples_u_CF = coda.samples(model_test_CF, c('u'), n.iter = 10000)
mcmcMat_u_CF = as.matrix(samples_u_CF , chains=TRUE )
# u = mcmcMat[,"u"]
u_test_CF <- colMeans(mcmcMat_u_CF)
u_test_CF <- u_test_CF[2:length(u_test_CF)]



# Classifier
X <- data.frame(u=u_train, age=train$age, credit_risk=train$credit_risk)
X_te <- data.frame(u=u_test, age=test$age, credit_risk=test$credit_risk)

classifier <- glm(credit_risk ~ u + age, family=binomial("logit"), data=X)

# Train accuracy
predictions_raw <- predict(classifier, type='response')
predictions <- ifelse(predictions_raw > 0.5, 1, 0)
error <- mean(predictions != train$credit_risk)
print(paste('Accuracy:', 1-error))

# Test accuracy
predictions_raw_te <- predict(classifier, newdata=X_te, type='response')
predictions_te <- ifelse(predictions_raw_te > 0.5, 1, 0)
error_te <- mean(predictions_te != test$credit_risk)
print(paste('Accuracy:', 1-error_te))

# CF Test
X_CF <- data.frame(u=u_test_CF, age=test_CF$age, credit_risk=test_CF$credit_risk)
predictions_raw_CF <- predict(classifier, newdata=X_CF, type='response')
predictions_CF <- ifelse(predictions_raw_CF > 0.5, 1, 0)
error_CF <- mean(predictions_CF != test_CF$credit_risk)
print(paste('Accuracy:', 1-error_CF))



# Plot
pred_m_te <- predictions_raw_te[male_test_idx]
m_compare_te <- data.frame(pred=pred_m_te, type=as.factor(rep("original", length(pred_m_te))))
pred_m_CF <- predictions_raw_CF[male_test_idx] 
m_compare_CF <- data.frame(pred=pred_m_CF, type=as.factor(rep("counterfactual", length(pred_m_CF))))
m_compare <- rbind(m_compare_te, m_compare_CF)

pred_f_te <- predictions_raw_te[-male_test_idx]
f_compare_te <- data.frame(pred=pred_f_te, type=as.factor(rep("original", length(pred_f_te))))
pred_f_CF <- predictions_raw_CF[-male_test_idx] 
f_compare_CF <- data.frame(pred=pred_f_CF, type=as.factor(rep("counterfactual", length(pred_f_CF))))
f_compare <- rbind(f_compare_te, f_compare_CF)

#orig_pred <- data.frame(pred=predictions_raw_te, type=as.factor(rep("original", length(predictions_raw_te))))
#cf_pred <- data.frame(pred=predictions_raw_CF, type=as.factor(rep("counterfactual", length(predictions_raw_CF))))
#compare_distr <- rbind(orig_pred, cf_pred)

# Comparison
sm.density.compare(m_compare$pred, m_compare$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (M)")
legend("topright", legend=levels(m_compare$type), fill=2+(0:nlevels(m_compare$type)))

sm.density.compare(f_compare$pred, f_compare$type, xlab="Credit risk probability", model="equal")
title("Density plot comparison of sex (F)")
legend("topright", legend=levels(f_compare$type), fill=2+(0:nlevels(f_compare$type)))

#sm.density.compare(compare_distr$pred, compare_distr$type, xlab="Credit risk probability", model="equal")
#title("Density plot comparison of sex")
#legend("topright", legend=levels(compare_distr$type), fill=2+(0:nlevels(compare_distr$type)))

