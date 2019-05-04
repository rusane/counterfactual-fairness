require(rjags)
require(coda)
library(caret)
library(sm)


load("gcd_data.Rdata")
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
                       n.iter = 2000) # increase iterations if necessary for final model
#save(samples, file='mcmc_samples.Rdata')
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

samples_u = coda.samples(model_test, c('u'), n.iter = 2000) # increase iterations if necessary for final model
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
samples_u_CF = coda.samples(model_test_CF, c('u'), n.iter = 2000) # increase iterations if necessary for final model
# save(samples_u, file='mcmc_samples.Rdata')
mcmcMat_u_CF = as.matrix(samples_u_CF , chains=TRUE )
# u = mcmcMat[,"u"]
u_test_CF <- colMeans(mcmcMat_u_CF)
u_test_CF <- u_test_CF[2:length(u_test_CF)]



# Now it just predicts every record to be a good credit risk because that's the majority (700 -> 0.7 accuracy)
# Predictions never more than 0.5 probability
# Without the categorical variable, the accuracy is much higher 0.935 so probably should look into that
X <- data.frame(u=u_train, age=train$age, credit_risk=train$credit_risk)
X_n <- data.frame(u=u_test, age=test$age, credit_risk=test$credit_risk)

model <- glm(credit_risk ~ u + age, family=binomial("logit"), data=X)

# Train accuracy
predictions_raw <- predict(model, type='response')
predictions <- ifelse(predictions_raw > 0.5, 1, 0)
error <- mean(predictions != train$credit_risk)
print(paste('Accuracy:', 1-error))

# Test accuracy
predictions_raw_n <- predict(model, newdata=X_n, type='response')
predictions_n <- ifelse(predictions_raw_n > 0.5, 1, 0)
error_n <- mean(predictions_n != test$credit_risk)
print(paste('Accuracy:', 1-error_n))

# CF Test
X_CF <- data.frame(u=u_test_CF, age=test_CF$age, credit_risk=test_CF$credit_risk)
predictions_raw_CF <- predict(model, newdata=X_CF, type='response')
predictions_CF <- ifelse(predictions_raw_CF > 0.5, 1, 0)
error_CF <- mean(predictions_CF != test_CF$credit_risk)
print(paste('Accuracy:', 1-error_CF))



# Plot
sm.density.compare(predictions_raw, factor(train$sex), xlab="Credit risk probability")
title("Density plot comparison of sex")
legend("topright", legend=levels(factor(train$credit_risk)), fill=2+(0:nlevels(factor(train$credit_risk))))

sm.density.compare(predictions_raw_n, factor(test$sex), xlab="Credit risk probability")
title("Density plot comparison of sex")
legend("topright", legend=levels(factor(test$credit_risk)), fill=2+(0:nlevels(factor(test$credit_risk))))

sm.density.compare(predictions_raw_CF, factor(test_CF$sex), xlab="Credit risk probability")
title("Density plot comparison of sex")
legend("topright", legend=levels(factor(test_CF$credit_risk)), fill=2+(0:nlevels(factor(test_CF$credit_risk))))

d <- density(predictions_raw)
plot(d, main="Density plot of predicted credit risk")
polygon(d, col="grey", border="black") 

