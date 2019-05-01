require(rjags)
require(coda)
library(caret)
library(sm)

# First try without AGE as (protected) attribtue


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


### Training the model

model = jags.model('gcd_model_train.jags',
                   data = list('N' = N_train, 'y' = train$credit_risk, 'a' = train$sex, 
                               'amt' = train$amount, 'dur' = train$duration
#                               'hous1' = train$housing1, 'hous2' = train$housing2, 'hous3' = train$housing3
                               ),
                   n.chains = 4)

samples = coda.samples(model, c('u', 
                                'amt0', 'amt_u', 'amt_a', 'amt_tau', 
                                'dur0', 'dur_u', 'dur_a', 'dur_tau',
                                'y0', 'y_u', 'y_a', 'y_amt', 'y_dur' 
                                ), 
                       n.iter = 2000) # increase iterations if necessary for final model
# save(samples, file='mcmc_samples.Rdata')
mcmcMat = as.matrix(samples , chains=TRUE )
means <- colMeans(mcmcMat)

amt0 <- means["amt0"]
amt_u <- means["amt_u"]
amt_a <- means["amt_a"]
amt_tau <- means["amt_tau"]

dur0 <- means["dur0"]
dur_u <- means["dur_u"]
dur_a <- means["dur_a"]
dur_tau <- means["dur_tau"]

y0 <- means["y0"]
y_u <- means["y_u"]
y_a <- means["y_a"]
y_amt <- means["y_amt"]
y_dur <- means["y_dur"]

u_train <- means[10:(length(means)-5)]


### Learning u for test data using learned parameters

model_test = jags.model('gcd_model_u.jags',
                   data = list('N' = N_test, 'a' = test$sex, 
                               'amt' = test$amount, 'dur' = test$duration,
                               'amt0' = amt0, 'amt_u' = amt_u, 'amt_a' = amt_a, 'amt_tau' = amt_tau,
                               'dur0' = dur0, 'dur_u' = dur_u, 'dur_a' = dur_a, 'dur_tau' = dur_tau
#                               'y0' = y0, 'y_u' = y_u, 'y_a' = y_a, 'y_amt' = y_amt, 'y_dur' = y_dur 
                   ),
                   n.chains = 4)

samples_u = coda.samples(model_test, c('u'), n.iter = 2000) # increase iterations if necessary for final model
# save(samples_u, file='mcmc_samples.Rdata')
mcmcMat_u = as.matrix(samples_u , chains=TRUE )
# u = mcmcMat[,"u"]
u_test <- colMeans(mcmcMat_u)
u_test <- u_test[2:length(u_test)]



# Now it just predicts every record to be a good credit risk because that's the majority (700 -> 0.7 accuracy)
# Predictions never more than 0.5 probability
# Without the categorical variable, the accuracy is much higher 0.935 so probably should look into that
X <- data.frame(u=u_train, credit_risk=train$credit_risk)
X_n <- data.frame(u=u_test, credit_risk=test$credit_risk)

model <- glm(credit_risk ~ u, family=binomial("logit"), data=X)

# Train accuracy
predictions_raw <- predict(model, type='response')
predictions <- ifelse(predictions_raw > 0.5, 1, 0)
error <- mean(predictions != train$credit_risk)
print(paste('Accuracy:', 1-error))

# Test accuracy
predictions_raw <- predict(model, newdata=X_n, type='response')
predictions <- ifelse(predictions_raw > 0.5, 1, 0)
error <- mean(predictions != test$credit_risk)
print(paste('Accuracy:', 1-error))



# Plot
sm.density.compare(predictions_raw, factor(train$sex), xlab="Credit risk probability")
title("Density plot comparison of sex")
legend("topright", legend=levels(factor(train$credit_risk)), fill=2+(0:nlevels(factor(train$credit_risk))))

d <- density(predictions_raw)
plot(d, main="Density plot of predicted credit risk")
polygon(d, col="grey", border="black") 

