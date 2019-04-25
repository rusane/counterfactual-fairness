require(rjags)
require(coda)
library(sm)

# First try without AGE as (protected) attribtue

N <- dim(data)[1]
load("gcd_data.Rdata")

### Training the model

model = jags.model('gcd_model_train.jags',
                   data = list('N' = N, 'y' = data$credit_risk, 'a' = data$sex, 
                               'amt' = data$amount, 'dur' = data$duration, 
                               'hous1' = data$housing1, 'hous2' = data$housing2, 'hous3' = data$housing3
                               ),
                   n.chains = 4)

samples = coda.samples(model, c('u'), n.iter = 2000) # increase iterations if necessary for final model
save(samples, file='mcmc_samples.Rdata')
mcmcMat = as.matrix(samples , chains=TRUE )
# u = mcmcMat[,"u"]
u_train <- colMeans(mcmcMat)

# jags.samples(model, c('u'), 1000)


# Now it just predicts every record to be a good credit risk because that's the majority (700 -> 0.7 accuracy)
# Predictions never more than 0.5 probability
X <- data.frame(u=u_train[2:1001], credit_risk=data$credit_risk)
model <- glm(credit_risk ~ u, family=binomial("logit"), data=X)

predictions_raw <- predict(model, newdata=X, type='response')
predictions <- ifelse(predictions_raw > 0.5, 1, 0)
error <- mean(predictions != data$credit_risk)
print(paste('Accuracy:', 1-error))

# Plot
sm.density.compare(predictions_raw, factor(data$sex), xlab="Credit risk probability")
title("Density plot comparison of sex")
legend("topright", legend=levels(factor(data$credit_risk)), fill=2+(0:nlevels(factor(data$credit_risk))))

d <- density(predictions_raw)
plot(d, main="Density plot of predicted credit risk")
polygon(d, col="grey", border="black") 

