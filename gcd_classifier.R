require(rjags)
require(coda)
library(caret)
library(doBy)

load("gcd_data_bin.Rdata")
N <- dim(data)[1]

set.seed(0)
trainIndex <- createDataPartition(data$credit_risk, p = .8, list = FALSE, times = 1)
train <- data[trainIndex,]
test <- data[-trainIndex,]
N_train <- dim(train)[1]
N_test <- dim(test)[1]
male_test_idx <- which(test$sex %in% '0')

### Training the causal model
model = jags.model('jags/gcd_model_train.jags',
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

#params <- c("u[1]", "amt_u", "hous_u")
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
model_test = jags.model('jags/gcd_model_u.jags',
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


### Classifier
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
