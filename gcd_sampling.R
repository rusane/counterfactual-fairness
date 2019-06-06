require(rjags)
require(coda)
library(caret)
library(sm)

load("gcd_data_bin.Rdata")
N <- dim(data)[1]

#load.module("glm")

#cut_hous = c(-0.1, 0.1)
#cut_sav = c(-0.3, -0.1, 0.1, 0.3)
#cut_stat = c(-0.2, 0, 0.2)
#cut_hous = c(-2.260, -1.051)
#cut_sav = c(-1.472, 1.335, 2.115, 3.021)
#cut_stat = c(-0.420, 0.716, 2.720)

model = jags.model('gcd_model_train.jags',
                   data = list('N' = N, 'y' = data$credit_risk, 'a' = data$sex, 
                               'amt' = data$amount, 'dur' = data$duration,
                               'age' = data$age,
                               'hous' = data$housing, 'sav' = data$savings, 'stat' = data$status
                               # 'nhous' = 2, 'nsav' = 4, 'nstat' = 3,
                               # 'cut_hous' = cut_hous, 'cut_sav' = cut_sav, 'cut_stat' = cut_stat
                   ),
                   n.chains = 1,
                   n.adapt = 1000)
update(model, 10000)
samples = coda.samples(model, c('u', 
                                'amt0', 'amt_u', 'amt_a', 'amt_tau', 'amt_c',
                                'dur0', 'dur_u', 'dur_a', 'dur_tau', 'dur_c',
                                'hous0', 'hous_u', 'hous_a', 'hous_c',
                                'sav0', 'sav_u', 'sav_a', 'sav_c',
                                'stat0', 'stat_u', 'stat_a', 'stat_c',
                                'y0', 'y_u', 'y_a', 'y_amt', 'y_dur', 'y_c', 'y_hous', 'y_sav', 'y_stat'
                              ), 
                       n.iter = 10000,
                       thin = 2)

#save(samples, file="samples_bin.Rdata")
params <- c("dur_u", "amt_u", "u[2]")
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

y0 <- means["y0"]
y_u <- means["y_u"]
y_a <- means["y_a"]
y_amt <- means["y_amt"]
y_dur <- means["y_dur"]
y_c <- means["y_c"]
y_hous <- means["y_hous"]
y_sav <- means["y_sav"]
y_stat <- means["y_stat"]

u <- means[23:(length(means)-9)] # change this
#u <- means[12:(length(means)-6)]


### Sampling with observed sex (original)

model_sampling = jags.model('gcd_model_u.jags',
                        data = list('N' = N, 'a' = data$sex, 'age' = data$age,
                                    #'amt' = data$amount, 'dur' = data$duration,
                                    #'hous' = data$housing, 'sav' = data$savings, 'stat' = data$status,
                                    'amt0' = amt0, 'amt_u' = amt_u, 'amt_a' = amt_a, 'amt_tau' = amt_tau, 'amt_c' = amt_c,
                                    'dur0' = dur0, 'dur_u' = dur_u, 'dur_a' = dur_a, 'dur_tau' = dur_tau, 'dur_c' = dur_c,
                                    'hous0' = hous0, 'hous_u' = hous_u, 'hous_a' = hous_a, 'hous_c' = hous_c,
                                    'sav0' = sav0, 'sav_u'= sav_u, 'sav_a' = sav_a, 'sav_c' = sav_c,
                                    'stat0' = stat0, 'stat_u' = stat_u, 'stat_a' = stat_a, 'stat_c' = stat_c,
                                    'y0' = y0, 'y_u' = y_u, 'y_a' = y_a, 'y_amt' = y_amt, 'y_dur' = y_dur, 'y_c' = y_c, 'y_hous' = y_hous, 'y_sav' = y_sav, 'y_stat' = y_stat
                                    # 'nhous' = 2, 'nsav' = 4, 'nstat' = 3,
                                    # 'cut_hous' = cut_hous, 'cut_sav' = cut_sav, 'cut_stat' = cut_stat
                        ),
                        n.chains = 1,
                        n.adapt = 1000)
update(model_sampling, 10000)
data_attr = c('y', 'u', 'amt', 'dur', 'hous', 'sav', 'stat')
sample_data = coda.samples(model_sampling, data_attr, n.iter = 10000)

params <- c("u[2]")
plot(sample_data[,params])
#gelman.diag(sample_data[,params])
#gelman.plot(sample_data[,params])

sample_mcmcMat = as.matrix(sample_data, chains=FALSE )
sample_means <- colMeans(sample_mcmcMat)

#u_og <- sample_means[1:1000]
#credit_risk_og <- sample_means[1001:2000]
amount <- sample_means[1:1000]
duration <- sample_means[1001:2000]
housing <- round(sample_means[2001:3000])
savings <- round(sample_means[3001:4000])
status <- round(sample_means[4001:5000])
u_og <- round(sample_means[5001:6000])
credit_risk <- round(sample_means[6001:7000])
#credit_risk <- sample_means[2001:3000]

#data_og <- data.frame("sex" = data$sex, "age" = data$age, "amount" = data$amount, "duration" = data$duration, "housing" = data$housing, "savings" = data$savings, "status" = data$status, "credit_risk" = round(credit_risk_og), "u" = u_og)
data_og <- data.frame("sex" = data$sex, "age" = data$age, amount, duration, housing, savings, status, "credit_risk" = data$credit_risk, "u" = u_og)
rownames(data_og) = NULL
save(data_og, file="data_samples_og.Rdata")


### Sampling with counterfactual sex (counterfactual)

sex_cf <- data$sex
male_idx <- which(data$sex %in% '0')
sex_cf[male_idx] <- 1
sex_cf[-male_idx] <- 0

model_sampling_cf = jags.model('gcd_model_u.jags',
                            data = list('N' = N, 'u' = u, 'a' = sex_cf, 'age' = data$age,
                                        #'amt' = data$amount, 'dur' = data$duration,
                                        #'hous' = data$housing, 'sav' = data$savings, 'stat' = data$status,
                                        'amt0' = amt0, 'amt_u' = amt_u, 'amt_a' = amt_a, 'amt_tau' = amt_tau, 'amt_c' = amt_c,
                                        'dur0' = dur0, 'dur_u' = dur_u, 'dur_a' = dur_a, 'dur_tau' = dur_tau, 'dur_c' = dur_c,
                                        'hous0' = hous0, 'hous_u' = hous_u, 'hous_a' = hous_a, 'hous_c' = hous_c,
                                        'sav0' = sav0, 'sav_u'= sav_u, 'sav_a' = sav_a, 'sav_c' = sav_c,
                                        'stat0' = stat0, 'stat_u' = stat_u, 'stat_a' = stat_a, 'stat_c' = stat_c,
                                        'y0' = y0, 'y_u' = y_u, 'y_a' = y_a, 'y_amt' = y_amt, 'y_dur' = y_dur, 'y_c' = y_c, 'y_hous' = y_hous, 'y_sav' = y_sav, 'y_stat' = y_stat
                                        #'nhous' = 2, 'nsav' = 4, 'nstat' = 3,
                                        #'cut_hous' = cut_hous, 'cut_sav' = cut_sav, 'cut_stat' = cut_stat
                            ),
                            n.chains = 1,
                            n.adapt = 1000)
update(model_sampling_cf, 10000)
data_attr_cf = c('y', 'amt', 'dur', 'hous', 'sav', 'stat')
sample_data_cf = coda.samples(model_sampling_cf, data_attr_cf, n.iter = 10000)

sample_mcmcMat_cf = as.matrix(sample_data_cf, chains=FALSE )
sample_means_cf <- colMeans(sample_mcmcMat_cf)

#u_cf <- sample_means_cf[1:1000]
#credit_risk_cf <- sample_means_cf[1001:2000]
amount <- sample_means_cf[1:1000]
duration <- sample_means_cf[1001:2000]
housing <- round(sample_means_cf[2001:3000])
savings <- round(sample_means_cf[3001:4000])
status <- round(sample_means_cf[4001:5000])
credit_risk <- round(sample_means_cf[5001:6000])

# <- data.frame("sex" = sex_cf, "age" = data$age, "amount" = data$amount, "duration" = data$duration, "housing" = data$housing, "savings" = data$savings, "status" = data$status, "credit_risk" = round(credit_risk_cf), "u" = u_cf)
data_cf <- data.frame("sex" = sex_cf, "age" = data$age, amount, duration, housing, savings, status, "credit_risk" = credit_risk, "u" = u)
rownames(data_cf) = NULL
save(data_cf, file="data_samples_cf.Rdata")


