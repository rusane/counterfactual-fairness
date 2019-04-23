require(rjags)
require(coda)

# First try without AGE as a protected attribtue

data <- read.table("german.data.txt", sep="", header=FALSE, comment.char="")
names(data) <- c("status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration", "installment_rate", "personal_status_sex", "other_debtors", "present_residence", "property", "age", "other_installment_plans", "housing", "number_credits", "job", "people_liable", "telephone", "foreign_worker", "credit_risk")

N <- dim(data)[1]

data$credit_risk[data$credit_risk==1] <- 0 # good
data$credit_risk[data$credit_risk==2] <- 1 # bad

A91_idx <- which(data$personal_status_sex %in% 'A91')
A92_idx <- which(data$personal_status_sex %in% 'A92')
A93_idx <- which(data$personal_status_sex %in% 'A93')
A94_idx <- which(data$personal_status_sex %in% 'A94')
male_idx <- c(A91_idx, A93_idx, A94_idx)

data$sex <- NA
data$sex[male_idx] <- 'M' # male
data$sex[-male_idx] <- 'F' # female
data$sex <- factor(data$sex)





### Training the model

model = jags.model('gcd_model_train.jags',
                   data = list('N' = N, 'y' = data$credit_risk, 'a'=data$sex),
                   n.chains = 4)

samples = coda.samples(model, c('u'), n.iter = 2000)
mcmcMat = as.matrix(samples , chains=TRUE )
# u = mcmcMat[,"u"]
colMeans(mcmcMat)

# jags.samples(model, c('u'), 1000)
