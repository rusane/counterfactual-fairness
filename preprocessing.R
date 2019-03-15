library(sm)

# Read file
data <- read.table("german.data.txt", sep="", header=FALSE, comment.char="")
# data_numeric <- read.table("german.data-numeric.txt", sep="", header=FALSE, comment.char="")

# Names adapted from: https://www.rdocumentation.org/packages/klaR/versions/0.6-14/topics/GermanCredit
names(data) <- c("status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration", "installment_rate", "personal_status_sex", "other_debtors", "present_residence", "property", "age", "other_installment_plans", "housing", "number_credits", "job", "people_liable", "telephone", "foreign_worker", "credit_risk")

# Check if there are any missing values
sum(is.na(data))
# sapply(data, function(x) sum(is.na(x)))

data$credit_risk = factor(data$credit_risk) # 1=good, 2=bad

# Seperate marital status from sex in personal_status_sex attribute
# A91 : male   : divorced/separated
# A92 : female : divorced/separated/married
# A93 : male   : single
# A94 : male   : married/widowed
# A95 : female : single
levels(data$personal_status_sex) # 4 levels A91, A92, A93, A94 (i.e. no A95 = single females)

A91_id <- which(data$personal_status_sex %in% 'A91')
A92_id <- which(data$personal_status_sex %in% 'A92')
A93_id <- which(data$personal_status_sex %in% 'A93')
A94_id <- which(data$personal_status_sex %in% 'A94')
male_id <- c(A91_id, A93_id, A94_id)

data$sex <- NA
data$sex[male_id] <- 'M'
data$sex[-male_id] <- 'F'



# Create counterfactual data set
data$sex <- NA
data$sex[male_id] <- 'F'
data$sex[-male_id] <- 'M'



# Create balanced train and test set (maybe use k-fold cross validation, which requires the caret library)
good_credit_risk <- which(data$credit_risk %in% 1)
bad_credit_risk <- which(data$credit_risk %in% 2)

train_good_id <- sample(good_credit_risk, 0.8*length(good_credit_risk))
train_bad_id <- sample(bad_credit_risk, 0.8*length(bad_credit_risk))
train_id <- sort(c(train_good_id, train_bad_id))

train <- data[train_id,]
test <- data[-train_id,]



# Model from paper Path-specific counterfactual fairness
model <- glm(credit_risk ~ status + savings + housing + amount + duration + age + sex, family=binomial("logit"), data=train)
summary(model)



# Model selection
# model <- glm(credit_risk ~ . - personal_status_sex, family=binomial("logit"), data=train)
# summary(model)
# 
# drop1(model, test='Chisq')
# model <- glm(credit_risk ~ . - job - housing - personal_status_sex, family=binomial("logit"), data=train)
# summary(model)



# Prediction
predictions_raw <- predict(model, newdata=test, type='response')
predictions <- ifelse(predictions_raw > 0.5, 2, 1)

error <- mean(predictions != test$credit_risk)
print(paste('Accuracy:', 1-error))
