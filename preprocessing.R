library(sm)
library(MASS)

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

A91_idx <- which(data$personal_status_sex %in% 'A91')
A92_idx <- which(data$personal_status_sex %in% 'A92')
A93_idx <- which(data$personal_status_sex %in% 'A93')
A94_idx <- which(data$personal_status_sex %in% 'A94')
male_idx <- c(A91_idx, A93_idx, A94_idx)

data$sex <- NA
data$sex[male_idx] <- 'M'
data$sex[-male_idx] <- 'F'
data$sex <- factor(data$sex)

# Function that splits data into balanced train and test set
splitData <- function(data, x) {
  good_credit_risk <- which(data$credit_risk %in% 1)
  bad_credit_risk <- which(data$credit_risk %in% 2)
  train_good_idx <- sample(good_credit_risk, x*length(good_credit_risk))
  train_bad_idx <- sample(bad_credit_risk, x*length(bad_credit_risk))
  train_idx <- sort(c(train_good_idx, train_bad_idx))
  train <- data[train_idx,]
  test <- data[-train_idx,]
  split <- list(data[train_idx,], data[-train_idx,])
  names(split) <- c("train", "test")
  return(split)
}

# Create balanced train and test set (maybe use k-fold cross validation, which requires the caret library)
train <- splitData(data, 0.8)$train
test <- splitData(data, 0.8)$test


# Create counterfactual data set (do not do it like this, use casual model to generate data and counterfactual data)
train_CF <- train
male_train_idx <- which(train_CF$sex %in% 'M')
train_CF$sex[male_train_idx] <- 'F'
train_CF$sex[-male_train_idx] <- 'M'
test_CF <- test
male_test_idx <- which(test_CF$sex %in% 'M')
test_CF$sex[male_test_idx] <- 'F'
test_CF$sex[-male_test_idx] <- 'M'

train <- train_CF
test <- test_CF


# Model from paper Path-specific counterfactual fairness
model <- glm(credit_risk ~ status + savings + housing + amount + duration + age + sex, family=binomial("logit"), data=train)
summary(model)

model <- glm(credit_risk ~ status + sex + age + foreign_worker + purpose, family=binomial("logit"), data=train)
summary(model)


# Model selection to find best fit (not the objective of this study...)
model <- glm(credit_risk ~ . - personal_status_sex, family=binomial("logit"), data=train)
summary(model)

model.step <- stepAIC(model, direction="backward", trace = FALSE)
model.step$anova

model <- model.step
summary(model)

# mdrop1(model, test='Chisq')
# model <- glm(credit_risk ~ . - job - housing - personal_status_sex, family=binomial("logit"), data=train)
# summary(model)


# Prediction
predictions_raw <- predict(model, newdata=test, type='response')

sm.density.compare(predictions_raw, test$sex, xlab="Credit risk probability")
title("Density plot comparison of sex")
legend("topright", levels(test$sex), fill=2+(0:nlevels(test$sex)))

predictions <- ifelse(predictions_raw > 0.5, 2, 1)
error <- mean(predictions != test$credit_risk)
print(paste('Accuracy:', 1-error))


# Miscellaneous code
d <- density(predictions_raw[male_idx], na.rm=TRUE)
plot(d, main="Density plot of predicted credit risk")
polygon(d, col="grey", border="black") 

d2 <- density(predictions_raw[-male_idx], na.rm=TRUE)
lines(d2)
polygon(d2, col="white", border="black") 

