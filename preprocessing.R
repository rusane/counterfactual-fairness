# Read file
data <- read.table("german.data.txt", sep="", header=FALSE, comment.char="")
# data_numeric <- read.table("german.data-numeric.txt", sep="", header=FALSE, comment.char="")

# https://www.rdocumentation.org/packages/klaR/versions/0.6-14/topics/GermanCredit
names(data) <- c("status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration", "installment_rate", "personal_status_sex", "other_debtors", "present_residence", "property", "age", "other_installment_plans", "housing", "number_credits", "job", "people_liable", "telephone", "foreign_worker", "credit_risk")

# Check if there are any missing values
sum(is.na(data))
# sapply(data, function(x) sum(is.na(x)))

data$credit_risk = factor(data$credit_risk) # 1=good, 2=bad


# TO-DO: personal_status_sex -> seperate sex and marital status


# Create balanced train and test set
# nrOfRows <- dim(data)[1]
# nrOfCols <- dim(data)[2]

good_credit_risk <- which(data$credit_risk %in% 1)
bad_credit_risk <- which(data$credit_risk %in% 2)

train_good_id <- sample(good_credit_risk, 0.8*length(good_credit_risk))
train_bad_id <- sample(bad_credit_risk, 0.8*length(bad_credit_risk))
train_id <- sort(c(train_good_id, train_bad_id))

train <- data[train_id,]
test <- data[-train_id,]


# Model selection
model <- glm(credit_risk ~ ., family=binomial("logit"), data=train)
summary(model)

drop1(model, test='Chisq')
model <- glm(credit_risk ~ . - job - housing, family=binomial("logit"), data=train)
# model <- glm(credit_risk ~ . - property - present_residence, family=binomial("logit"), data=train)
summary(model)


# Prediction
predictions <- predict(model, newdata=test, type='response')
predictions <- ifelse(predictions > 0.5, 2, 1)

error <- mean(predictions != test$credit_risk)
print(paste('Accuracy:', 1-error))
