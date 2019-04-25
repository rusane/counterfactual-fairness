# Load data
data <- read.table("german.data.txt", sep="", header=FALSE, comment.char="")

# Names adapted from: https://www.rdocumentation.org/packages/klaR/versions/0.6-14/topics/GermanCredit
names(data) <- c("status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration", "installment_rate", "personal_status_sex", "other_debtors", "present_residence", "property", "age", "other_installment_plans", "housing", "number_credits", "job", "people_liable", "telephone", "foreign_worker", "credit_risk")

# Binary outcome
data$credit_risk[data$credit_risk==1] <- 0 # good
data$credit_risk[data$credit_risk==2] <- 1 # bad

# Extract sex from compound attribute
A91_idx <- which(data$personal_status_sex %in% 'A91')
A92_idx <- which(data$personal_status_sex %in% 'A92')
A93_idx <- which(data$personal_status_sex %in% 'A93')
A94_idx <- which(data$personal_status_sex %in% 'A94')
male_idx <- c(A91_idx, A93_idx, A94_idx)

data$sex <- NA
data$sex[male_idx] <- '0' # male
data$sex[-male_idx] <- '1' # female

# Relevant attributes for the casual model
vars <- c("status", "duration", "amount", "savings", "age", "housing", "sex", "credit_risk")
data <- data[, vars]

# One hot encoding of all remaining categorical variables
levels(data$status) # 4 levels
data$status1 <- as.numeric(data$status == "A11")
data$status2 <- as.numeric(data$status == "A12")
data$status3 <- as.numeric(data$status == "A13")
data$status4 <- as.numeric(data$status == "A14")

levels(data$savings) # 5 levels
data$savings1 <- as.numeric(data$savings == "A61")
data$savings2 <- as.numeric(data$savings == "A62")
data$savings3 <- as.numeric(data$savings == "A63")
data$savings4 <- as.numeric(data$savings == "A64")
data$savings5 <- as.numeric(data$savings == "A65")

levels(data$housing) # 3 levels
data$housing1 <- as.numeric(data$housing == "A151")
data$housing2 <- as.numeric(data$housing == "A152")
data$housing3 <- as.numeric(data$housing == "A153")

# Remove old columns of categorical attributes
data <- within(data, rm("status","savings","housing"))
data <- data[, !(colnames(data) %in% c("status","savings","housing"))]

# Save preprocessed data
save(data, file='gcd_data.Rdata')




