library(tidyverse)
library(data.table)
library(gamlr)

set.seed(480)


### Load Data
df1 <- fread("data/clean_1.csv")
df2 <- fread("data/clean_2.csv")
df3 <- fread("data/clean_3.csv")
df4 <- fread("data/clean_4.csv")
df5 <- fread("data/clean_5.csv")


df <- rbind(df1, df2, df3, df4, df5)
rm(df1, df2, df3, df4, df5)


df <- df %>% drop_na(MOST_SEVERE_INJURY) %>% as.data.frame()



### Variables
# Factor variables
factor_cols <- c(
  'CRASH_DATE_EST_I',
  #'POSTED_SPEED_LIMIT', 
  'TRAFFIC_CONTROL_DEVICE', 
  'DEVICE_CONDITION', 
  'WEATHER_CONDITION', 
  'LIGHTING_CONDITION', 
  'FIRST_CRASH_TYPE', 
  'TRAFFICWAY_TYPE', 
  'ALIGNMENT', 
  'ROADWAY_SURFACE_COND', 
  'ROAD_DEFECT', 
  'REPORT_TYPE', 
  'CRASH_TYPE', 
  'INTERSECTION_RELATED_I',
  'NOT_RIGHT_OF_WAY_I',
  'HIT_AND_RUN_I',
  'DAMAGE', ### attention to this one! potential response var??
  'PRIM_CONTRIBUTORY_CAUSE', 
  'SEC_CONTRIBUTORY_CAUSE',  
  'DOORING_I',
  'WORK_ZONE_I',
  'WORK_ZONE_TYPE', ### > 99% are N/A
  'WORKERS_PRESENT_I',
  'MOST_SEVERE_INJURY', 
  'CRASH_HOUR', 
  'CRASH_DAY_OF_WEEK', 
  'CRASH_MONTH',
  'Crash_Year', # Recently Included #
  'Police_district', # Recently Inc, using instead of 'BEAT_OF_OCCURRENCE'
  'LANE_CNT' # Recently Included #
)
df[factor_cols] <- lapply(df[factor_cols], as.factor)


not_useful <- c(
  'CRASH_RECORD_ID', 
  'BEAT_OF_OCCURRENCE',
  'STREET_NO', 
  'STREET_DIRECTION', 
  'STREET_NAME',
  'LATITUDE', 
  'LONGITUDE'
)


date_vars <- c('CRASH_DATE', 'DATE_POLICE_NOTIFIED') 


responses <- c(
  'CRASH_TYPE', #1
  'MOST_SEVERE_INJURY', #2
  'INJURIES_TOTAL', #3
  'INJURIES_FATAL', 
  'INJURIES_INCAPACITATING', 
  'INJURIES_NON_INCAPACITATING', 
  'INJURIES_REPORTED_NOT_EVIDENT', 
  'INJURIES_NO_INDICATION'
  #'DAMAGE'
)






### Design Matrix
response <- "CRASH_TYPE"


predictors <- df %>%
  select(-all_of(c(responses, not_useful, date_vars))) %>%
  colnames()


formula <- as.formula(paste(response, "~", paste(predictors, collapse = " + ")))


X <- model.matrix(formula, data = df)
y <- df[[response]]


# Treatment variable
d <- X[, "PRIM_CONTRIBUTORY_CAUSEUNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)"]
XX <- X[, colnames(X) != "PRIM_CONTRIBUTORY_CAUSEUNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)"]






#### Double Lasso

## 1st Lasso
lasso1 <- gamlr(x = XX, y = d, family = "binomial", standardize=TRUE)
B1 <- coef(lasso1)

min.AICc.lambda <- lasso1$lambda[ which.min( AICc(lasso1) ) ] 
paste("Min AICc lambda: ", min.AICc.lambda)

# Predicted causal variable
d_hat <- predict(lasso1, XX, type = "response") 

# R squared
r2 <- cor(drop(d_hat),d)^2
paste("In-sample R^2: ", r2) # 0.928505221949197




## 2nd Lasso
lasso2 <- gamlr(x = cbind(d, d_hat, XX), y = y , free=2, family = "binomial", standardize=TRUE)
B2 <- coef(lasso2)

min.AICc.lambda <- lasso2$lambda[ which.min( AICc(lasso2) ) ] 
paste("Min AICc lambda: ", min.AICc.lambda)

# Treatment effect after controlling for confounders
treatment_effect <- B2[2] # 0
treatment_effect




## naive lasso
naive <- gamlr(x = cbind(d, XX), y = y , family = "binomial", standardize=TRUE)
B <- coef(naive)

min.AICc.lambda <- naive$lambda[ which.min( AICc(naive) ) ] 
paste("Min AICc lambda: ", min.AICc.lambda)

# Treatment effect
naive_effect <- B[2] # -1.107208
naive_effect












































### Design Matrix
response <- "CRASH_TYPE"


predictors <- df %>%
  select(-all_of(c(responses, not_useful, date_vars))) %>%
  colnames()


formula <- as.formula(paste(response, "~", paste(predictors, collapse = " + ")))


X <- model.matrix(formula, data = df)
y <- df[[response]]


# Treatment variable
d <- X[, "ROADWAY_SURFACE_CONDSNOW OR SLUSH"]
XX <- X[, colnames(X) != "ROADWAY_SURFACE_CONDSNOW OR SLUSH"]






#### Double Lasso

## 1st Lasso
lasso3 <- gamlr(x = XX, y = d, family = "binomial", standardize=TRUE)
B3 <- coef(lasso3)

min.AICc.lambda <- lasso3$lambda[ which.min( AICc(lasso3) ) ] 
paste("Min AICc lambda: ", min.AICc.lambda)

# Predicted causal variable
d_hat <- predict(lasso3, XX, type = "response") 

# R squared
r2 <- cor(drop(d_hat),d)^2
paste("In-sample R^2: ", r2) # 0.700471566598306




## 2nd Lasso
lasso4 <- gamlr(x = cbind(d, d_hat, XX), y = y , free=2, family = "binomial", standardize=TRUE)
B4 <- coef(lasso4)

min.AICc.lambda <- lasso4$lambda[ which.min( AICc(lasso4) ) ] 
paste("Min AICc lambda: ", min.AICc.lambda)

# Treatment effect after controlling for confounders
treatment_effect <- B4[2]
treatment_effect # 0.




## naive lasso
naive <- gamlr(x = cbind(d, XX), y = y , family = "binomial", standardize=TRUE)
BB <- coef(naive)

min.AICc.lambda <- naive$lambda[ which.min( AICc(naive) ) ] 
paste("Min AICc lambda: ", min.AICc.lambda)

# Treatment effect
naive_effect <- BB[2]
naive_effect # 0.09930429