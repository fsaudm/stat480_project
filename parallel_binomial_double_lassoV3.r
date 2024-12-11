# Same as v2, but smaller lambda.min.ratio for naive lasso, extend lookup range.

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
  'LONGITUDE',
  'DAMAGE',
  'REPORT_TYPE'
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
)






### Design Matrix
response <- "CRASH_TYPE"


predictors <- df %>%
  select(-all_of(c(responses, not_useful, date_vars))) %>%
  colnames()


formula <- as.formula(paste(response, "~", paste(predictors, collapse = " + ")))


X <- model.matrix(formula, data = df)
y <- df[[response]]

print(object.size(X), units = "auto") # 1.8 Gb
print(object.size(y), units = "auto") # 3.4 Mb



## ---- "Naive" Lasso -------------------------------------------------------- ##
"
Keeping lambda.min.ratio as default, model at seg100. Exploring further seems to not like regularization...
"

naive <- gamlr(X, y = y , family = "binomial", standardize=TRUE, lambda.min.ratio = 1e-4) 
saveRDS(naive, file = "R_output/naive_binomial_lassoV3.rds") # save the model
naive <- readRDS("R_output/naive_binomial_lassoV3.rds") # load the model
naive.B <- coef(naive)

min.AICc.lambda <- naive$lambda[ which.min( AICc(naive) ) ] 
paste("Min AICc lambda: ", min.AICc.lambda)

# # Path plot
png("R_output/naive_binomial_lasso_pathV3.png", width = 800, height = 600) # save the
plot(naive, against = "pen")
dev.off()


# Non-zero coefficients
non.zero.vars <- rownames(naive.B)[which(naive.B != 0)][-1] # -1 to exclude intercept, 127 non-zero variables

# Exclue problematic variables
non.zero.vars <- setdiff(
  rownames(naive.B)[which(naive.B != 0)][-1], # Exclude intercept
  c(
  "CRASH_DATE_EST_IY", 
  "PRIM_CONTRIBUTORY_CAUSEDISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)", 
  "PRIM_CONTRIBUTORY_CAUSERELATED TO BUS STOP"
  )
)








## ---- Double Lasso -------------------------------------------------------- ##
# "
# Double Lasso for non-zero variables. 
# Parallelized using future for shared memory, 64 cores.
# "

library(future)
library(future.apply)
library(gamlr)

# Future parallel backend
options(future.globals.maxSize = 4 * 1024^3)
plan(multicore)

parallel.time <- system.time({
  results <- future_sapply(non.zero.vars, function(var_name) {
    gc()
    tryCatch({
      # Step 1: Treatment variable
      d <- X[, var_name]
      XX <- X[, colnames(X) != var_name]
      
      # Step 2: Family type
      family.type <- if (all(d %in% c(0, 1))) {
        "binomial"
      } else {
        "gaussian"
      }
      
      # Step 3: First Lasso
      lasso1 <- gamlr(x = XX, y = d, family = family.type, standardize = TRUE)
      d_hat <- predict(lasso1, XX, type = "response")
      r2 <- cor(drop(d_hat), d)^2
      
      # Step 4: Second Lasso
      lasso2 <- gamlr(x = cbind(d, d_hat, XX), y = y, free = 2, family = "binomial", standardize = TRUE)
      B2 <- coef(lasso2)
      
      # results
      list(
        variable = var_name,
        lasso2_coefs = B2[2],
        r2 = r2
      )
    }, error = function(e) {
      # Handle errors "gracefully"
      list(variable = var_name, error = e$message)
    })
  }, simplify = FALSE)
})

print(parallel.time)
saveRDS(results, file = "R_output/parallel_lasso_resultsV3.rds")
results <- readRDS("R_output/parallel_lasso_resultsV3.rds") # load the model



results.df <- do.call(rbind, lapply(results, function(res) {
  naive_coef <- naive.B[res$variable, 1]  #
  data.frame(
    variable = res$variable,
    naive_coefs = naive_coef,
    lasso2_coefs = res$lasso2_coefs,
    r2 = res$r2,
    stringsAsFactors = FALSE
  )
}))


# Save results :)
write.csv(results.df, file = "R_output/parallel_lasso_resultsV3.csv", row.names = FALSE) #false bc we have variable names


#read the results
# results.df <- read.csv("R_output/lasso_results_with_naive.csv")
