import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import patsy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

import os
import glob

pd.options.display.float_format = '{:.4f}'.format

# Seed
np.random.seed(480)


# Load Clean Data
file_list = glob.glob("data/clean_*.csv") 

df = pd.concat([pd.read_csv(file) for file in file_list], # list comprehension, supposedly faster than for loop
               ignore_index=True
               )

df.isna().sum()

# drop rows with missing values in MOST_SEVERE_INJURY
df = df.dropna(subset=['MOST_SEVERE_INJURY'])



## Variables
factor_cols = [
    'CRASH_DATE_EST_I',
    #'POSTED_SPEED_LIMIT', 
    'TRAFFIC_CONTROL_DEVICE', 
    'DEVICE_CONDITION', 
    'WEATHER_CONDITION', 
    'LIGHTING_CONDITION', 
    'FIRST_CRASH_TYPE', ##### attention to this one! what to do?? keep in x?
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
    'Crash_Year', # Recently Incuded #
    'Police_district', # Recently Inc, using instead of 'BEAT_OF_OCCURRENCE'
    'LANE_CNT', # Recently Incuded #
]

# Convert to categorical
for col in factor_cols:
    df[col] = df[col].astype('category')


# 5 variables not useful for predictive analysis
not_usefull = ['CRASH_RECORD_ID', 
               'BEAT_OF_OCCURRENCE',
               'STREET_NO', 
               'STREET_DIRECTION', 
               'STREET_NAME',
               'LATITUDE', 
               'LONGITUDE',
               ]


# 2 Date variables
date_vars = ['CRASH_DATE', 
             'DATE_POLICE_NOTIFIED',
             ] # using Report_vs_Police_Notified: hours between crash and notification


# 8 potential response variables
responses = ['CRASH_TYPE', #1
             'MOST_SEVERE_INJURY', #2
             'INJURIES_TOTAL', #3
             'INJURIES_FATAL', 'INJURIES_INCAPACITATING', 'INJURIES_NON_INCAPACITATING', 'INJURIES_REPORTED_NOT_EVIDENT', 'INJURIES_NO_INDICATION',
             #'DAMAGE',
             ]







# Multiclass response variable
# 5 classes
response = 'MOST_SEVERE_INJURY' 

predictors = df.drop(columns=responses+not_usefull+date_vars).columns

formula = f"{response} ~ {' + '.join(predictors)}"

_, X = patsy.dmatrices(formula, data = df, return_type='dataframe')
y = df[response]







#### Multinomial Logistic Regression
# Number of CPU cores available
num_cores = os.cpu_count()  # or multiprocessing.cpu_count()
print("Total CPU cores available:", num_cores)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_logistic = LogisticRegressionCV(
    penalty='l1',
    solver='saga',  # SAGA supports L1 regularization, faster for large datasets & Multi-class/Multinomial
    cv=5, 
    max_iter=1000,
    tol=1e-3,
    random_state=480,
    Cs=10,  # Number of lambda values,
    n_jobs=-1,  # Use all processors: -1
)

lasso_logistic.fit(X_scaled, y)


print("Best Lambda:", 1 / lasso_logistic.C_[0]) 
print("Coef. shape: ", lasso_logistic.coef_.shape) 


coef_df = pd.DataFrame(
    lasso_logistic.coef_.T,
    index = X.columns, 
    columns = lasso_logistic.classes_  # Class labels as column names
)

# Save Coefficients
coef_df.to_csv("output/lasso_multinomial.csv")










# THRESHOLD
threshold = 5e-2
coef_df[coef_df.abs()>threshold].dropna()


# Plot
n_classes = coef_df.shape[1]  

fig, axes = plt.subplots(3, 2, figsize=(15, 10))
axes = axes.flatten()  

## Plot for each class
for i, class_label in enumerate(coef_df.columns):
    ax = axes[i]
    coef_series = coef_df[class_label]
    
    # coef Bar plot
    coef_series.plot.bar(ax=ax, color='b', alpha=0.7, 
                         title=f"Class: {class_label}")
    
    # threshold
    ax.axhline(threshold, color='r', linestyle='--', linewidth=0.5)
    ax.axhline(-threshold, color='r', linestyle='--', linewidth=0.5)
    
    ax.set_xticks([])
    ax.set_ylabel("Coefficient")

# Hide empty subplot (6th one in a 3x2 grid)
if n_classes < len(axes):
    for j in range(n_classes, len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Save Plot
plt.savefig("output/lasso_multinomial.png")