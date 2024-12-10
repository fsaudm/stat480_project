# stat480_project



## Initial Data Splitting

[5-fold split](/data) performed using this [script](split.ipynb).



## Exploratory Data Analysis (EDA)

View the notebook here: [View EDA Notebook](eda.ipynb).



## LASSO Regression Models

- For Binary response `CRASH_TYPE`: [Binomial LASSO Regression](binomial_logreg.py)
- For multi-class response `MOST_SEVERE_INJURY` : [Multinomial LASSO Regression](multinomial_logreg.py)


## Results 

Summarized in this [Analysis Notebook](results_analisys.ipynb) and the estimated coefficients can be found in this [directory](/output).


## Double Lasso

R Scripts for double lasso to test causality:

- Implementation details can be found [here](binomial_double_lasso.r). This looks at how to run double lasso for 2 non-zero variables (`PRIM_CONTRIBUTORY_CAUSEUNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)`, `ROADWAY_SURFACE_CONDSNOW OR SLUSH`)
- This [script](parallel_binomial_double_lasso.r) looks at how to run double lasso for all non-zero variables in parallel using future and future.apply.





