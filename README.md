# Machine Learning Model for Grade Prediction
> Create and compare to select best machine learning model to predict final student grades.  


## Table of Contents
* [Overview](#overview)
* [Business Problem](#business-problem)
* [Imports](#imports)
* [Get the Data](#get-the-data)
* [Data Exploration](#data-exploration)
* [Exploratory Analyses](#exploratory-analyses)
* [Descriptive statistics](#descriptive-statistics)
* [Models](#models)
* [Conclusion](#conclusion)



## Overview
The purpose of this project is to a create machine learning model to accurately predict students' final performances to use for our advising team.  We will look at various attributes that may contribute to students' performances. 


## Business Problem
Based on the available data, we need to be able to identify students who are struggling and will need additional educational support or intervention to improve their grades in the course with the use of machine learning.

## Imports
- import pandas as pd

- import numpy as np

- import matplotlib as mpl

- import matplotlib.pyplot as plt

- import seaborn as sns

- from sklearn.model_selection import train_test_split

- from sklearn.base import BaseEstimator, TransformerMixin

- from sklearn.pipeline import make_pipeline

- from sklearn.impute import SimpleImputer

- from sklearn.preprocessing import StandardScaler

- from sklearn.compose import ColumnTransformer

- from sklearn.linear_model import LinearRegression

- from sklearn.model_selection import cross_val_score

- from sklearn.tree import DecisionTreeRegressor

- from sklearn.ensemble import RandomForestRegressor

- from sklearn.model_selection import GridSearchCV

- from sklearn.metrics import mean_squared_error


## Get the Data
student-mat.csv

Label/Target Identification

The first thing we will do is to define our labels (features) and target from the data set. Since we are trying to predict the grade of the students for the final period, the G3 feature will be defined as the target while the rest of the other features will be the label features. To do so, we will drop the G3 column from the original data set and rename the dropped columns as a Series called G3_target. The rest of the dataset will be named student_features.

Creating the training and testing data

Before we go any further in examing the data, we will first create a training and testing data sets for the machine learning process.



## Data Exploration
-  Updated dataset to only include the variables used
-  Determine and remove outliers from data frame using quantile function within 3 standard deviations of each end of the distribution, using 99.85%       and 0.15% of the upper and lower bounds of each variable
-  Box plot visualization of outliers within the data

<img width="617" alt="Screenshot 2023-12-27 at 4 12 57 PM" src="https://github.com/Navy-Neang/R-programming-project/assets/154766577/7537db7b-eb5e-4d24-acdf-11c90d07e46b">



## Exploratory Analyses
- Exploratory analyses (for each variable) doing appropriate visualizations with ggplot2
<img width="718" alt="Screenshot 2023-12-27 at 4 35 40 PM" src="https://github.com/Navy-Neang/R-programming-project/assets/154766577/96687fd5-09fc-4c6e-8017-6fa7f9868f6c">


## Descriptive statistics
- PHYSHLTH 
  - The variance of this variable is quite high. This tells us that the data is pretty spread out. This finding in the variance is in agreement with the histogram of this variable. You see that      the data is spread out across the range of the dataset. The SD is low, which means that the data are clustered tightly around the mean.
- ALCDAY5
  - The variance of this variable is quite high. This tells us that the data is pretty spread out. This finding in the variance is in agreement with the histogram of this variable. You see that the data is spread out across the range of the dataset. The SD is low, which means that the data are clustered tightly around the mean.
- SMOKE100
  - The variance and SD are both zero, which means that there is essentially there is no spread of the data set and that it is located very tightly around the mean. Since the distribution of those who have and those who havent smoked 100 cigarettes in their entire lives are almost the same, the basic descriptive statistics tells us that there is slightly higher number of those people who have not smoked 100 cigarettes.
- CVDSTRK
  - The variance and SD are both zero, which means that there is essentially there is no spread of the data set and that it is located very tightly around the mean. This means sense since we see that the majority if not all of the respondents did not suffered from a stroke. 


## Models
Model 1 
  - CVDSTRK3 and SMOKE100 predictor combination
  - Linear regression showed positive correlation between the number of days physical health was affected by whether or not the respondent ever suffered from a stroke and those who have and          haven't smoked 100 cigarettes in their entire lives

Model 2 
  - SMOKE100
  - Linear regression showed positive correlation for physical health and smoking


## Conclusion
- Regressions of predictors ( ALCDAY5, SMOKE100 and CVDSTRK3) showed positive correlation for PHYSHLTH
- AIC was used to compare the fit of this regression model vs the other one.
- AIC explains the most variation  in the data, while penalizing the models that use excessive number of parameters.
- The lower the AIC value, the better the model fits. 
- Model 1 performed better than model 2 since it has a lower AIC value
