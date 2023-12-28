# Machine Learning Model for Grade Prediction
> Create and compare to select best machine learning model to predict final student grades.  


## Table of Contents
* [Overview](#overview)
* [Business Problem](#business-problem)
* [Imports](#imports)
* [Get the Data](#get-the-data)
* [Data Exploration](#data-exploration)
* [Prepare the Data](#prepare-the-data)
* [Promising Models](#promising-models)
* [Fine-Tune the System](#fine-tune-the-system)
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
- The first thing we will do is to define our labels (features) and target from the data set. Since we are trying to predict the grade of the students for the final period, the G3 feature will    be defined as the target while the rest of the other features will be the label features. To do so, we will drop the G3 column from the original data set and rename the dropped columns as a     Series called G3_target. The rest of the dataset will be named student_features.

Creating the training and testing data





## Data Exploration
<img width="1007" alt="Screenshot 2023-12-27 at 7 45 35 PM" src="https://github.com/Navy-Neang/Python/assets/154766577/c955fe1e-7228-4def-a111-c898cb996d18">

1. The features are distributed differently. Many of them are right skewed. This tells us that many of the
instances are in the lower range while there are few instances in the higher range resulting in a lower 
median value than the mean. 

2. There are some outliers present in the data. 

3. These features have very different scales. This will be dealt with later on.



<img width="713" alt="Screenshot 2023-12-27 at 7 45 49 PM" src="https://github.com/Navy-Neang/Python/assets/154766577/cd4db057-91a4-4225-8727-359e799dd6e6">

This box-whisker plot of the absences by age in the 3rd period of males and females highlights the outliers present when looking at this relationship. There are 12 outliers present and 6 out of those 12 occurs within the female demographics at 16 years old.

<img width="933" alt="Screenshot 2023-12-27 at 7 46 00 PM" src="https://github.com/Navy-Neang/Python/assets/154766577/97ed195e-d532-473a-841b-e04960209836">

From this Trivariate histogram with two categorical variables, we can see that there is not much of a difference in the study time for students who did have educational support from the school regardless if they received support from their family. However, when there was no support from the school we see a significant difference in the study time for both groups who did received support from their family and for those who did not. There seems to be a high number of students who studied for about 2-5 hours with the support of their family.

<img width="656" alt="Screenshot 2023-12-27 at 7 46 09 PM" src="https://github.com/Navy-Neang/Python/assets/154766577/bff0824e-469a-4d9d-bd19-60c3df1a5049">

This plot shows a strong correlation between the grades from the first period and the grades from the second period for students from Gabriel Pereira and Mousinho da Silveira. There are multiple outliers from the Gabriel Pereira school. It is not surprising there is a strong correlation of the G1 feature to the G2 feature since the grades from the first period likely correlates the students' grade in the second period. This can also be said for the G3 feature, which is what we are trying to predict for this project.


Correlation

G2 had the strongest positive correlation to G3 while failures had the strongest negative correlation


## Prepare the Data
A series of pipelines is created 

- Custom Transformer parameters:
  - when equal to True, drops the G1 and G2 columns, and when False, leaves the columns in the data
  - creates a new column in the data that sums the absences_G1, absences_G2, and absences_G3 data and then drops those three columns
- Num_Pipeline
- Cat_Pipeline
- Column Transformer


## Promising Models
Linear Regression 
  - rmse when true: 4.633957
  - rmse when false: 1.849123

Decision Tree Regression 
  - rmse when true: 5.614318
  - rmse when false: 1.834065

Random Forest Regression 
  - rmse when true: 4.245715
  - rmse when false: 1.380058

Based on the rmse value, the best model out of the three models is the random forest model when the parameter is false. While it is the best model is when the parameter is false, we will choose the random forest model when it is true just to explore the possiblity of it being a good model without the G1 and G2 data, which is inherently strongly correlated with the final grade. 


## Fine-Tune the System
We will perform a grid search using GridSearchCV() on the random forest model when it was set to false to help us determine which optimal hyperparameters to use. It will then be evaluated by cross-validation.

- GridSearchCV
- Final Model Selection and Evaluation
  - final_rmse: 4.4

The final rmse (4.4) is slightly higher than the original random forest rmse value (4.26). 

## Conclusion
The goal of this project was to build a machine learning model that will be able to accurately predict the final grades of student to determine whether or not educational support should be provided to ensure improvements in their grades for the course. Our machine learning model that was built does not show a promising performance in accurately predicing students' final grade with a final rmse value of 4.4. Based on the range of our data set of 0-20, the normalized rmse of our model would be 0.22. This means that the model is not able to accurately predict students' final grade. As noted earlier, the model of when we retained the G1 and G2 features performed way better than the selected model. Perhaps we should use that model that does include the G1 and Ge features to obtain better predictions to determine whether or not the school should provide educational support.

