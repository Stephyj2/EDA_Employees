import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

#loading a data file
df = pd.read_csv('student_info.csv')
#print(df.head())
#print(df.isnull().sum())

mn = df['study_hours'].mean()
df['study_hours'] = df['study_hours'].fillna(mn)
#print(df.isnull().sum())

#checking for linearity
plt.scatter(x=df['study_hours'],y=df['student_marks'])
plt.title('Linear relation between study hrs and student marks')
plt.xlabel('Study hours')
plt.ylabel('Students marks')
plt.show()

#correlation value
cr = df['study_hours'].corr(df['student_marks'])
print(cr)

#Task try to fit using linear regression, summary stastics
from sklearn import linear_model
import statsmodels.api as sm # Don't forget to import this module

# Initialize model
df[['study_hours','student_marks']].describe() # Use a list to select multiple columns
df.plot(kind="scatter",x="study_hours",y="student_marks",figsize = (3,4),color="black")

#model fitting
y = df['study_hours']
x = df['student_marks']
x = sm.add_constant(x)
print(x)

# Linear regression using statsmodel OLS: ordinary least square method
model_sm = sm.OLS(y,x).fit()
print(model_sm.summary2())
model_sm.rsquared

# Fitting model using Scikit-learn
from sklearn import linear_model
# Initialize model
regression_model = linear_model.LinearRegression()

# Train the model using the df data
regression_model.fit(X = pd.DataFrame(df["study_hours"]),y=y)

# Check trained model y-intercept
print(regression_model.intercept_)

# Check trained model coefficients
print(regression_model.coef_)

from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
regression_model.fit(X=pd.DataFrame(df["study_hours"]), y=df["student_marks"])

r_squared = regression_model.score(X=pd.DataFrame(df["study_hours"]), y=df["student_marks"])
print(f"R² value: {r_squared}")
# 95.78% of variation is understood by LR model.
# r^2 is coefficient of determination, 95.78% of variation is undertood by our model

train_prediction = regression_model.predict(X = pd.DataFrame(df["study_hours"])) # Changed 'study_hrs' to 'study_hours'
print(train_prediction)

# Plot the new model
df.plot(kind="scatter", x="study_hours", y="student_marks",color="black", xlim=(1,11), ylim=(1,40)) # Changed 'study_hrs' to 'study_hours'
# Plot regression line
plt.plot(df["study_hours"], train_prediction,color="blue") # Changed 'study_hrs' to 'study_hours'
error_sm = model_sm.resid
print(round(sum(error_sm),2))
residuals = y - train_prediction

data = pd.DataFrame({'actual_y':y,
       'predicted_y':train_prediction,
       'error1': error_sm,
       'error2':residuals})
print(data)