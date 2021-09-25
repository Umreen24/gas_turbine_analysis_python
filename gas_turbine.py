#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:48:27 2021

@author: umreenimam
"""

"""""""""""""""
IMPORTING PACKAGES
"""""""""""""""
import os 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import normal_ad
from collections import defaultdict
from plotnine import ggplot, aes, geom_histogram, labs

"""""""""""""""
FUNCTIONS
"""""""""""""""
def read_data(file):
    loaded = pd.read_csv(file)
    
    return loaded

"""""""""""""""
INTRODUCTION: 

With the ever increasing demand for energy significant amount of strain has been placed on economies to meet these 
requirements. In doing so it has inadvertently pose additional vital threat the global society causing tremendous environment 
and health concerns. The emissions of harmful gasses has become an international discussion among many world leaders. 
So much so strict laws and penalties have been enforced to aid in the control of pollutants and greenhouse gasses 
that results in catastrophic damages to forestry, wildlife and human society.  

Two of the main harmful gasses are CO and NOx. One popular source of these pollutant is through the combustion process 
from gas turbines at power plants. The goal of our team is to analyze data acquired from data gather from sensors that 
were installed in a gas turbine to validate if a linear regression model could be effective computed using statistical 
approaches to justify our findings. The hope is that we will be able to identify variables that significantly contribute 
to the emission of CO and NOx gasses and recommend solutions to aid in the reduction of these pollutants without 
impacting the energy yield by these machines.  

The dataset was sourced from the 
UCI Machine Learning Repository: (https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set#)  

It consist of over 36,000 records with 11 attributes. These include Ambient temperature (AT), 
Ambient pressure (AP), Ambient humidity (AH), Air filter difference (AFDP), 
Gas turbine exhaust pressure (GTEP), Turbine inlet temperature (TIT), 
Compressor discharge pressure (CDP), Turbine energy yeild (TEY), 
Carbon monoxide (CO) and Nitrogen oxides (NOx) which obtained from senors 
strategically installed on the gas turbine. The 11 sensor measurement were aggregated over an hour (by means of average sum).   

The data was acquired from 2011 to 2015.However, for the scope of this project 
only the data from 2013 (over seven thousand records) was used due to 
large number of records and limited computer power. The 11 sensor measurement were 
aggregated over an hour (by means of average sum) from a gas turbine located in Turkey's north western region.

"""""""""""""""

"""""""""""""""
ANALYSIS
"""""""""""""""
# Bring in gas turbine data from 2013 
gt_data = read_data('gt_2013.csv')
CO = gt_data.iloc[:, 9]
NOX = gt_data.iloc[:, 10]

"""""""""""""""
The initial distribution of both the CO and NOX data shows that it is positively skewed. 
This could be due to multiple reasons ranging from faulty or improper 
maintained instrumentation that were out of calibration. 
"""""""""""""""
# Visualize carbon monoxide and nitrogen oxide distribution 
# Carbon monoxide distribution
(
     ggplot(gt_data, aes(x = CO)) + 
     geom_histogram(binwidth = 0.5, 
                    fill = 'pink', 
                    colour = 'red', 
                    size = 0.5, 
                    alpha = 0.5) + 
     labs(title = "Carbon Monoxide Distribution", 
          x = "CO", 
          y = "Count")
)


# Nitrogen oxide (NOX) distribution
(
     ggplot(gt_data, aes(x = NOX)) + 
     geom_histogram(binwidth = 2, 
                    fill = 'pink', 
                    colour = 'red', 
                    size = 0.5, 
                    alpha = 0.5) + 
     labs(title = "Nitrogen Oxide Distribution", 
          x = "NOX", 
          y = "Count")
)

"""""""""""""""
To help with the visualization we created a few box plots. 
They clearly indicates presence of data that far exceed the 
4th quantile and may potentially be outliers. However, the majority of the 
data still falls the first and 3rd quantile.
"""""""""""""""

# Boxplots for CO and NOX 
plt.boxplot([CO, NOX])
plt.xticks([1, 2], ["CO", "NOX"])
plt.title("CO and NOX Emissions Boxplot")

"""""""""""""""
To help remove possible outliners that introduce the skewness potential 
inaccurate data we carfully applied filters based on the 
box plot representation and analysis of Cook's distance.
"""""""""""""""
# Remove outliers    
gt_filtered = gt_data[(gt_data.CO > 0.5) & (gt_data.CO < 2.5) 
                      & (gt_data.NOX > 25) & (gt_data.NOX < 90)]

# Re-plot CO & NOX data
CO_new = gt_filtered.iloc[:, 9]
NOX_new = gt_filtered.iloc[:, 10]

(
     ggplot(gt_filtered, aes(x = CO_new)) + 
     geom_histogram(binwidth = 0.5,
                    bins = 20,
                    fill = 'pink', 
                    colour = 'red', 
                    size = 0.5, 
                    alpha = 0.5) + 
     labs(title = "Carbon Monoxide Distribution", 
          x = "CO", 
          y = "Count")
)


# Nitrogen oxide (NOX) distribution
(
     ggplot(gt_filtered, aes(x = NOX_new)) + 
     geom_histogram(binwidth = 2, 
                    fill = 'pink', 
                    colour = 'red', 
                    size = 0.5, 
                    alpha = 0.5) + 
     labs(title = "Nitrogen Oxide Distribution", 
          x = "NOX", 
          y = "Count")
)

"""""""""""""""
Correlation was observed across a few of the regressor variables, 
specifically between the GTEP, TIT, TAT, TEY and CDP. 
This highlights possible existence of multicollinearity. 
We may have to remove some of the attribute to eliminate the 
presence of redundancy between the features.
"""""""""""""""
# Create visual correlation between attributes
corr_mat = gt_filtered.corr()
plt.figure(figsize = (16, 10))
sns.heatmap(corr_mat, annot = True)
plt.show()

# Create linear regression models
#for col in gt_filtered.columns:
#    print(col)

"""""""""""""""
Initially, we prepared a linear model using all of the data variables to see 
if there are any indication of significant regression inspect how the well 
the model will fit by plotting the distribution of the residuals.   
"""""""""""""""
# NOX model
reg_NOX = smf.ols('NOX ~ AT + AP + AH + AFDP + GTEP + TIT + TAT + TEY + CDP + CO', 
              data = gt_filtered)

res_NOX = reg_NOX.fit()
print(res_NOX.summary())

# CO model
reg_CO = smf.ols('CO ~ AT + AP + AH + AFDP + GTEP + TIT + TAT + TEY + CDP + NOX', 
              data = gt_filtered)

res_CO = reg_CO.fit()
print(res_CO.summary())

"""""""""""""""
Looking at the residual plots it shows the data for NOX has still suffers from 
some positive skewness. 
However, the CO appears residuals that are normally distributed. 
The points on the normal q-q plot mostly fall on the straight line. 
"""""""""""""""
# Plotting residuals of both CO and NOX models to check normality
# NOX model
sm.qqplot(res_NOX.resid)

# CO model
sm.qqplot(res_CO.resid)

"""""""""""""""
To dive in a bit deeper and help verify the level of normal distribution of 
the residual we performed an Anderson-Darling Normality Test. 
"""""""""""""""
# Perform Anderson-Darling normality test
# NOX model
ad_NOX = normal_ad(res_NOX.resid)
pval_NOX = ad_NOX[1]
print('NOX p-value: ' + str(pval_NOX))

# CO model
ad_CO = normal_ad(res_CO.resid)
print(ad_CO)
pval_CO = ad_CO[1]
print('CO p-value: ' + str(pval_CO))

"""""""""""""""
Based on the results from the Anderson-Darling Normality Test we can see that 
its p_value is extremely low much lower than 0.05 for NOX data 
but higher for the CO data. For the NOX we can reject the null hypothesis in 
favor of the alternative and state that the response variable 
does not fits a normal distribution. 

However we see for the CO model that the P_value is greater than 0.05 and 
we can state that with over 95% confidence the data 
does fits a normal distribution.  

For the scope of this project we plan to focus only on CO as the main 
response variable moving forward. 
As part as a future opportunity analysis we can always re-visit looking more 
into NOX but for now we will place our attention on the CO data. 

We also re-visited the model and removed any non-significant 
regressors with high p_values (i.e > 0.05).
"""""""""""""""
# Revisit CO summary
print(res_CO.summary())

# Remove non-significant regressors with high p-values
# Based on the OLS summary CDP seems to have a p-value> 0.05
gt_filtered = gt_filtered.drop('CDP', axis = 1)

# Re-run model without CDP regressor 
reg_CO = smf.ols('CO ~ AT + AP + AH + AFDP + GTEP + TIT + TAT + TEY + NOX', 
              data = gt_filtered)

res_CO = reg_CO.fit()
print(res_CO.summary())

"""""""""""""""
Based on the initial results we see that multicollienarity may exist between 
many of the regressors as it was shown earlier from the correlation plot 
that some regressors were highly correlated. 
"""""""""""""""
# Check for multicollinearity 
vif_check = pd.DataFrame()
vif_check = gt_filtered.drop('CO', axis = 1)

cc = np.corrcoef(vif_check, rowvar = False)
VIF = np.linalg.inv(cc)
vif_diag = VIF.diagonal()

# Get a better view of which elements have multicollinearity 
# Get a list of the column names from filtered data frame
gt_names = gt_filtered.columns.values.tolist()

# Remove 'CO' from gt_names list
index = 8
gt_names.pop(index)

gt_names = pd.Series(gt_names)
vif_diag = pd.Series(vif_diag)
vif_df = pd.concat([gt_names, vif_diag], axis = 1)

vif_df.columns = ['Gas', 'VIF']
print(vif_df)
