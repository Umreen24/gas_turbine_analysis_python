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
import math
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import normal_ad

from plotnine import ggplot, aes, geom_histogram, labs, geom_boxplot

"""""""""""""""
FUNCTIONS
"""""""""""""""
def read_data(file):
    loaded = pd.read_csv(file)
    
    return loaded


"""""""""""""""
ANALYSIS
"""""""""""""""
# Bring in gas turbine data from 2013 
gt_data = read_data('gt_2013.csv')
CO = gt_data.iloc[:, 9]
NOX = gt_data.iloc[:, 10]

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

# Boxplots for CO and NOX 
plt.boxplot([CO, NOX])
plt.xticks([1, 2], ["CO", "NOX"])
plt.title("CO and NOX Emissions Boxplot")

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

# Create visual correlation between attributes
corr_mat = gt_filtered.corr()
plt.figure(figsize = (16, 10))
sns.heatmap(corr_mat, annot = True)
plt.show()

# Create linear regression models
for col in gt_filtered.columns:
    print(col)

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

# Plotting residuals of both CO and NOX models to check normality
# NOX model
sm.qqplot(res_NOX.resid)

# CO model
sm.qqplot(res_CO.resid)

# Perform Anderson-Darling normality test
# NOX model
normal_ad(res_NOX.resid)

# CO model
normal_ad(res_CO.resid)
