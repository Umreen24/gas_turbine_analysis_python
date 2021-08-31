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

from plotnine import ggplot, aes, geom_histogram, labs


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
     geom_histogram(binwidth = 2, 
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
     geom_histogram(binwidth = 0.5, 
                    fill = 'pink', 
                    colour = 'red', 
                    size = 0.5, 
                    alpha = 0.5) + 
     labs(title = "Nitrogen Oxide Distribution", 
          x = "NOX", 
          y = "Count")
)
