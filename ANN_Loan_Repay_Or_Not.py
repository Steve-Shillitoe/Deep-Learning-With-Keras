"""
A model to predict whether someone will pay back their load or not, based on historical data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])

feat_info('mort_acc')

######################################################
# Load Data
######################################################
df = pd.read_csv('lending_club_loan_two.csv')

print(df.info())

#####################################################
# Exploratory Data Analysis
#####################################################