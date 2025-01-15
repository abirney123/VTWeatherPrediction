#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:50:58 2024

@author: Alaina
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

mergedEDA = pd.read_csv("CleanMerged.csv")

# drop RB10 and RB11 and SPST, get dates as own col
mergedEDA = mergedEDA.copy()
#mergedEDA = mergedEDA[~mergedEDA['location'].isin(['RB10', 'RB11', "SPST"])]
mergedEDA['TIMESTAMP'] = pd.to_datetime(mergedEDA['TIMESTAMP'])
mergedEDA['Date'] = mergedEDA['TIMESTAMP'].dt.date
# change name of SUMM_NewData to SUMM
mergedEDA['location'] = mergedEDA['location'].replace('SUMM_NewData', 'SUMM')

print("min ", mergedEDA["DBTCDT"].min())
print("max ",mergedEDA["DBTCDT"].max())

# groupby location
grouped = mergedEDA.groupby("location")

# eda
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 14))

for location, group in grouped:
    # extract date from timestamp for x axis labels
    group.plot(x="Date", y="AirTC_Avg", ax= axes[0], label=location, title="Air Temperature Average Over Time")
    group.plot(x="Date", y="Soil_Moisture", ax= axes[1], label=location, title="Soil Moisture Over Time")
    group.plot(x="Date", y="DBTCDT", ax= axes[2], label=location, title="Snow Depth (DBTCDT) Over Time")

# Set y-axis labels
axes[0].set_ylabel('Scaled Air Temperature')
axes[1].set_ylabel('Scaled Soil Moisture')
axes[2].set_ylabel('Snow Depth')

# Add legends
axes[0].legend(title='Location')
axes[1].legend(title='Location')
axes[2].legend(title='Location')

plt.tight_layout()
plt.show()


# Correlation matrix among attributes
mergedEDA_no_time = mergedEDA.drop(labels=["TIMESTAMP", "Unnamed: 0.1", "Unnamed: 0", "location", "Date", "Target_Depth_noisy", "WindDir", "WS_ms"], axis=1)
correlation_matrix = mergedEDA_no_time.corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 10))
ax = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Features: Multiple locations')
plt.tight_layout()
plt.show()

print(mergedEDA_no_time.describe())

