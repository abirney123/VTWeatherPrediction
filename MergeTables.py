#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:49:17 2024

@author: Alaina

Merge tables (table1 and wind only)
"""
import pandas as pd


# import data
def importData():
    # dont need precip or snowpk
    table1 = pd.read_csv("Exported_data_new/table1.csv")
    wind = pd.read_csv("Exported_data_new/Wind.csv")
    return table1, wind

# merge on timestamp
def merge(table1, wind):
    merged_tables = pd.merge(table1, wind, on=["TIMESTAMP","location"], how="outer")
    # save old for comparison
    merged_old_method = pd.merge(table1, wind, on="TIMESTAMP", how="outer")
    return merged_tables, merged_old_method

    
if __name__ == "__main__":
    table1, wind = importData()
    merged_tables, merged_old_method = merge(table1, wind)
    # check for missing values
    missing_vals = merged_tables.isna().sum()
    print("Missing Vals: ", missing_vals)
    # get shape
    print("Merged shape: ", merged_tables.shape)
    merged_tables.to_csv("mergedNEW.csv")
    
    """
    Old has way more because of how rows with the same value for the merge column
    are treated 
    Under old method (from GPT):
        If the column on which you are merging has non-unique values 
        (i.e., there are duplicates), each match found in the second 
        table will create a new row in the merged table for every 
        corresponding match in the first table. This is known as a Cartesian 
        product for each matched set of rows. As a result, if both tables have 
        multiple rows with the same values in the merge column, the output can 
        expand rapidly.
        
write about that mistake in report?
    """