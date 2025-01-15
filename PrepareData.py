#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:49:17 2024

@author: Alaina

Merge tables and get subset to use as parameters
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# import data
def importData():
    precipitation = pd.read_csv("DS_Exported_data/Precipitation.csv", dtype=str)
    SnowpkTempProfile = pd.read_csv("DS_Exported_data/SnowpkTempProfile.csv", dtype=str)
    table1 = pd.read_csv("DS_Exported_data/table1.csv", dtype=str)
    wind = pd.read_csv("DS_Exported_data/Wind.csv", dtype=str)
    return precipitation, SnowpkTempProfile, table1, wind

# merge on timestamp
def merge(precipitation, SnowpkTempProfile, table1, wind):
    precip_snow = pd.merge(precipitation, SnowpkTempProfile, on="TIMESTAMP", how="outer", suffixes=("_precip", "_snow"))
    precip_snow_table1 = pd.merge(precip_snow, table1, on="TIMESTAMP", how="outer", suffixes=("_precip_snow", "_table1"))
    merged_tables = pd.merge(precip_snow_table1, wind, on="TIMESTAMP", how="outer", suffixes=("precip_snow_table1", "_wind"))
    return merged_tables

# replace blank strings with nan
def replace_with_nan(value):
    if isinstance(value, str) and (value.isspace() or value == ""):
        return np.nan
    else:
        return value

def prepare_data(merged_tables, target):
    """
    target is string representing name of column to use for labels
    
    handles missing values, gets train test split, converts values to float 
    for use in LSTM, reshapes features
    """
    # check for whitespace and empty strings
    for col in merged_tables.columns:
        is_blank = merged_tables[col].apply(lambda x: isinstance(x, str) and (x.isspace() or x == ""))
        print(f"col {col} blanks: {is_blank.sum()}")
 
    
    # replace whitespace and empty strings with nan if blanks present
    if is_blank.sum() > 0:
        merged_tables = merged_tables.map(replace_with_nan)

    
    # see how many rows contain missing values
    print("na:", merged_tables.isna().sum())
    print("filling na...")
    # cannot drop all rows that contain missing values because we will be left
    # with an empty table. Use last observation carried forward (LOCF) instead 
    # length of gaps should be checked to see how robust using LOCF is
    merged_tables.ffill(inplace=True)
    print("na filled")
    print("na:", merged_tables.isna().sum())
    
    print("new shape before separating: ", merged_tables.shape)
    
    # separate into features and labels
    features = merged_tables[['TIMESTAMP', 'location_precip', 'location_snow',
                              'locationprecip_snow_table1','location_wind',
                              'AirTC_Avg','Soil_Temperature_C', 'SW_in',
                              'SW_out','LW_in', 'LW_out', 'RH',
                              'Soil_Moisture','WS_ms', 'WindDir']]
    
    labels = merged_tables[target]

    print("converting values to float, splitting data, preforming one-hot encoding...")
    features = features.copy()
    # Extract year, month, day, hour, minute, and second components for converting timestamp to float
    features["TIMESTAMP"] = pd.to_datetime(features["TIMESTAMP"], errors="coerce")
    features['year'] = features["TIMESTAMP"].dt.year.astype(float)
    features['month'] = features["TIMESTAMP"].dt.month.astype(float)
    features['day'] = features["TIMESTAMP"].dt.day.astype(float)
    features['hour'] = features["TIMESTAMP"].dt.hour.astype(float)
    features['minute'] = features["TIMESTAMP"].dt.minute.astype(float)
    features['second'] = features["TIMESTAMP"].dt.second.astype(float)
    
    # drop timestamp
    features.drop("TIMESTAMP", axis=1, inplace=True)

    labels = labels.astype(float)
    
    # convert values to float

    # AirTC_Avg, SW_in, RH, Soil_Moisture, DBTCDT all numeric
    numeric_feat = ["AirTC_Avg", "SW_in", "RH", "Soil_Moisture"]
    for col in numeric_feat:
        #features.loc[:, col] = features[col].astype(float)
        features.loc[:,col] = pd.to_numeric(features.loc[:,col], errors = "coerce")
 
    # train test split before one hot encoding to convert categorical vars to float
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2, shuffle=False)
    print(f"X_train na: {X_train.isna().sum()}")
    print(f"X_test na: {X_test.isna().sum()}")
    print(f"y_train na: {y_train.isna().sum()}")
    print(f"y_test na: {y_test.isna().sum()}")
    
    # location_precip, location_snow, locationprecip_snow_table1, location_wind all categorical
    # need one-hot encoding- fit to train to prevent data leakage
    categorical_features_train = X_train[["location_precip", "location_snow",
                                    "locationprecip_snow_table1", "location_wind"]]
    enc = OneHotEncoder(sparse_output=False)
    train_encoded_features = enc.fit_transform(categorical_features_train)
    encoded_columns = enc.get_feature_names_out(input_features=categorical_features_train.columns)
    train_encoded_features_df = pd.DataFrame(train_encoded_features, columns=encoded_columns, index=X_train.index)
    # drop old cols and put new encoded features in as new columns
    X_train = X_train.drop(columns=categorical_features_train.columns)  # Drop original categorical columns
    X_train = pd.concat([X_train, train_encoded_features_df], axis=1)
    
    categorical_features_test = X_test[["location_precip", "location_snow",
                                    "locationprecip_snow_table1", "location_wind"]]
    test_encoded_features = enc.transform(categorical_features_test)
    # put new encoded features in as new columns and drop old cols
    encoded_columns = enc.get_feature_names_out(input_features=categorical_features_test.columns)
    test_encoded_features_df = pd.DataFrame(test_encoded_features, columns=encoded_columns, index=X_test.index)
    
    X_test = X_test.drop(columns=categorical_features_test.columns)
    X_test = pd.concat([X_test, test_encoded_features_df], axis=1)
    

    return X_train, X_test, y_train, y_test

def main():
    precipitation, SnowpkTempProfile, table1, wind = importData()
    print("data imported")
    merged_tables = merge(precipitation, SnowpkTempProfile, table1, wind)
    print("tables merged")
    # predicting snow depth- DBTCDT
    X_train, X_test, y_train, y_test = prepare_data(merged_tables, "DBTCDT")
    print("data prepared")
    X_train.to_csv("X_train.csv")
    X_test.to_csv("X_test.csv")
    y_train.to_csv("y_train.csv")
    y_test.to_csv("y_test.csv")
    print("data saved to csvs")
    
if __name__ == "__main__":
    main()