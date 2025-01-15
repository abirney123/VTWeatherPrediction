#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:47:32 2024

@author: Alaina
"""

import LSTM as lstm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

sequence_lengths = [20, 30, 60, 120, 240, 480]
batch_sizes = [16, 32, 64]
#num_epochs = [10, 15, 20]
num_epochs = [15]

best_mse = float('inf')
best_params = {}


# import merged data
merged = pd.read_csv("merged.csv")
print("merged na", merged.isna().sum())
print("merged shape", merged.shape)

merged.replace("", None, inplace=True)

merged = merged[(merged['DBTCDT'] >= 0) & (merged["SW_in"] >= 0) &
                (merged["SW_out"] >= 0) & (merged["Soil_Temperature_C"] >= 0) &
                 (merged["LW_in"] >= 0) & (merged["LW_out"] >= 0) &
                 (merged["RH"] >= 0) & (merged["Soil_Moisture"] >= 0) &
                 (merged["WS_ms"] >= 0) & (merged["WindDir"] >= 0) &
                 (merged["DBTCDT"] >= 0)]
print("merged shape after removing negatives", merged.shape)
merged.dropna(inplace=True)
print("na dropped")
print("merged na", merged.isna().sum())
print("merged shape after dropping na", merged.shape)
# separate features and labels
# separate into features and labels
# location_x is location from table1 location_y is location from wind
features = merged[['TIMESTAMP', 'location_x', 'location_y','AirTC_Avg','Soil_Temperature_C', 'SW_in',
                          'SW_out','LW_in', 'LW_out', 'RH',
                          'Soil_Moisture','WS_ms', 'WindDir']]

features = features.copy()
# extract aspects of timestamp vals so they can be stored as float
features = lstm.extract_timestamp(features)

# make datetime cols in merged match features for future predictions
merged = merged.copy()
merged = lstm.extract_timestamp(merged)

# Save timestamps 
timestamps = merged["TIMESTAMP"]

# drop timestamp
features.drop("TIMESTAMP", axis=1, inplace=True)

labels = merged["DBTCDT"]
print(f"Target Standard Dev: {labels.std()}")

# check data qual
# check data quality
lstm.qual_check(features, labels)

# train test split
X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = train_test_split(features,
                                                                                       labels, timestamps, test_size=.2, shuffle=False)

# train validation split
X_train, xval, y_train, yval, timestamps_train, timestamps_val= train_test_split(X_train,
                                                                                 y_train, timestamps_train, test_size=.2, shuffle=False)

# one hot encoding for locations
# location_x, location_y all categorical
# need one-hot encoding- fit to train to prevent data leakage
categorical_features_train = X_train[["location_x", "location_y"]]

enc = OneHotEncoder(sparse_output=False)
train_encoded_features, encoded_columns = lstm.one_hot(enc,
                                                  categorical_features_train, train_or_test = "train")


train_encoded_features_df = pd.DataFrame(train_encoded_features,
                                         columns=encoded_columns, index=X_train.index)

X_train = lstm.replace_cols(X_train, categorical_features_train.columns,
                       train_encoded_features_df)


categorical_features_test = X_test[["location_x", "location_y"]]
# transform test
test_encoded_features, encoded_columns = lstm.one_hot(enc,
                                                 categorical_features_test, train_or_test = "test")

test_encoded_features_df = pd.DataFrame(test_encoded_features, columns=encoded_columns, index=X_test.index)

X_test = lstm.replace_cols(X_test, categorical_features_test.columns,
                      test_encoded_features_df)

# transform validationd ata
categorical_features_val = xval[["location_x", "location_y"]]
val_encoded_features, encoded_columns = lstm.one_hot(enc, categorical_features_val, train_or_test = "test")
val_encoded_features_df = pd.DataFrame(val_encoded_features, columns = encoded_columns, index=xval.index)
xval = lstm.replace_cols(xval, categorical_features_val.columns, val_encoded_features_df)

# standardize features and labels, save scaler_y for transforming predictions back
X_train, X_test, y_train, y_test, xval, yval, scaler_y, scaler_X = lstm.standardize(X_train,
                                                                   X_test, y_train, y_test, xval, yval)

# reshape
# reshape features and adjust labels

for seq_len in sequence_lengths:
    X_train_seq = lstm.reshape_features(X_train, seq_len)
    X_test_seq = lstm.reshape_features(X_test, seq_len)
    y_train_seq = lstm.adjust_labels(y_train, seq_len)
    y_test_seq = lstm.adjust_labels(y_test, seq_len)
    xval = lstm.reshape_features(xval, seq_len)
    yval = lstm.adjust_labels(yval, seq_len)
    
    for batch_size in batch_sizes:
        for epoch in num_epochs:
            print(f"testing sequence length {seq_len}, batch size {batch_size}, epochs {epoch}")
            model, history = lstm.define_train_model(X_train_seq, y_train_seq,
                                            epochs=epoch, batch_size=batch_size,
                                            sequence_length=seq_len,
                                            xval = xval, yval = yval)
            predictions = lstm.get_predictions(X_test_seq, model)
            mse = lstm.get_mse(y_test_seq, predictions)
            
            if mse < best_mse:
                best_mse = mse
                best_params = {'sequence_length': seq_len, 'batch_size': batch_size, "epochs": epoch}

print("Best MSE:", best_mse)
print("Best parameters:", best_params)