#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:22:48 2024

@author: Alaina
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import L2, L1
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


"""
changed architecture to be more similar to vamsis, using mergedNEW 
(merged on timestamp and location), added cyclical encoding for hour and month
changed timestamp extraction to be before feature extraction (was that even
                                                              happening in test before???)
"""
def define_train_model(train_features, train_labels, epochs, batch_size, sequence_length, xval, yval):
    # define model
    n_features = train_features.shape[2]
    print("model n_features", n_features)
    hidden_units = 3 #250
    responses = 1 # 1 unit output layer
    layers = [
        Input(shape=(sequence_length,n_features)),
        LSTM(hidden_units, kernel_regularizer=L1(0.01), recurrent_regularizer=L1(0.01)),
        #LSTM(hidden_units, kernel_regularizer=L2(0.1), recurrent_regularizer=L2(0.1)),
        Dropout(.2), 
        # dont use softplus it squishes
        # relu makes a lot of 0s but better than squishing
        Dense(responses, activation="relu")
        ]
    model = Sequential(layers)

    optimizer = Adam(learning_rate=.001) 
    model.compile(optimizer=optimizer, loss='mse', metrics=["mean_squared_error"]) # mse cost, adam updates (weight optimization)
    print("Model summary",model.summary())
    print("train features shape", train_features.shape," train labels shape", train_labels.shape)
    # train
    history = model.fit(train_features, train_labels, epochs = epochs,
                        batch_size = batch_size, validation_data=(xval, yval))#,
                        #callbacks = [callback])
    
    return model, history
    
def get_predictions(test_features, model):
    predictions = model.predict(test_features)
    return predictions

def get_mse(test_labels, predictions):
    mse = mean_squared_error(test_labels, predictions)
    return mse

def reshape_features(features, sequence_length):
    """
    reshape features into sequences of specified length (length is how many
    timesteps for LSTM to consider at a time)
    """
    sequences = []
    
    n_features = features.shape[1]  # Number of features
    n_samples = features.shape[0]  # Number of samples relative to timesteps
    n_sequences = n_samples - sequence_length + 1
    
    for i in range(n_sequences):
        sequence = features[i:i+sequence_length]
        sequences.append(sequence)
        
    return np.array(sequences).astype("float32")
    """
    # shape(n_samples, timesteps, features)
    features_reshaped = features.reshape((n_samples, 1, n_features))
    # keep as float
    features_reshaped = features_reshaped.astype('float32')
    print("reshape n_features: ", n_features)
    return features_reshaped
    """
def adjust_labels(y, sequence_length):
    """
    adjust labels according to sequence length
    """
    return y[sequence_length - 1:]

def adjust_timestamps(timestamps, sequence_length):
    """
    Adjust timestamps to align with the sequences of features
    """
    return timestamps[sequence_length - 1:]

def standardize(X_train, X_test, y_train, y_test, xval, yval):
    # features
    scaler_X = MinMaxScaler()
    #scaler_X = RobustScaler() # trying different scaler to handle outliers better
    #scaler_X = StandardScaler()
    # fit to train to avoid leakage
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    xval_scaled = scaler_X.transform(xval)
    # labels
    scaler_y = MinMaxScaler()
    #scaler_y = RobustScaler()
    #scaler_y = StandardScaler()
    # reshape y train and y test
    y_train_reshaped = y_train.values.reshape(-1, 1) 
    y_test_reshaped = y_test.values.reshape(-1, 1)
    yval_reshaped = yval.values.reshape(-1, 1)
    # fit and transform
    y_train_scaled = scaler_y.fit_transform(y_train_reshaped)
    y_test_scaled = scaler_y.transform(y_test_reshaped)
    yval_scaled = scaler_y.transform(yval_reshaped)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, xval_scaled, yval_scaled, scaler_y, scaler_X

def predict_future_date(model, last_sequence, scaler_X, scaler_y, sequence_length):
    # Make sure the last_sequence is a numpy array
    last_sequence = np.array(last_sequence)
    
    # Reshape and scale the last sequence
    last_sequence_scaled = scaler_X.transform(last_sequence.reshape(1, -1))
    last_sequence_reshaped = last_sequence_scaled.reshape(1, sequence_length, -1)
    
    # Make the prediction
    future_prediction_scaled = model.predict(last_sequence_reshaped)
    
    # Inverse transform the prediction to the original scale
    future_prediction = scaler_y.inverse_transform(future_prediction_scaled)
    
    return future_prediction

def prepare_last_sequence(merged, sequence_length, enc, scaler_X):
    # Select the last sequence from the dataset
    last_sequence = merged.iloc[-sequence_length:]
    
    # Separate categorical and numerical features
    categorical_features = ['location_x', 'location_y']  # update if different
    numerical_features = [col for col in last_sequence.columns if col not in categorical_features + ['TIMESTAMP']]

    # Encode categorical data
    categorical_data_encoded = enc.transform(last_sequence[categorical_features])
    encoded_columns = enc.get_feature_names_out(categorical_features)
    
    # Create DataFrame for the encoded categorical data
    categorical_data_encoded_df = pd.DataFrame(categorical_data_encoded, columns=encoded_columns, index=last_sequence.index)

    # Combine the numerical and encoded categorical data
    numerical_data = last_sequence[numerical_features]
    combined_data = pd.concat([numerical_data, categorical_data_encoded_df], axis=1)

    # Scale all features
    combined_data_scaled = scaler_X.transform(combined_data)

    # Reshape for LSTM [samples, time steps, features]
    last_sequence_prepared = combined_data_scaled.reshape(1, sequence_length, -1)

    return last_sequence_prepared

def qual_check(features, labels):
    dfs = {
    "features": features,
    "labels": labels
    }
    
    # Convert each array to a DataFrame if it's not already one
    for key in dfs:
        if not isinstance(dfs[key], pd.DataFrame):
            dfs[key] = pd.DataFrame(dfs[key])
    # print range of vals in cols
    for key in dfs:
        for col in dfs[key].columns:
            col_max = dfs[key][col].max()
            col_min = dfs[key][col].min()
            print(f"{key} col {col} range: {col_min}:{col_max}")
    # print num na
    for key in dfs:
        num_na = dfs[key].isna().sum()
        print(f"{key} na vals: {num_na}")
    # print dtypes
    for key in dfs:
        print(f"{key} dtypes: {dfs[key].dtypes}")
        
def extract_timestamp(data):
    data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"], errors="coerce")
    data['year'] = data["TIMESTAMP"].dt.year.astype(float)
    data['month'] = data["TIMESTAMP"].dt.month.astype(float)
    data['day'] = data["TIMESTAMP"].dt.day.astype(float)
    data['hour'] = data["TIMESTAMP"].dt.hour.astype(float)
    data['minute'] = data["TIMESTAMP"].dt.minute.astype(float)
    data['second'] = data["TIMESTAMP"].dt.second.astype(float)
    return data

def one_hot(enc, categorical_cols, train_or_test):
    """
    categorical_cols must be df, train_or_test sepcifies if train or test
    to decide to use fit_transform or just transform
    """
    if train_or_test == "train":
        encoded_features = enc.fit_transform(categorical_cols)
    if train_or_test == "test":
        encoded_features = enc.transform(categorical_cols)
        
    encoded_columns = enc.get_feature_names_out(input_features=categorical_cols.columns)
    return encoded_features, encoded_columns

def replace_cols(data, col_to_drop, new_col):
    # drop old cols and put new encoded features in as new columns 
    data = data.drop(columns=col_to_drop)
    data = pd.concat([data, new_col], axis=1)
    return data


    
# import merged data
merged = pd.read_csv("mergedNEW.csv")
print("merged na on import", merged.isna().sum())
print("merged shape", merged.shape)

# check for blank strings
count_empty = (merged == "").sum()
print("Count of blank strings in each column on import:\n", count_empty)
# no blank strings present

#print("Merged Na")
#print(merged.isna().sum()) # see what columns have na, if not used in features or 
# targets can just remove those cols before dropping na
"""
cols with na that aren't being used for features or labels:
TS_LOC_RECORD          18784
RECORD_x               18784
Batt_volt_Min          18784
PTemp                  18784
shf                    58403
SWE                   409547
Ice_Content           409548
Water_Content         409545
Snowpack_Density      409545
Qual                  112836
TCDT                  134000
RECORD_y               87398
WS_ms_Max             102740
WS_ms_TMx             270967
WS_ms_S_WVT           106471
WindDir_D1_WVT        106471
WindDir_SD1_WVT       106471
WS_ms_Min             106471
WS_ms_TMn             274698
    
    drop these cols before dropping na to retain more data
"""


# add LaPlace noise to "Target_Depth" to avoid data leakage
scale = 0.1 
merged['Target_Depth_noisy'] = merged['Target_Depth'] + np.random.laplace(0, scale, merged['Target_Depth'].shape[0])

"""
dont remove batt volt min, ptemp, shf, TCDT, Qual- needed for correlation matrix
merged.drop(labels=["TS_LOC_RECORD", "RECORD_x", "Batt_volt_Min", "PTemp", 
                    "shf", "SWE", "Ice_Content", "Water_Content", "Snowpack_Density",
                    "Qual", "TCDT", "RECORD_y", "WS_ms_Max", "WS_ms_TMx",
                    "WS_ms_S_WVT", "WindDir_D1_WVT", "WindDir_SD1_WVT",
                    "WS_ms_Min", "WS_ms_TMn"], axis=1, inplace=True)
"""
merged.drop(labels=["TS_LOC_RECORD", "SWE", "Ice_Content", "Water_Content", "Snowpack_Density",
                    "RECORD_x", "RECORD_y", "WS_ms_Max", "WS_ms_TMx", "WS_ms_S_WVT",
                    "WindDir_D1_WVT", "WindDir_SD1_WVT", "WS_ms_Min", "WS_ms_TMn"], axis=1, inplace=True)


# print ranges of cols- not doing locations because non numeric
# and cant be negative anyways
feature_label_cols = ['TIMESTAMP', "DBTCDT", 'AirTC_Avg','Soil_Temperature_C', 'SW_in',
                          'SW_out','LW_in', 'LW_out', 'RH',
                          'Soil_Moisture','WS_ms', 'WindDir', "Target_Depth_noisy"]

print("Feature ranges before splitting/ handling negatives and na")

for col in feature_label_cols:
    col_max = merged[col].max()
    col_min = merged[col].min()
    print(f"col {col} range: {col_min}:{col_max}")

print("merged shape before removing negatives", merged.shape)
    

# For columns that shouldn't be negative, values below 0 but close to it
# are likely meant to be 0. Values far below 0 are erroneous. if within -5 of 0 change to 
# 0, else remove

non_neg_cols = ["DBTCDT", "SW_in", "SW_out", "LW_in", "LW_out", "RH",
                "Soil_Moisture", "WS_ms", "WindDir", "Target_Depth_noisy"]
for col in non_neg_cols:
    merged[col] = np.where((merged[col] >= -5) & (merged[col] < 0),0, merged[col])
    
merged = merged[(merged['DBTCDT'] >= 0) & (merged["SW_in"] >= 0) &
                (merged["SW_out"] >= 0) & (merged["LW_in"] >= 0) & 
                (merged["LW_out"] >= 0) & (merged["RH"] >= 0) &
                (merged["Soil_Moisture"] >= 0) & (merged["WS_ms"] >= 0) &
                (merged["WindDir"] >= 0)]

print("Feature ranges after handling negatives")

for col in feature_label_cols:
    col_max = merged[col].max()
    col_min = merged[col].min()
    print(f"col {col} range: {col_min}:{col_max}")

print("merged shape after removing negatives", merged.shape)
merged = merged.copy()
print("merged na after removing negatives", merged.isna().sum())
merged.dropna(inplace=True)

print("na dropped")
#print("merged na", merged.isna().sum())
print("merged shape after dropping na", merged.shape)

# save merged to csv for eda
merged.to_csv("CleanMerged.csv")


# extract aspects of timestamp vals so they can be stored as float
#features = features.copy()
#features = extract_timestamp(features)

merged = extract_timestamp(merged)

# Cyclical encoding for hour and month
merged['hour_sin'] = np.sin(merged['hour'] * (2. * np.pi / 24))
merged['hour_cos'] = np.cos(merged['hour'] * (2. * np.pi / 24))
merged['month_sin'] = np.sin((merged['month'] - 1) * (2. * np.pi / 12))
merged['month_cos'] = np.cos((merged['month'] - 1) * (2. * np.pi / 12))


# Save timestamps 
timestamps = merged["TIMESTAMP"]

# separate features and labels
# separate into features and labels
# location_x is location from table1 location_y is location from wind
features = merged[['TIMESTAMP', 'location','AirTC_Avg','Soil_Temperature_C', 'SW_in',
                          'SW_out','LW_in', 'LW_out', 'RH',
                          'Soil_Moisture','WS_ms', 'WindDir', "hour_sin", "hour_cos",
                          "month_sin", "month_cos", "Target_Depth_noisy"]]




# drop timestamp
features = features.copy()
features.drop("TIMESTAMP", axis=1, inplace=True)

labels = merged["DBTCDT"]

# check data qual
# check data quality
#qual_check(features, labels)

# train test split
X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = train_test_split(features,
                                                                                       labels, timestamps, test_size=.2, shuffle=False)

# train validation split
X_train, xval, y_train, yval, timestamps_train, timestamps_val= train_test_split(X_train,
                                                                                 y_train, timestamps_train, test_size=.2, shuffle=False)



# one hot encoding for locations
# location_x, location_y all categorical
# need one-hot encoding- fit to train to prevent data leakage
categorical_features_train = X_train[["location"]]

# store column indices to location names mapping
location_mapping_train = {index: location_name for index, location_name in enumerate(categorical_features_train['location'].unique())}


enc = OneHotEncoder(handle_unknown="ignore",sparse_output=False) # added handle_unknown = ignore
train_encoded_features, encoded_columns = one_hot(enc,
                                                  categorical_features_train, train_or_test = "train")


train_encoded_features_df = pd.DataFrame(train_encoded_features,
                                         columns=encoded_columns, index=X_train.index)

X_train = replace_cols(X_train, categorical_features_train.columns,
                       train_encoded_features_df)


categorical_features_test = X_test[["location"]]

# store column indices to location names mapping
location_mapping_test = {index: location_name for index, location_name in enumerate(categorical_features_test['location'].unique())}

# transform test
test_encoded_features, encoded_columns = one_hot(enc,
                                                 categorical_features_test, train_or_test = "test")

test_encoded_features_df = pd.DataFrame(test_encoded_features, columns=encoded_columns, index=X_test.index)

X_test = replace_cols(X_test, categorical_features_test.columns,
                      test_encoded_features_df)

# save x test cols
X_test_cols = X_test.columns
print(f"X_test Columns: \n {X_test_cols}")

# transform validation data (one hot)
categorical_features_val = xval[["location"]]
val_encoded_features, encoded_columns = one_hot(enc, categorical_features_val, train_or_test = "test")
val_encoded_features_df = pd.DataFrame(val_encoded_features, columns = encoded_columns, index=xval.index)
xval = replace_cols(xval, categorical_features_val.columns, val_encoded_features_df)

#save to csv if desired
X_train.to_csv("X_train.csv")
X_test.to_csv("X_test.csv")
y_train.to_csv("y_train.csv")
y_test.to_csv("y_test.csv")

print("Num PROC in X train", X_train["location_PROC"].sum())
print("Num RB10 in X train", X_train["location_RB10"].sum())
print("Num SPST in X train", X_train["location_SPST"].sum())

print("Num PROC in X test", X_test["location_PROC"].sum())
print("Num RB10 in X test", X_test["location_RB10"].sum())
print("Num SPST in X test", X_test["location_SPST"].sum())

# standardize features and labels, save scaler_y for transforming predictions back
X_train, X_test, y_train, y_test, xval, yval, scaler_y, scaler_X = standardize(X_train,
                                                                   X_test, y_train, y_test, xval, yval)
batch_size = 32
sequence_length = 10#144 #20 # 60 144 timestamps is a day

# reshape
# reshape features
X_train_reshaped = reshape_features(X_train, sequence_length)
X_test_reshaped = reshape_features(X_test, sequence_length)
xval_reshaped = reshape_features(xval, sequence_length)

print("features reshaped")
print("X train dtype", X_train_reshaped.dtype)
print("X test dtype", X_test_reshaped.dtype)
print("y train dtype", y_train.dtype)
print("y test dtype", y_test.dtype)


# adjust labels according to sequence
y_train = adjust_labels(y_train, sequence_length)
y_test = adjust_labels(y_test, sequence_length)
yval = adjust_labels(yval, sequence_length)

# match timestamps to sequences
timestamps_train = adjust_timestamps(timestamps_train, sequence_length)
timestamps_test = adjust_timestamps(timestamps_test, sequence_length)
timestamps_val = adjust_timestamps(timestamps_val, sequence_length)


# set epochs 
epochs = 10

# get variance of target
sd = np.std(y_train)
variance = sd**2
print("target variance", variance)

has_negatives = np.any(y_train < 0)

print("Does y_train have negatives?", has_negatives)

# train model
print("Beginning model training...")
print(f"{X_train.shape[0]} samples for training")
model, history = define_train_model(X_train_reshaped, y_train, epochs,
                                    batch_size, sequence_length, xval_reshaped, yval)
print("model trained")
# get predictions
predictions = get_predictions(X_test_reshaped, model)
# revert predictions
predictions = scaler_y.inverse_transform(predictions)
print("predictions found")
predictions_with_timestamps = pd.DataFrame({
'Timestamp': timestamps_test.reset_index(drop=True),
'Prediction': predictions.flatten()})

# initialize dataframe to store predictions with locations and timestamps
#predictions_timestamps_locs = pd.DataFrame(columns=["Timestamp", "Location", "Prediction"])

# create X_test_df 
X_test_df = pd.DataFrame(X_test, columns = X_test_cols)
# identify location of each prediction to output with prediction and timestamp
# initialize list to store predictions
all_predictions = []
for index, row in predictions_with_timestamps.iterrows():
    prediction = row["Prediction"]
    timestamp = row["Timestamp"]
    location_cols = [col for col in X_test_df.columns if "location_" in col]
    encoded_location = X_test_df.loc[index, location_cols]
    location_name = location_cols[np.argmax(encoded_location.values)]
    
    location_name = location_name.replace("location_", "")
    # add to list
    all_predictions.append({"Timestamp": timestamp,
                            "Location": location_name,
                            "Prediction": prediction})
# add list to df
predictions_timestamps_locs = pd.DataFrame(all_predictions)
    
    
# store truth with timestamps and locations for comparison
truth_with_timestamps_locs = pd.DataFrame({
    "Timestamp": timestamps_test.reset_index(drop=True),
    "Y test": y_test.flatten()})

print("Test Predictions: \n", predictions_timestamps_locs)
unique_predictions = np.unique(predictions)
print(unique_predictions)
# get mse - first inverse transform y_test
y_test = scaler_y.inverse_transform(y_test)
mse = get_mse(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print("MSE: ", mse)
print("MAE: ", mae)

plt.figure()
plt.plot(history.history['mean_squared_error'], label="mse")
plt.plot(history.history['val_mean_squared_error'], label="val_mse")
plt.title("Train and Validation MSE")
plt.legend()
plt.show()

# visualize predictions and actual values
plt.figure(figsize=(12, 6))
plt.plot(timestamps_test, y_test, label='Actual Values')
plt.plot(timestamps_test, predictions, label='Predicted Values', alpha=0.7)
plt.title('Actual vs Predicted Snow Depth')
plt.xlabel('Date')
plt.ylabel('Snow Depth')
plt.legend()
plt.show()


