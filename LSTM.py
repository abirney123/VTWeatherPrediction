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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import L2, L1
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


"""
target depth and tcdt are not features
because that would be data leakage (dbtcdt is a calculation based on target depth
                                    and tcdt)

tried increasing sequence length, made take very long time and didnt improve
(overall mse was worse, everything else was about the same)

1 day is approx 247 rows of data, sequence length of 500 did not really help at all



dropout .2, 250 hidden units, 10 epochs got 2939 mse
.1, .2, .2 L2 reg

current settings nicer mse but few predictions
250 hidden, .01 reg in first lstm layer did a bit better with sequence len 20
"""
def define_train_model(train_features, train_labels, epochs, batch_size, sequence_length, xval, yval):
    # define model
    n_features = train_features.shape[2]
    print("model n_features", n_features)
    hidden_units = 30 #250
    responses = 1 # 1 unit output layer
    layers = [
        Input(shape=(sequence_length,n_features)),
        # adding l2 reg. and dropout to help with overfititing
        Dropout(.2), 
        LSTM(hidden_units, kernel_regularizer=L1(0.01), recurrent_regularizer=L1(0.01),
             return_sequences=True),#, .001, .05
        LSTM(hidden_units, kernel_regularizer=L1(0.1), recurrent_regularizer=L1(0.1)),#,
             #return_sequences = True),
        #LSTM(hidden_units, kernel_regularizer=L2(0.1), recurrent_regularizer=L2(0.1)), # new
        # use softplus to avoid negative output but also avoid squashing to one val like relu did
        #Dense(hidden_units, activation="tanh", activity_regularizer = L2(0.25)),
        Dense(hidden_units, activation="tanh", activity_regularizer = L1(0.1)),
        # had one more dense but removed
        Dense(responses, activation="softplus")
        ]
    model = Sequential(layers)
    # define early stopping criteria
    #callback = EarlyStopping(monitor="val_loss", patience=1)

    #model.add(Input(shape=(1,n_features)))
    #model.add(LSTM(units=4, activation="relu"))
    #model.add(Dense(1)) # 1 unit in output layer
    optimizer = Adam(learning_rate=.0001) 
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

def main():
    
    # import merged data
    merged = pd.read_csv("merged.csv")
    print("merged na", merged.isna().sum())
    print("merged shape", merged.shape)
    
    merged.replace("", None, inplace=True)

    #print("Merged Na")
    #print(merged.isna().sum()) # see what columns have na, if not used in features or 
    # targets can just remove those cols before dropping na
    """
    cols with na that aren't being used for features or labels:
        - shf
        - SWE
        - Ice_Content
        -Water_Content         2755227
        -Snowpack_Density      2755227
        -Target_Depth           623672
        -Qual                   623672
        -TCDT                   773051
        -RECORD_y                12463
        -WS_ms_Max               73786
        -WS_ms_TMx             1571144
        -WS_ms_S_WVT             97929
        -WindDir_D1_WVT          97929
        -WindDir_SD1_WVT         97929
        -WS_ms_Min               97929
        -WS_ms_TMn             1595287
        
        drop these cols before dropping na to retain more data
    """

    merged.drop(labels=["shf", "SWE", "Ice_Content", "Water_Content", "Snowpack_Density",
                "Target_Depth", "Qual", "TCDT", "RECORD_y", "WS_ms_Max", "WS_ms_TMx",
                "WS_ms_S_WVT", "WindDir_D1_WVT", "WindDir_SD1_WVT", "WS_ms_Min",
                "WS_ms_TMn"], axis=1, inplace=True)
    
    # drop rows where any of the following are negative:
    """
    (some were found for SW_in, SW_out) in data quality check
    SW_in, SW_out, LW_in, LW_out, RH, 
    Soil_Moisture, WS_ms, WindDir, DBTCDT
    """
    # print ranges of cols- not doing locations because non numeric
    # and cant be negative anyways
    feature_label_cols = ['TIMESTAMP', "DBTCDT", 'AirTC_Avg','Soil_Temperature_C', 'SW_in',
                              'SW_out','LW_in', 'LW_out', 'RH',
                              'Soil_Moisture','WS_ms', 'WindDir']
    
    print("Feature ranges before splitting/ handling negatives and na")
    for col in feature_label_cols:
        col_max = merged[col].max()
        col_min = merged[col].min()
        print(f"col {col} range: {col_min}:{col_max}")
        
    print("merged shape before removing negatives", merged.shape)
        
    """
    For columns that shouldn't be negative, values below 0 but close to it
    are likely meant to be 0. Values far below 0 are erroneous if within -1 of 0 change to 
    0, else remove
    """
    non_neg_cols = ["DBTCDT", "SW_in", "SW_out", "LW_in", "LW_out", "RH",
                    "Soil_Moisture", "WS_ms", "WindDir"]
    for col in non_neg_cols:
        merged[col] = np.where((merged[col] >= -.1) & (merged[col] < 0),0, merged[col])
        
    merged = merged[(merged['DBTCDT'] >= 0) & (merged["SW_in"] >= 0) &
                    (merged["SW_out"] >= 0) & (merged["LW_in"] >= 0) & 
                    (merged["LW_out"] >= 0) & (merged["RH"] >= 0) &
                    (merged["Soil_Moisture"] >= 0) & (merged["WS_ms"] >= 0) &
                    (merged["WindDir"] >= 0)]
    
    print("Feature ranges after splitting/ handling negatives and na")
    for col in feature_label_cols:
        col_max = merged[col].max()
        col_min = merged[col].min()
        print(f"col {col} range: {col_min}:{col_max}")
    print("merged shape after removing negatives", merged.shape)
    
    merged.dropna(inplace=True)

    
    print("na dropped")
    #print("merged na", merged.isna().sum())
    print("merged shape after dropping na", merged.shape)
    
    # separate features and labels
    # separate into features and labels
    # location_x is location from table1 location_y is location from wind
    features = merged[['TIMESTAMP', 'location_x', 'location_y','AirTC_Avg','Soil_Temperature_C', 'SW_in',
                              'SW_out','LW_in', 'LW_out', 'RH',
                              'Soil_Moisture','WS_ms', 'WindDir']]
    
    features = features.copy()
    # extract aspects of timestamp vals so they can be stored as float
    features = extract_timestamp(features)

    # make datetime cols in merged match features for future predictions
    merged = merged.copy()
    merged = extract_timestamp(merged)

    # Save timestamps 
    timestamps = merged["TIMESTAMP"]
    
    # drop timestamp
    features.drop("TIMESTAMP", axis=1, inplace=True)
    
    labels = merged["DBTCDT"]
    print(f"Target Standard Dev: {labels.std()}")
    
    # check data qual
    # check data quality
    qual_check(features, labels)

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
    
    enc = OneHotEncoder(handle_unknown="ignore",sparse_output=False) # added handle_unknown = ignore
    train_encoded_features, encoded_columns = one_hot(enc,
                                                      categorical_features_train, train_or_test = "train")
    
    
    train_encoded_features_df = pd.DataFrame(train_encoded_features,
                                             columns=encoded_columns, index=X_train.index)
    
    X_train = replace_cols(X_train, categorical_features_train.columns,
                           train_encoded_features_df)

    
    categorical_features_test = X_test[["location_x", "location_y"]]
    # transform test
    test_encoded_features, encoded_columns = one_hot(enc,
                                                     categorical_features_test, train_or_test = "test")
    
    test_encoded_features_df = pd.DataFrame(test_encoded_features, columns=encoded_columns, index=X_test.index)
    
    X_test = replace_cols(X_test, categorical_features_test.columns,
                          test_encoded_features_df)
    
    # transform validationd ata
    categorical_features_val = xval[["location_x", "location_y"]]
    val_encoded_features, encoded_columns = one_hot(enc, categorical_features_val, train_or_test = "test")
    val_encoded_features_df = pd.DataFrame(val_encoded_features, columns = encoded_columns, index=xval.index)
    xval = replace_cols(xval, categorical_features_val.columns, val_encoded_features_df)
    
    X_train.to_csv("X_train.csv")
    X_test.to_csv("X_test.csv")
    y_train.to_csv("y_train.csv")
    y_test.to_csv("y_test.csv")
    
    # standardize features and labels, save scaler_y for transforming predictions back
    X_train, X_test, y_train, y_test, xval, yval, scaler_y, scaler_X = standardize(X_train,
                                                                       X_test, y_train, y_test, xval, yval)

    
    
    # found to be optimal hyperparams through hp sweep
    batch_size = 32
    sequence_length = 20 #20 # 60 
    
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
    epochs = 15 

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
    
    print("Test Predictions: \n", predictions_with_timestamps)
    unique_predictions = np.unique(predictions)
    print(unique_predictions)
    # get mse - first inverse transform y_test
    y_test = scaler_y.inverse_transform(y_test)
    mse = get_mse(y_test, predictions)
    print("MSE: ", mse)
    
    plt.figure()
    plt.plot(history.history['mean_squared_error'], label="mse")
    plt.plot(history.history['val_mean_squared_error'], label="val_mse")
    plt.title("Train and Validation MSE")
    plt.legend()
    plt.show()
    

    # Target SD= 50 (variance = 2500), so MSE should be < 2500 to represent 
    # better performance than predicting mean (with negative values)
    
    # Target SD=49.58 (variance=2458.1764), so MSE should be < 2458 to 
    #  represent better performance than predicting mean (without negative values)
    """
    Attempting to predict the future- need to handle columns during enc better
    and associate with timestamp for interpretability
    
    # get last sequence for predicting future
    last_sequence = merged.iloc[-sequence_length:][features.columns]
    
    # one hot encoding
    categorical_features_last_seq = last_sequence[["location_x", "location_y"]]
    numerical_features_last_seq = last_sequence.drop(columns=["location_x", "location_y"])
    
    last_sequence_enc = enc.transform(categorical_features_last_seq)

    
    encoded_columns = enc.get_feature_names_out(input_features=categorical_features_last_seq.columns)
    
    # make last_sequence dataframe
    last_sequence = pd.DataFrame(last_sequence_enc, columns= last_sequence.columns, index=last_sequence.index)
    
    last_seq_encoded_features_df = pd.DataFrame(last_sequence, columns=encoded_columns, index=last_sequence.index)
    
    # drop old cols and put new encoded features in as new columns for train
    last_sequence = last_sequence.drop(columns=categorical_features_last_seq.columns)
    last_sequence = pd.concat([last_sequence, last_seq_encoded_features_df], axis=1)
    
    # standardize
    last_sequence_scaled = scaler_X.transform(last_sequence)
    
    # Reshape for LSTM [samples, time steps, features]
    last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, last_sequence_scaled.shape[1])
    
    future_prediction = predict_future_date(model, last_sequence, scaler_X, scaler_y, sequence_length)
    print("Future prediction:", future_prediction)
    """
if __name__ == "__main__":
    main()