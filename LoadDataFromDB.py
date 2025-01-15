#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:36:56 2024
"""
import pandas as pd
import numpy as np
from mysql.connector import connect, ProgrammingError, IntegrityError
from pathlib import Path
 
# Database configuration
USER = 'root'
PASSWORD = ''  # Securely fetch this value in a real application
HOST = '127.0.0.1'
DATABASE = 'FinalProject'
ROOT_DIRECTORY = Path('/Users/lu/Desktop/DataScience/FinalProject/DS_New_Data')

 
ataLOCATIONS_TO_LOAD = ['RB05']
 
def create_database_connection():
    """Establishes database connection."""
    return connect(user=USER, password=PASSWORD, host=HOST, database=DATABASE)
 
def get_table_columns(cursor, table_name):
    """Retrieves column names for a specified table."""
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    return [column[0] for column in cursor.fetchall()]
 
def adjust_dst(df, time_column='TIMESTAMP'):
    """Adjusts datetime values that fall within the DST change window."""
    dst_start = pd.to_datetime('2023-03-12 02:00:00')
    dst_end = pd.to_datetime('2023-11-05 02:00:00')
    mask_spring = (df[time_column] >= dst_start) & (df[time_column] < dst_start + pd.Timedelta(hours=1))
    mask_fall = (df[time_column] >= dst_end) & (df[time_column] < dst_end + pd.Timedelta(hours=1))
    df.loc[mask_spring, time_column] += pd.Timedelta(hours=1)
    df.loc[mask_fall, time_column] -= pd.Timedelta(hours=1)
    return df
 
def process_csv_file(file_path):
    """Processes a CSV file into a DataFrame, ready for insertion into the database."""
    df = pd.read_csv(file_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df = adjust_dst(df)
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.strftime('%Y-%m-%d %H:%M:%S').replace('NaT', None)
 
    columns_to_replace_zeros = ['SW_in', 'SW_out', 'LW_in', 'LW_out']
    for column in columns_to_replace_zeros:
        if column in df.columns:
            df[column] = df[column].replace({0: None})
 
    df = df.where(pd.notnull(df), None)
    data_quality_flag = 0
 
    med_attributes_mapping = {
        'DBTCDT_Med': 'DBTCDT',
        'Target_Depth_Med': 'Target_Depth',
        'Qual_Med': 'Qual',
        'TCDT_Med': 'TCDT'
    }
   
    for med_attr, standard_attr in med_attributes_mapping.items():
        if med_attr in df.columns:
            df.rename(columns={med_attr: standard_attr}, inplace=True)
            data_quality_flag = 1
   
    df['DataQualityFlag'] = data_quality_flag
   
    return df
 
def insert_data(df, table_name, cursor):
    """Dynamically inserts data into the specified table based on available columns."""
    if df.empty:
        print(f"No data found for table {table_name}. Skipping.")
        return
   
    table_columns = get_table_columns(cursor, table_name)
    common_columns = [col for col in df.columns if col in table_columns]
 
    columns_sql = ', '.join([f"`{column}`" for column in common_columns])
    placeholders = ', '.join(['%s'] * len(common_columns))
    update_clause = ', '.join([f"`{column}`=VALUES(`{column}`)" for column in common_columns])
   
    insert_stmt = f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_clause};"
   
    data_tuples = [tuple(row[col] for col in common_columns) for index, row in df.iterrows()]
    batch_size = 500  # Adjust based on your environment's optimal batch size
    for i in range(0, len(data_tuples), batch_size):
        batch_records = data_tuples[i:i+batch_size]
        try:
            cursor.executemany(insert_stmt, batch_records)
        except IntegrityError as e:
            print(f"IntegrityError inserting records into {table_name}: {e}")
        except ProgrammingError as e:
            print(f"ProgrammingError inserting records into {table_name}: {e}")
 
def extract_table_name(file_name):
    """Extracts table name from the given file name."""
    return file_name.split('_')[-1].split('.')[0]
 
def main():
    """Main function to execute database operations."""
    conn = create_database_connection()
    cursor = conn.cursor()
 
    for location in LOCATIONS_TO_LOAD:
        location_directory = ROOT_DIRECTORY / location
        if location_directory.is_dir():
            for file_path in location_directory.glob('*.csv'):
                table_name = extract_table_name(file_path.name)
                df = process_csv_file(file_path)
                df['location'] = location
                df = df[[col for col in get_table_columns(cursor, table_name) if col in df.columns]]
                insert_data(df, table_name, cursor)
                conn.commit()
 
    cursor.close()
    conn.close()
 
if __name__ == "__main__":
    main()