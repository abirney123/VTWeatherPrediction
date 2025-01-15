#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:36:04 2024

@author: Alaina

This script drops columns that produce errors when uploading to the database.
These columns will not be needed for our model anyways, so in the interest of 
time we just drop them at this stage.
"""

import pandas as pd
from pathlib import Path

def dropCols(locations):
    for location in locations:
        # load data: precipitation, SnowpkTempProfile, and wind for each location
        # change to your relative file paths
        wind_path = Path(f"DS_New_data/{location}/Wind.csv")
        if wind_path.exists():
            wind = pd.read_csv(f"DS_New_data/{location}/Wind.csv")
            if "WS_ms_TMx" in wind.columns:
                # drop the unnecessary columns with errors from the corresponding table
                wind.drop("WS_ms_TMx", axis=1, inplace=True)
            if "WS_ms_TMn" in wind.columns:
                wind.drop("WS_ms_TMn", axis=1, inplace=True)
            wind.to_csv(f"DS_New_data/{location}/Wind.csv")
        else:
            print(f"Wind data not found for {location}")
        precip_path = Path(f"DS_New_data/{location}/precipitation.csv")
        if precip_path.exists():
            precipitation = pd.read_csv(f"DS_New_data/{location}/precipitation.csv")
            # drop the unnecessary columns with errors from the corresponding table
            if "Accu_RT_NRT" in precipitation.columns:
                precipitation.drop("Accu_RT_NRT", axis=1, inplace=True)
                # save CSV
                # change this to your relative file path
                precipitation.to_csv(f"DS_New_data/{location}/precipitation.csv")
        else:
            print(f"Precipitation data not found for {location}")
        snow_path = Path(f"DS_New_data/{location}/SnowpkTempProfile.csv")
        if snow_path.exists():
            SnowpkTempProfile = pd.read_csv(f"DS_New_data/{location}/SnowpkTempProfile.csv")
            if "T107_C_160cm_Avg" in SnowpkTempProfile.columns:
                # drop the unnecessary columns with errors from the corresponding table
                SnowpkTempProfile.drop("T107_C_160cm_Avg", axis=1, inplace=True)
            if "T107_C_80cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_80cm_Avg", axis=1, inplace=True)
            if "T107_C_170cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_170cm_Avg", axis=1, inplace=True)
            if "T107_C_180cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_180cm_Avg", axis=1, inplace=True)
            if "T107_C_190cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_190cm_Avg", axis=1, inplace=True)
            if "T107_C_200cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_200cm_Avg", axis=1, inplace=True)
            if "T107_C_100cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_100cm_Avg", axis=1, inplace=True)
            if "T107_C_60cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_60cm_Avg", axis=1, inplace=True)
            if "T107_C_130cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_130cm_Avg", axis=1, inplace=True)
            if "T107_C_280cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_280cm_Avg", axis=1, inplace=True)
            if "T107_C_110cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_110cm_Avg", axis=1, inplace=True)
            if "T107_C_290cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_290cm_Avg", axis=1, inplace=True)
            if "T107_C_90cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_90cm_Avg", axis=1, inplace=True)
            if "T107_C_0cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_0cm_Avg", axis=1, inplace=True)
            if "T107_C_140cm_Avg" in SnowpkTempProfile.columns:
                SnowpkTempProfile.drop("T107_C_140cm_Avg", axis=1, inplace=True)
            SnowpkTempProfile.to_csv(f"DS_New_data/{location}/SnowpkTempProfile.csv")
                
        else:
            print(f"SnowpkTempProfile data not found for {location}")
        

def main():
    locations = ["PTSH", "RB01", "RB02", "RB03", "RB04", "RB05", "SPST", "SUMM"]
    dropCols(locations)
    
if __name__ == "__main__":
    main()