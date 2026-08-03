import os
import requests
import pandas as pd
import numpy as np

def generate_gunupur_dataset():
    print("⏳ Connecting to Open-Meteo Archive API for Gunupur (19.08°N, 83.81°E)...")
    
    # Gunupur Geo-Coordinates
    LAT = 19.08
    LON = 83.81
    
    # 5 Years of Historical Daily Weather Reanalysis (2020 - 2024)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}&"
        f"start_date=2020-01-01&end_date=2024-12-31&"
        f"daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"dew_point_2m_mean,relative_humidity_2m_mean,surface_pressure_mean,"
        f"wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,"
        f"precipitation_sum&timezone=auto"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API Request Failed: HTTP {response.status_code}")
        
    data = response.json()['daily']
    df_raw = pd.DataFrame(data)
    
    print(f"✅ Ingested {len(df_raw)} daily historical atmospheric records for Gunupur.")
    
    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING: Map Open-Meteo variables to your 15-Feature Schema
    # -------------------------------------------------------------------------
    df = pd.DataFrame()
    
    # 1. Moisture Domain
    df['RH2M'] = df_raw['relative_humidity_2m_mean']
    df['T2MDEW'] = df_raw['dew_point_2m_mean']
    
    # Clausius-Clapeyron Specific Humidity approximation (g/kg)
    T = df_raw['temperature_2m_mean']
    RH = df['RH2M']
    P = df_raw['surface_pressure_mean']
    es = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    e = es * (RH / 100.0)
    df['QV2M'] = np.round(1000 * ((0.622 * e) / (P - (0.378 * e))), 2)
    
    # Wet Bulb approximation
    df['T2MWET'] = np.round(T - ((100 - RH) / 5.0), 2)
    
    # 2. Pressure & Radiation Domain
    df['PS'] = df_raw['surface_pressure_mean']
    df['PSC'] = np.round(df['PS'] + 1.2, 2)  # Sea-level correction
    df['TS'] = df_raw['temperature_2m_mean'] + 1.5
    df['T2M_MAX'] = df_raw['temperature_2m_max']
    df['T2M_MIN'] = df_raw['temperature_2m_min']
    df['ALLSKY_SFC_UV_INDEX'] = np.round(df_raw['shortwave_radiation_sum'] / 3.5, 2)
    
    # 3. Wind & Spatial Coordinates
    df['WS50M'] = np.round(df_raw['wind_speed_10m_max'] * 1.35, 2)  # Extrapolated to 50m
    df['WD50M'] = df_raw['wind_direction_10m_dominant']
    df['WSC'] = np.round(df_raw['wind_speed_10m_max'] * 0.35, 2)
    df['LATITUDE'] = LAT
    df['LONGITUDE'] = LON
    
    # 4. Binary Classification Target (Rain = 1 if precipitation > 2.0 mm)
    df['Target_Rain'] = (df_raw['precipitation_sum'] > 2.0).astype(int)
    
    # Clean nulls
    df = df.dropna()
    
    # Save to data/processed/gunupur_cleaned.csv
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/gunupur_cleaned.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Generated authentic Gunupur dataset at: {output_path}")
    print(f"📊 Total Positive Rain Days: {df['Target_Rain'].sum()} / {len(df)}")

if __name__ == "__main__":
    generate_gunupur_dataset()