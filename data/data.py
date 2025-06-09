import pandas as pd
import os

df = pd.read_csv('GlobalWeatherRepository.csv')

df_mexico = df[df['country'] == 'Mexico'].copy()

df_mexico['last_updated'] = pd.to_datetime(df_mexico['last_updated'], errors='coerce')
df_mexico['hora_local']  = df_mexico['last_updated'].dt.hour
df_mexico['día_del_año'] = df_mexico['last_updated'].dt.dayofyear  

def agrupar_condicion(texto):
    t = str(texto).lower()
    if 'sunny' in t or 'clear' in t:
        return 'Despejado'
    if 'cloudy' in t or 'overcast' in t:
        return 'Nublado'
    if 'rain' in t or 'drizzle' in t:
        return 'Lluvia'
    return 'Otro'

df_mexico['CondCategoria'] = df_mexico['condition_text'].apply(agrupar_condicion)

a_descartar = [
    'location_name',
    'latitude', 'longitude',
    'timezone',
    'last_updated_epoch',
    'last_updated',
    'temperature_fahrenheit',
    'feels_like_celsius', 'feels_like_fahrenheit',
    'condition_text',
    'wind_mph',
    'wind_degree', 'wind_direction',
    'pressure_in',
    'precip_mm', 'precip_in',
    'visibility_km', 'visibility_miles',
    'gust_mph', 'gust_kph',
    'air_quality_Carbon_Monoxide', 'air_quality_Ozone',
    'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide',
    'air_quality_PM2.5', 'air_quality_PM10',
    'air_quality_us-epa-index', 'air_quality_gb-defra-index',
    'sunrise', 'sunset',
    'moonrise', 'moonset', 'moon_phase', 'moon_illumination'
]

df_pre = df_mexico.drop(columns=a_descartar, errors='ignore')

os.makedirs("filtered_data", exist_ok=True)
df_pre.to_csv("filtered_data/weather_mexico_preprocessed.csv", index=False)

print("Columnas finales conservadas:")
print(df_pre.columns.tolist())
print("\nPrimeras filas del CSV preprocesado:")
print(df_pre.head())
