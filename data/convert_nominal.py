import pandas as pd
import os

df = pd.read_csv('filtered_data/weather_mexico_preprocessed.csv')

num_cols = [
    'temperature_celsius', 'wind_kph', 'pressure_mb',
    'humidity', 'cloud', 'uv_index', 'hora_local', 'día_del_año'
]

for col in num_cols:
    labels = [f'{col}_bin{i}' for i in range(1, 6)]
    df[col] = pd.cut(df[col], bins=5, labels=labels)

os.makedirs("filtered_data", exist_ok=True)
output_path = 'filtered_data/weather_mexico_nominal.csv'
df.to_csv(output_path, index=False)

print(f"Discretizadas columnas {num_cols} en 5 bins cada una.")
print(f"Nuevo CSV guardado en {output_path}")
