import pandas as pd
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('../../data/filtered_data/weather_mexico_preprocessed.csv')

predictors = ['wind_kph', 'pressure_mb', 'humidity', 'cloud', 
              'uv_index', 'hora_local', 'día_del_año']
results = []
kf10 = KFold(n_splits=10, shuffle=True, random_state=42)

for var in predictors:
    X = df[[var]].values
    y = df['temperature_celsius'].values
    model = LinearRegression()
    
    y_pred = cross_val_predict(model, X, y, cv=kf10, n_jobs=-1)

    mae  = mean_absolute_error(y, y_pred)
    mse  = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    start = time.time()
    train_idx, _ = next(kf10.split(X))
    model.fit(X[train_idx], y[train_idx])
    duration = time.time() - start
    
    results.append({
        'Variable':       var,
        'MAE (10-fold)': round(mae, 4),
        'RMSE (10-fold)': round(rmse, 3),
        'Tiempo (s)':     round(duration, 4)
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

