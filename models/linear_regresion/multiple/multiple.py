import pandas as pd
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('../../data/filtered_data/weather_mexico_preprocessed.csv')

predictors = ['wind_kph', 'pressure_mb', 'humidity', 'cloud', 
              'uv_index', 'hora_local', 'día_del_año']
X = df[predictors].values
y = df['temperature_celsius'].values

kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
model = LinearRegression()

y_pred = cross_val_predict(model, X, y, cv=kf10, n_jobs=-1)

mae  = mean_absolute_error(y, y_pred)
mse  = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

start = time.time()
train_idx, _ = next(kf10.split(X))
model.fit(X[train_idx], y[train_idx])
duration = time.time() - start

model.fit(X, y)
w0 = model.intercept_
w  = model.coef_
duration = time.time() - start


print(f"\nIntercepto (w0): {w0:.4f}")
for name, coef in zip(predictors, w):
    print(f"Coeficiente de {name:12s}: {coef:.4f}")

print(f"MAE (10-fold):  {mae:.4f}")
print(f"RMSE (10-fold): {rmse:.4f}")
print(f"Tiempo (s):     {duration:.4f}")


