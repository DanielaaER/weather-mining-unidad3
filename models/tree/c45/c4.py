import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../../data/filtered_data/weather_mexico_preprocessed.csv')
features = ['temperature_celsius','wind_kph','pressure_mb','humidity','cloud','uv_index','hora_local','día_del_año']
X = df[features].values
y = df['CondCategoria'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.02, min_samples_leaf=2, random_state=42)

kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X, y_enc, cv=kf10, n_jobs=-1)
y_proba = cross_val_predict(clf, X, y_enc, cv=kf10, method='predict_proba', n_jobs=-1)

accuracy = accuracy_score(y_enc, y_pred)
mae      = mean_absolute_error(y_enc, y_pred)
mse      = mean_squared_error(y_enc, y_pred)
rmse     = np.sqrt(mse)

y_mean = y_enc.mean()
ra   = mae  / np.mean(np.abs(y_enc - y_mean)) * 100
rrse = np.sqrt(np.sum((y_enc - y_pred)**2) / np.sum((y_enc - y_mean)**2)) * 100

y_bin = pd.get_dummies(y_enc).values
roc_auc = roc_auc_score(y_bin, y_proba, average=None)

start = time.time()
train_idx, _ = next(kf10.split(X))
clf.fit(X[train_idx], y_enc[train_idx])
duration = time.time() - start

print("=== C4.5 pruned (sklearn DecisionTreeClassifier) ===\n")
print(f"Accuracy:                 {accuracy*100:.4f}%")
print(f"Mean absolute error (MAE):{mae:.4f}")
print(f"Root mean squared error:  {rmse:.4f}")
print(f"Rel. absolute error (%):  {ra:.4f}%")
print(f"RRSE (%):                 {rrse:.4f}%")
print(f"Tiempo entrenamiento:     {duration:.4f} s\n")

roc_auc_mean = roc_auc_score(y_bin, y_proba, average='macro')
print(f"ROC AUC promedio (macro): {roc_auc_mean:.4f}")


print("=== Detailed Accuracy By Class ===")
print(classification_report(y_enc, y_pred, target_names=le.classes_, digits=4))

print("=== Confusion Matrix ===")
print(pd.DataFrame(confusion_matrix(y_enc, y_pred), index=le.classes_, columns=le.classes_))
