import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    mean_absolute_error, mean_squared_error,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../../data/filtered_data/weather_mexico_nominal.csv')
X = df.drop(columns=['CondCategoria']).astype(str)
y = df['CondCategoria'].astype(str)
for col in X.columns:
    X[col] = LabelEncoder().fit_transform(X[col])
le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)

clf = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.0, random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

y_pred = cross_val_predict(clf, X, y_enc, cv=kf, n_jobs=-1)
y_proba = cross_val_predict(clf, X, y_enc, cv=kf, method='predict_proba', n_jobs=-1)

acc = accuracy_score(y_enc, y_pred)
kappa = cohen_kappa_score(y_enc, y_pred)
mae = mean_absolute_error(y_enc, y_pred)
mse = mean_squared_error(y_enc, y_pred)
rmse = np.sqrt(mse)
y_mean = np.full_like(y_enc, int(round(y_enc.mean())))
ra = mae / mean_absolute_error(y_enc, y_mean) * 100
rrse = np.sqrt(mse / mean_squared_error(y_enc, y_mean)) * 100
unclassified = np.sum(np.isnan(y_pred))

start = time.time()
train_idx, _ = next(kf.split(X))
clf.fit(X.iloc[train_idx], y_enc[train_idx])
train_time = time.time() - start

y_bin = pd.get_dummies(y_enc).values
roc_auc_macro = roc_auc_score(y_bin, y_proba, average='macro')

print("=== ID3 (sklearn DecisionTreeClassifier sin poda) ===\n")
print(f"Correctly Classified Instances: {int(acc * len(y_enc))}\t{acc*100:6.4f} %")
print(f"Incorrectly Classified Instances: {len(y_enc) - int(acc*len(y_enc))}\t{(1-acc)*100:6.4f} %")
print(f"Kappa statistic:                 {kappa: .4f}")
print(f"Mean absolute error:             {mae:.4f}")
print(f"Root mean squared error:         {rmse:.4f}")
print(f"Relative absolute error:         {ra:.4f} %")
print(f"Root relative squared error:     {rrse:.4f} %")
print(f"UnClassified Instances:          {unclassified}")
print(f"ROC AUC (macro):                 {roc_auc_macro:.4f}")
print(f"Training time (one fold):        {train_time:.4f} s\n")

print("=== Detailed Accuracy By Class ===")
print(classification_report(y_enc, y_pred, target_names=le_target.classes_, digits=4))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_enc, y_pred)
print(pd.DataFrame(cm, index=le_target.classes_, columns=le_target.classes_))
