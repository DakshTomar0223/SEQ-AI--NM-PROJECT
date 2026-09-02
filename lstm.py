import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import kagglehub
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from tensorflow import keras

os.makedirs("outputs", exist_ok=True)

dataset_folder = kagglehub.dataset_download("stytch16/jena-climate-2009-2016")
csv_file = glob.glob(os.path.join(dataset_folder, "*.csv"))[0]
df = pd.read_csv(csv_file)
print(f"Imported dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(df.head())


date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col], format="%d.%m.%Y %H:%M:%S")
df = df.set_index(date_col).resample("1h").mean().ffill().bfill()

TARGET = "T (degC)"                     
target_idx = df.columns.get_loc(TARGET)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)


WINDOW = 24
X, y = [], []
for i in range(len(scaled) - WINDOW):
    X.append(scaled[i:i + WINDOW])
    y.append(scaled[i + WINDOW, target_idx])
X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

split = int(len(X) * 0.8)              
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train examples: {len(X_train):,} | Test examples: {len(X_test):,}")


model = keras.Sequential([
    keras.layers.Input(shape=(WINDOW, df.shape[1])),
    keras.layers.LSTM(50),              
    keras.layers.Dense(1),               
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

model.fit(X_train, y_train, validation_split=0.1, epochs=15, batch_size=64)


pred_scaled = model.predict(X_test).flatten()

def unscale(col_values):
    dummy = np.zeros((len(col_values), df.shape[1]))
    dummy[:, target_idx] = col_values
    return scaler.inverse_transform(dummy)[:, target_idx]

y_pred = unscale(pred_scaled)
y_true = unscale(y_test)


mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-3, 1e-3, y_true))) * 100
r2 = r2_score(y_true, y_pred)


actual_up = (y_true[1:] > y_true[:-1]).astype(int)
pred_up = (y_pred[1:] > y_pred[:-1]).astype(int)
acc = accuracy_score(actual_up, pred_up)
prec = precision_score(actual_up, pred_up, zero_division=0)
rec = recall_score(actual_up, pred_up, zero_division=0)
f1 = f1_score(actual_up, pred_up, zero_division=0)

print("\n--- ERROR METRICS ---")
print(f"MAE  (avg degrees off):  {mae:.2f} °C")
print(f"RMSE (avg, big misses count extra): {rmse:.2f} °C")
print(f"MAPE (avg % off): {mape:.1f}%")
print(f"R^2  (0-1, pattern explained): {r2:.3f}")

print("\n--- 'TEMP UP OR DOWN?' CLASSIFICATION METRICS ---")
print(f"Accuracy:  {acc:.1%}")
print(f"Precision: {prec:.1%}")
print(f"Recall:    {rec:.1%}")
print(f"F1 score:  {f1:.3f}")


plt.figure(figsize=(10, 4))
plt.plot(y_true[:200], label="Actual")
plt.plot(y_pred[:200], label="Predicted")
plt.xlabel("Hours into test set")
plt.ylabel("Temperature (°C)")
plt.title("LSTM: Predicted vs Actual Temperature")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/prediction_vs_actual.png", dpi=150)
print("\nSaved plot to outputs/prediction_vs_actual.png")

model.save("outputs/simple_lstm_model.keras")
print("Saved model to outputs/simple_lstm_model.keras")