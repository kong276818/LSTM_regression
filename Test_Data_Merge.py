import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 데이터 로딩
train = pd.read_csv("data_train.csv")
test = pd.read_csv("data_test.csv")
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
full = pd.concat([train, test], axis=0).sort_values("date").reset_index(drop=True)

# 메타데이터에서 유가 관련 변수 추출
metadata = pd.read_excel("metadata.xlsx", engine="openpyxl")
oil_vars = metadata[metadata['Korean description'].str.contains("유가|WTI|브렌트", na=False)]['name'].tolist()
valid_oil_vars = [var for var in oil_vars if var in full.columns]

# 시계열 파생 변수
target_col = "Natural_Gas_US_Henry_Hub_Gas"
full["month"] = full["date"].dt.month
full["year"] = full["date"].dt.year
for lag in [1, 2, 3]:
    full[f"lag_{lag}"] = full[target_col].shift(lag)
full["rolling_3"] = full[target_col].rolling(3).mean()
full["rolling_6"] = full[target_col].rolling(6).mean()
for var in valid_oil_vars:
    full[f"{var}_lag1"] = full[var].shift(1)

# feature 구성
features = ['month', 'year', 'lag_1', 'lag_2', 'lag_3', 'rolling_3', 'rolling_6'] + \
           [f"{v}_lag1" for v in valid_oil_vars if f"{v}_lag1" in full.columns]

# 학습 데이터 (7/31까지)
min_required_date = full['date'].min() + pd.Timedelta(days=6)
train_data = full[(full['date'] >= min_required_date) & (full['date'] <= '2024-07-31')].dropna().reset_index(drop=True)

# 정규화
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(train_data[features])
y_scaled = scaler_y.fit_transform(train_data[[target_col]])

# LSTM 입력 구성
window_size = 7
X_lstm, y_lstm = [], []
for i in range(window_size, len(X_scaled)):
    X_lstm.append(X_scaled[i - window_size:i])
    y_lstm.append(y_scaled[i])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# 모델 학습
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_lstm, y_lstm, epochs=20, batch_size=16, verbose=1)

# 예측 (8~10월)
future_dates = pd.date_range("2024-08-01", "2024-10-31")
target_dates = pd.to_datetime(["2024-08-31", "2024-09-30", "2024-10-31"])
predicted = []

last_window = X_scaled[-window_size:]
last_known_row = train_data.iloc[-1].copy()

for date in future_dates:
    input_window = last_window.reshape(1, window_size, -1)
    pred_scaled = model.predict(input_window, verbose=0)
    pred_value = scaler_y.inverse_transform(pred_scaled)[0][0]

    if date in target_dates:
        predicted.append(round(pred_value, 5))

    # 다음 입력용 업데이트
    new_row = last_known_row.copy()
    new_row["date"] = date
    new_row["month"] = date.month
    new_row["year"] = date.year
    new_row["lag_1"] = pred_value
    new_row["lag_2"] = new_row["lag_1"]
    new_row["lag_3"] = new_row["lag_2"]
    new_row["rolling_3"] = np.mean([pred_value] * 3)
    new_row["rolling_6"] = np.mean([pred_value] * 6)
    for var in valid_oil_vars:
        new_row[f"{var}_lag1"] = full[full['date'] <= date][var].iloc[-1]
    
    next_scaled = scaler_X.transform(pd.DataFrame([new_row[features]]))
    last_window = np.vstack([last_window[1:], next_scaled])
    last_known_row = new_row

# 저장
df_submit = pd.DataFrame([{
    "date": "2024-07-31",
    "pred_1": predicted[0],
    "pred_2": predicted[1],
    "pred_3": predicted[2],
}])
df_submit.to_csv("submission.csv", index=False)

