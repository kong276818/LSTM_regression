import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. 데이터 로딩 및 결합
train = pd.read_csv("data_train.csv")
test = pd.read_csv("data_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test], axis=0).sort_values("date").reset_index(drop=True)

# 2. 메타데이터 로딩 및 유가 변수 추출
metadata = pd.read_excel("metadata.xlsx", engine="openpyxl")
oil_vars = metadata[metadata["Korean description"].str.contains("유가|WTI|브렌트", na=False)]["name"].tolist()
valid_oil_vars = [var for var in oil_vars if var in full.columns]

# 3. 시계열 피처 생성
target_col = "Natural_Gas_US_Henry_Hub_Gas"
full["month"] = full["date"].dt.month
full["year"] = full["date"].dt.year
for lag in [1, 2, 3]:
    full[f"lag_{lag}"] = full[target_col].shift(lag)
full["rolling_3"] = full[target_col].rolling(window=3).mean()
full["rolling_6"] = full[target_col].rolling(window=6).mean()

# 외부 변수 lag 생성
for var in valid_oil_vars:
    if var in full.columns:
        full[f"{var}_lag1"] = full[var].shift(1)

# 최종 feature 리스트 구성
features = [
    "month", "year", "lag_1", "lag_2", "lag_3", "rolling_3", "rolling_6"
] + [f"{v}_lag1" for v in valid_oil_vars if f"{v}_lag1" in full.columns]

# 4. 학습 데이터 구성
min_required_date = full["date"].min() + pd.Timedelta(days=6)
train_data = full[(full["date"] >= min_required_date) & (full["date"] <= "2024-06-30")].dropna(subset=features + [target_col]).reset_index(drop=True)

# 5. 정규화
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(train_data[features])
y_scaled = scaler_y.fit_transform(train_data[[target_col]])

# 6. LSTM 입력 준비
window_size = 7
X_lstm, y_lstm = [], []
for i in range(window_size, len(X_scaled)):
    X_lstm.append(X_scaled[i - window_size:i])
    y_lstm.append(y_scaled[i])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# 7. LSTM 모델 학습
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_lstm, y_lstm, epochs=30, batch_size=16, verbose=1)

# 8. 미래 예측
future_dates = pd.date_range("2024-07-01", "2024-09-30", freq='D')
predicted = []
last_window = X_scaled[-window_size:]
last_known_row = train_data.iloc[-1].copy()

for date in future_dates:
    input_window = last_window.reshape(1, window_size, -1)
    pred_scaled = model.predict(input_window, verbose=0)
    pred_value = scaler_y.inverse_transform(pred_scaled)[0][0]
    predicted.append({"date": date, "predicted_price": round(pred_value, 5)})

    new_row = last_known_row.copy()
    new_row["date"] = date
    new_row["month"] = date.month
    new_row["year"] = date.year
    new_row["lag_1"] = pred_value
    new_row["lag_2"] = new_row["lag_1"]
    new_row["lag_3"] = new_row["lag_2"]
    new_row["rolling_3"] = np.mean([pred_value] * 3)
    new_row["rolling_6"] = np.mean([pred_value] * 6)

    # 외부 변수 lag 안전 처리
    for var in valid_oil_vars:
        col_name = f"{var}_lag1"
        if col_name in features and var in full.columns:
            try:
                new_row[col_name] = full[full["date"] <= date][var].iloc[-1]
            except IndexError:
                new_row[col_name] = np.nan

    # 누락 피처 채움
    for col in features:
        if col not in new_row:
            new_row[col] = 0  # 또는 np.nan

    next_scaled = scaler_X.transform(pd.DataFrame([new_row[features]]))
    last_window = np.vstack([last_window[1:], next_scaled])
    last_known_row = new_row

# 9. 결과 저장
pred_df = pd.DataFrame(predicted)
pred_df["month"] = pred_df["date"].dt.month
pred_df["day"] = pred_df["date"].dt.day

# 월별 평균 저장
summary_df = pd.DataFrame([{
    "date": pd.to_datetime("2024-07-01"),
    "pred_1": round(pred_df[pred_df["month"] == 7]["predicted_price"].mean(), 5),
    "pred_2": round(pred_df[pred_df["month"] == 8]["predicted_price"].mean(), 5),
    "pred_3": round(pred_df[pred_df["month"] == 9]["predicted_price"].mean(), 5),
}])
summary_df.to_csv("submission_lstm_monthly.csv", index=False)

# 일자별 저장
july = pred_df[pred_df["month"] == 7][["day", "predicted_price"]].rename(columns={"predicted_price": "pred_1"})
aug = pred_df[pred_df["month"] == 8][["day", "predicted_price"]].rename(columns={"predicted_price": "pred_2"})
sep = pred_df[pred_df["month"] == 9][["day", "predicted_price"]].rename(columns={"predicted_price": "pred_3"})
by_day_df = pd.DataFrame({"date": range(1, 32)})
by_day_df = by_day_df.merge(july, left_on="date", right_on="day", how="left").drop(columns=["day"])
by_day_df = by_day_df.merge(aug, left_on="date", right_on="day", how="left").drop(columns=["day"])
by_day_df = by_day_df.merge(sep, left_on="date", right_on="day", how="left").drop(columns=["day"])
mean_row = {
    "date": "mean",
    "pred_1": by_day_df["pred_1"].mean(),
    "pred_2": by_day_df["pred_2"].mean(),
    "pred_3": by_day_df["pred_3"].mean()
}
by_day_df = pd.concat([by_day_df, pd.DataFrame([mean_row])], ignore_index=True)
by_day_df.to_csv("submission_lstm_by_month.csv", index=False)
