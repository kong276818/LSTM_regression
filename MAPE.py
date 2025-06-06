import pandas as pd
import numpy as np

# 1. 파일 로드
pred_df = pd.read_csv("submission.csv")
real_df = pd.read_csv("real_gas_prices_2023_aug_sep_oct.csv")

# 2. 예측값 및 실제값 추출 (1행 기준)
pred_values = pred_df.loc[0, ["pred_1", "pred_2", "pred_3"]].values.tolist()
real_values = real_df["Natural_Gas_US_Henry_Hub_Gas"].values.tolist()

# 3. 길이 일치 여부 검증
assert len(pred_values) == len(real_values), "예측값과 실제값의 길이가 일치하지 않습니다."

# 4. MAPE 계산 함수
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 0 나누기 방지
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100

# 5. MAPE 계산
mape_score = mean_absolute_percentage_error(real_values, pred_values)

# 6. 결과 출력
print(f"MAPE: {mape_score:.2f}%")
