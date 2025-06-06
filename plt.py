import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV 파일 로드
real_df = pd.read_csv("real_gas_prices_2023_aug_sep_oct.csv")
pred_df = pd.read_csv("submission.csv")

# 2. 월 레이블 지정
months = ['August', 'September', 'October']

# 3. 실제값 및 예측값 추출
real_values = real_df["Natural_Gas_US_Henry_Hub_Gas"].tolist()
pred_values = [pred_df['pred_1'][0], pred_df['pred_2'][0], pred_df['pred_3'][0]]

# 4. 선 그래프 시각화
plt.figure(figsize=(8, 5))
plt.plot(months, real_values, label='Real Price (2023)', marker='o', linewidth=2)
plt.plot(months, pred_values, label='Predicted Price (2024)', marker='o', linewidth=2)
plt.title("Henry Hub Natural Gas Prices: Real (2023) vs Predicted (2024)")
plt.ylabel("Price (USD/MMBtu)")
plt.ylim(0, 10)  # y축 범위 0~10으로 설정
plt.grid(True)
plt.legend()
plt.tight_layout()

# 5. 그래프 저장
plt.savefig("real_vs_predicted_gas_price_scaled_0to10.png")
plt.show()
