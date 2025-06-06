import pandas as pd

# 1. 데이터 로드
train = pd.read_csv("data_train.csv")
test = pd.read_csv("data_test.csv")

# 2. 날짜 파싱
train["date"] = pd.to_datetime(train["date"], errors="coerce")
test["date"] = pd.to_datetime(test["date"], errors="coerce")

# 3. 병합 및 필터링
full = pd.concat([train, test], axis=0).dropna(subset=["date"])
target_dates = ["2023-08-31", "2023-09-30", "2023-10-31"]
filtered = full[full["date"].isin(pd.to_datetime(target_dates))]

# 4. 필요한 컬럼만 선택
filtered = filtered[["date", "Natural_Gas_US_Henry_Hub_Gas"]].sort_values("date").reset_index(drop=True)

# 5. 저장
filtered.to_csv("real_gas_prices_2023_aug_sep_oct.csv", index=False)
print("✅ 저장 완료: real_gas_prices_2023_aug_sep_oct.csv")
print(filtered)
