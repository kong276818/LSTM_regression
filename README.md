# LSTM_regression
LSTM_regression

### 주요 코드 파일
| 파일명 | 설명 |
|--------|------|
| `test_LSTM.py` | 일 단위 예측 및 월 평균 예측 수행 (LSTM 기반) |
| `Test_Data_Merge.py` | 별도의 예측 파이프라인: 월 단위 직접 예측 수행 |
| `MAPE.py` | 예측값과 실제값을 기반으로 MAPE(평균 절대 백분율 오차) 계산 |
| `plt.py` | 예측값 vs 실제값 시각화 그래프 생성 |
| `ydata.py` | 2023년 실제 데이터 필터링 및 저장 |

---

## 🧠 모델 개요

- **모델 구조**: 단일 LSTM 레이어 + Dense 출력층
- **입력 특징 (Features)**:
  - 날짜 정보 (month, year)
  - 과거 1~3일의 시차 변수 (lag_1~lag_3)
  - 이동평균 (rolling_3, rolling_6)
  - 유가 관련 외부 변수들의 lag
- **정규화**: `MinMaxScaler` 적용
- **예측 대상**: 

---

## 📁 실행 순서

1. `ydata.py` 실행 → 실제값 파일 생성 (`...csv`)
2. `test_LSTM.py` 실행 → LSTM 모델 학습 및 예측값 저장 (`...csv`)
3. `MAPE.py` 실행 → 예측값 vs 실제값 기반 MAPE 계산
4. `plt.py` 실행 → 예측 결과 시각화

---

## 📚 참고 기술

- TensorFlow Keras (`LSTM`, `Sequential`)
- 시계열 데이터 전처리 및 Feature Engineering
- MAPE (Mean Absolute Percentage Error)
- Matplotlib 시각화
