import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# 페이지 설정
st.set_page_config(page_title="임주혁의 글로벌 주식 예측 AI", layout="centered")

# 앱 제목과 캡션
st.title("📈 임주혁의 글로벌 주식 예측 대시보드")
st.caption("S&P500 + KOSPI200 종목 예측 | LSTM 기반 | by 임주혁")

# 종목 목록 가져오는 함수 (자동으로 종목 목록을 불러옴)
def get_all_tickers():
    tickers = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"]  # 예시 종목들
    return tickers

# 데이터 다운로드 함수 (Yahoo Finance를 이용)
def fetch_data(ticker, start_date="2010-01-01", end_date="2025-01-01"):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.to_csv(f"data/{ticker}.csv")
        return True
    except Exception as e:
        return False

# 데이터 전처리 함수
def preprocess_data(stock_data):
    df = stock_data[['Close']].dropna()
    
    # 'Close' 열이 문자열인 경우를 처리 (예를 들어 'AAP' 같은 값이 있을 경우)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()  # Non-numeric or missing values are dropped
    
    # 데이터를 0과 1 사이로 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    return scaled_data, scaler

# 시퀀스 생성 함수 (LSTM 모델에 맞게)
def create_dataset(data, time_step=60):
    x_data, y_data = [], []
    for i in range(len(data) - time_step - 1):
        x_data.append(data[i:i + time_step, 0])
        y_data.append(data[i + time_step, 0])
    return np.array(x_data), np.array(y_data)

# LSTM 모델 정의
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # 종가 예측
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# 모델 훈련 함수
def train_model(x_train, y_train):
    model = build_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    return model

# 예측 함수
def predict_stock_price(model, x_test, scaler):
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)  # 예측 결과를 원래 값으로 복원
    return predicted_price

# 예측 및 훈련 전체 프로세스
def train_and_predict(ticker, data_path):
    stock_data = pd.read_csv(data_path)
    
    # 데이터 전처리 및 확인
    scaled_data, scaler = preprocess_data(stock_data)

    time_step = 60
    x_data, y_data = create_dataset(scaled_data, time_step)

    train_size = int(len(x_data) * 0.8)
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]

    model = train_model(x_train, y_train)
    predicted_price = predict_stock_price(model, x_test, scaler)

    # 예측된 가격과 실제값을 데이터프레임에 추가
    predicted_price = predicted_price.flatten()
    df = stock_data.iloc[train_size + time_step:][['Close']].copy()
    df['Predicted'] = predicted_price

    # 정확도 계산 (예측값과 실제값 비교)
    accuracy = np.mean(np.abs(predicted_price - y_test[-len(predicted_price):]) / y_test[-len(predicted_price):]) * 100
    return df, accuracy

# 종목 불러오기
with st.spinner("📡 종목 목록 불러오는 중..."):
    tickers = get_all_tickers()
    tickers = sorted(tickers)

# 종목 선택 UI
ticker = st.selectbox("📌 예측할 종목을 선택하세요:", tickers)

if st.button("예측 시작 🚀"):
    with st.spinner("📥 데이터 다운로드 중..."):
        success = fetch_data(ticker)
    
    if success:
        data_path = f"data/{ticker}.csv"
        st.success("✅ 데이터 다운로드 완료")

        # 데이터 유효성 검사
        df = pd.read_csv(data_path)
        if df.empty or 'Close' not in df.columns:
            st.error("❌ 다운로드된 데이터에 오류가 있습니다. 데이터를 다시 확인해 주세요.")
        else:
            with st.spinner("🤖 AI 예측 중..."):
                try:
                    df_predicted, accuracy = train_and_predict(ticker, data_path)
                    if not df_predicted.empty:
                        st.success(f"📈 {ticker} 예측 종가: **{df_predicted['Predicted'].iloc[-1]:.2f}**")
                        st.write(f"모델 정확도: **{accuracy:.2f}%**")
                        
                        # 예측 결과와 실제 데이터를 표로 보여줌
                        st.subheader(f"{ticker} 예측 결과")
                        st.dataframe(df_predicted.tail(100))  # 예측된 종가와 실제 종가를 비교

                        # 종가 차트
                        st.line_chart(df_predicted[['Close', 'Predicted']].tail(200))
                    else:
                        st.error("❌ 예측에 실패했습니다. 모델 예측 오류.")
                except Exception as e:
                    st.error(f"❌ 예측 중 오류 발생: {str(e)}")
    else:
        st.error("❌ 데이터 가져오기에 실패했습니다.")

# Footer
st.markdown("---")
st.markdown("© 2025 임주혁. All rights reserved.")
st.markdown("문의: haru090400@gmail.com")
