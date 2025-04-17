import streamlit as st
import pandas as pd
from fetch_tickers import get_all_tickers
from fetch_data import fetch_data
from train_predict import train_and_predict
import os

st.set_page_config(page_title="임주혁의 글로벌 주식 예측 AI", layout="centered")

st.title("📈 임주혁의 글로벌 주식 예측 대시보드")
st.caption("S&P500 + KOSPI200 종목 예측 | LSTM 기반 | by 임주혁")

# 종목 불러오기
with st.spinner("📡 종목 목록 불러오는 중..."):
    tickers = get_all_tickers()
    tickers = sorted(tickers)

ticker = st.selectbox("📌 예측할 종목을 선택하세요:", tickers)

if st.button("예측 시작 🚀"):
    with st.spinner("📥 데이터 다운로드 중..."):
        success = fetch_data(ticker)
    
    if success:
        data_path = f"data/{ticker}.csv"
        st.success("✅ 데이터 다운로드 완료")

        with st.spinner("🤖 AI 예측 중..."):
            predicted_price = train_and_predict(ticker, data_path)

        if predicted_price:
            st.success(f"📈 {ticker} 예측 종가: **{predicted_price:.2f}**")
            
            df = pd.read_csv(data_path)
            st.line_chart(df[['Close']].tail(200))
        else:
            st.error("❌ 예측에 실패했습니다.")
    else:
        st.error("❌ 데이터 가져오기에 실패했습니다.")

# Footer
st.markdown("---")
st.markdown("© 2025 임주혁. All rights reserved.")
