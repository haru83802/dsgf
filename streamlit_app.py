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

        # 데이터 유효성 검사
        df = pd.read_csv(data_path)
        if df.empty or 'Close' not in df.columns:
            st.error("❌ 다운로드된 데이터에 오류가 있습니다. 데이터를 다시 확인해 주세요.")
        else:
            with st.spinner("🤖 AI 예측 중..."):
                try:
                    predicted_price, accuracy = train_and_predict(ticker, data_path)
                    if predicted_price:
                        st.success(f"📈 {ticker} 예측 종가: **{predicted_price:.2f}**")
                        st.write(f"모델 정확도: **{accuracy:.2f}**")
                        st.line_chart(df[['Close']].tail(200))
                    else:
                        st.error("❌ 예측에 실패했습니다. 모델 예측 오류.")
                except Exception as e:
                    st.error(f"❌ 예측 중 오류 발생: {str(e)}")
    else:
        st.error("❌ 데이터 가져오기에 실패했습니다.")

# Footer
st.markdown("---")
st.markdown("© 2025 임주혁. All rights reserved.")
st.markdown("문의: haru0904@gmail.com")
