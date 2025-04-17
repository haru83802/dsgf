from fetch_tickers import get_all_tickers
from fetch_data import fetch_data
from train_predict import train_and_predict
import os
from datetime import datetime

RESULT_FILE = f"logs/prediction_{datetime.now().strftime('%Y%m%d')}.csv"

def main():
    tickers = get_all_tickers()
    os.makedirs("logs", exist_ok=True)
    results = []

    for ticker in tickers[:30]:  # 처음엔 30개만 테스트용
        print(f"▶ {ticker} 처리 중...")
        if fetch_data(ticker):
            predicted = train_and_predict(ticker, f"data/{ticker}.csv")
            if predicted:
                print(f"  ➤ 예측가: {predicted:.2f}")
                results.append([ticker, predicted])
    
    import pandas as pd
    df = pd.DataFrame(results, columns=["Ticker", "Predicted_Close"])
    df.to_csv(RESULT_FILE, index=False)
    print(f"\n✅ 모든 결과 저장 완료: {RESULT_FILE}")

if __name__ == "__main__":
    main()
