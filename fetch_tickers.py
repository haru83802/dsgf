import pandas as pd

# 미국 S&P500 종목
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table['Symbol'].tolist()

# 한국 코스피 종목
def get_kospi_tickers():
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
    df = pd.read_html(url, header=0, encoding='euc-kr')[0]
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    return (df['종목코드'] + ".KS").tolist()

# 통합
def get_all_tickers():
    return get_sp500_tickers() + get_kospi_tickers()

if __name__ == "__main__":
    tickers = get_all_tickers()
    print(f"{len(tickers)}개 종목 수집 완료")
