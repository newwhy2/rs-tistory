import os
import json
import datetime as dt
from io import StringIO

import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

load_dotenv()

# ====== Blogger 설정 ======
BLOGGER_AUTO_POST = os.getenv("BLOGGER_AUTO_POST", "false").lower() == "true"
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN", "")
BLOGGER_BLOG_ID = os.getenv("BLOGGER_BLOG_ID", "")

# ====== RS 파라미터 ======
# 252 거래일(약 1년)까지 써야 해서 넉넉하게 260일로 설정
LOOKBACK_DAYS = 260
TOP_N = 30


def yahoo_ticker_from_wiki(ticker: str) -> str:
    """
    위키피디아 티커를 yfinance용으로 변환.
    BRK.B -> BRK-B, BF.B -> BF-B 등
    기본적으로 '.' 를 '-' 로 치환.
    """
    special_map = {
        "BRK.B": "BRK-B",
        "BF.B": "BF-B",
    }
    if ticker in special_map:
        return special_map[ticker]
    return ticker.replace(".", "-")


def get_company_meta(tickers: list[str]) -> dict[str, dict[str, str]]:
    """yfinance를 이용해 티커별 회사 이름/섹터 정보를 가져온다."""
    meta: dict[str, dict[str, str]] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
            name = info.get("shortName") or info.get("longName") or t
            sector = info.get("sector") or "Unknown"
        except Exception:
            name = t
            sector = "Unknown"
        meta[t] = {"name": name, "sector": sector}
    return meta


def load_sp500_universe() -> list[str]:
    """S&P 500 전체 구성 종목 티커를 위키피디아에서 가져온다."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].astype(str).tolist()
    tickers = [yahoo_ticker_from_wiki(t) for t in tickers]
    return sorted(set(tickers))


def load_nasdaq100_universe() -> list[str]:
    """NASDAQ 100 전체 구성 종목 티커를 위키피디아에서 가져온다."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))

    for df in tables:
        cols_lower = [str(c).lower() for c in df.columns]
        if any("ticker" in c or "symbol" in c for c in cols_lower):
            for col in df.columns:
                if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                    tickers = df[col].astype(str).tolist()
                    tickers = [yahoo_ticker_from_wiki(t) for t in tickers]
                    return sorted(set(tickers))

    raise RuntimeError("NASDAQ 100 티커 테이블을 찾을 수 없습니다.")


def download_price_data(tickers, lookback_days):
    """종목들의 종가 데이터 다운로드."""
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days * 2)

    # auto_adjust=True 로 배당/액분 등을 조정한 종가(Close)만 사용
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if raw.empty:
        raise RuntimeError("yfinance에서 가격 데이터를 가져오지 못했습니다.")

    # 여러 티커를 받으면 MultiIndex 컬럼(예: ('Close', 'AAPL'))이 올 수 있음
    if isinstance(raw.columns, pd.MultiIndex):
        # 첫 번째 레벨(가격 종류)에서 'Close' 선택
        if "Close" in raw.columns.get_level_values(0):
            df = raw.xs("Close", axis=1, level=0)
        else:
            # 혹시 다른 이름인 경우 대비: 첫 레벨의 첫 항목 사용
            first_level = raw.columns.levels[0][0]
            df = raw.xs(first_level, axis=1, level=0)
    else:
        # 단일 티커일 때는 그냥 'Close' 컬럼 사용
        if "Close" in raw.columns:
            df = raw["Close"].to_frame()
        else:
            # 마지막 수단: 전체를 그대로 사용
            df = raw

    df = df.dropna(how="all").tail(lookback_days + 1)  # 가장 최근 포함한 과거 데이터
    return df


def compute_rs_ibd_style(df_prices: pd.DataFrame) -> pd.DataFrame:
    """IBD 스타일(가중치)로 RS 점수 계산.

    IBD style Relative Strength (대략적 표현):
        RS_raw = 2 * (C0 / C63) + (C0 / C126) + (C0 / C189) + (C0 / C252)
    그런 다음, 전체 종목 중에서 백분위수로 1~99 등급화.

    신규상장(IPO) 종목 처리 (IBD 방식):
        - 거래일 5일 미만: RS = 1
        - 거래일 부족 시: 가용 데이터로 외삽
    """
    if len(df_prices) < 253:
        raise RuntimeError("IBD 스타일 계산을 위해 최소 252거래일 이상의 데이터가 필요합니다.")

    latest = df_prices.iloc[-1]
    eps = 1e-9

    # 각 종목별 실제 거래일 수 (NaN이 아닌 일수)
    trading_days = df_prices.notna().sum()

    # 기간별 기준 가격 (해당 시점에 데이터가 없으면 NaN)
    c21 = df_prices.iloc[-min(21, len(df_prices))]
    c63 = df_prices.iloc[-min(63, len(df_prices))]
    c126 = df_prices.iloc[-min(126, len(df_prices))]
    c189 = df_prices.iloc[-min(189, len(df_prices))]
    c252 = df_prices.iloc[-min(252, len(df_prices))]

    # 각 종목별 가장 오래된 유효 가격 (첫 거래일 가격)
    first_valid_price = pd.Series(index=df_prices.columns, dtype=float)
    for col in df_prices.columns:
        valid = df_prices[col].dropna()
        if len(valid) > 0:
            first_valid_price[col] = valid.iloc[0]
        else:
            first_valid_price[col] = float("nan")

    # --- 기간별 수익률 계산 (NaN-safe) ---
    def safe_return(current, past):
        """NaN이면 NaN 유지, 아니면 수익률 계산."""
        return current / (past + eps) - 1.0

    ret_1m = safe_return(latest, c21)
    ret_3m = safe_return(latest, c63)
    ret_12m = safe_return(latest, c252)

    # 신규상장 종목: 해당 기간의 데이터가 없으면(NaN) 가용 데이터로 대체
    for col in df_prices.columns:
        td = int(trading_days[col])
        if td < 5:
            # 5거래일 미만: 수익률 0으로 설정 (나중에 RS=1 처리)
            ret_1m[col] = 0.0
            ret_3m[col] = 0.0
            ret_12m[col] = 0.0
            continue

        # 실제 첫 유효 가격으로 가용 수익률 계산
        valid_data = df_prices[col].dropna()
        if len(valid_data) < 2:
            continue

        actual_first_price = valid_data.iloc[0]
        actual_days = len(valid_data)
        total_return = latest[col] / (actual_first_price + eps) - 1.0

        # 1개월 수익률: 21거래일 미만이면 가용 데이터 사용
        if pd.isna(c21[col]) or td < 21:
            ret_1m[col] = total_return

        # 3개월 수익률: 63거래일 미만이면 외삽
        if pd.isna(c63[col]) or td < 63:
            if actual_days > 1 and total_return > -1:
                # 연율화 후 3개월치로 환산
                annualized = (1 + total_return) ** (252.0 / actual_days) - 1.0
                ret_3m[col] = (1 + annualized) ** (63.0 / 252.0) - 1.0
            else:
                ret_3m[col] = total_return

        # 12개월 수익률: 252거래일 미만이면 외삽
        if pd.isna(c252[col]) or td < 252:
            if actual_days > 1 and total_return > -1:
                annualized = (1 + total_return) ** (252.0 / actual_days) - 1.0
                ret_12m[col] = annualized
            else:
                ret_12m[col] = total_return

    # --- IBD RS Raw 계산 ---
    # 각 구간별 비율 (가용 데이터 사용)
    ratio_q1 = 1 + ret_3m.fillna(0)   # 최근 분기 (2배 가중)
    ratio_q2 = pd.Series(index=df_prices.columns, dtype=float)
    ratio_q3 = pd.Series(index=df_prices.columns, dtype=float)
    ratio_q4 = pd.Series(index=df_prices.columns, dtype=float)

    for col in df_prices.columns:
        td = int(trading_days[col])
        if td < 5:
            ratio_q2[col] = 1.0
            ratio_q3[col] = 1.0
            ratio_q4[col] = 1.0
            continue

        # Q2: 126일 전 대비 (6개월)
        if not pd.isna(c126[col]) and c126[col] > eps:
            ratio_q2[col] = latest[col] / c126[col]
        else:
            ratio_q2[col] = ratio_q1[col]  # 외삽

        # Q3: 189일 전 대비 (9개월)
        if not pd.isna(c189[col]) and c189[col] > eps:
            ratio_q3[col] = latest[col] / c189[col]
        else:
            ratio_q3[col] = ratio_q2[col]  # 외삽

        # Q4: 252일 전 대비 (12개월)
        if not pd.isna(c252[col]) and c252[col] > eps:
            ratio_q4[col] = latest[col] / c252[col]
        else:
            ratio_q4[col] = ratio_q3[col]  # 외삽

    rs_raw = 2 * ratio_q1 + ratio_q2 + ratio_q3 + ratio_q4

    # --- 기간별 RS (백분위 순위, 1~99) ---
    rs_1m = (ret_1m.rank(pct=True) * 98 + 1).round(0)
    rs_3m = (ret_3m.rank(pct=True) * 98 + 1).round(0)
    rs_12m = (ret_12m.rank(pct=True) * 98 + 1).round(0)

    # --- 최종 RS 등급 (IBD 종합) ---
    rs_rating = (rs_raw.rank(pct=True) * 98 + 1).round(0)

    # 5거래일 미만 IPO 종목: RS=1 고정
    ipo_mask = trading_days < 5
    rs_rating[ipo_mask] = 1
    rs_1m[ipo_mask] = 1
    rs_3m[ipo_mask] = 1
    rs_12m[ipo_mask] = 1

    tickers = list(df_prices.columns)
    meta_map = get_company_meta(tickers)

    result = pd.DataFrame(
        {
            "Ticker": tickers,
            "Name": [meta_map.get(t, {}).get("name", t) for t in tickers],
            "Sector": [meta_map.get(t, {}).get("sector", "Unknown") for t in tickers],
            "RS_Raw": rs_raw.values,
            "RS_Rating": rs_rating.values,
            "Return_1M": ret_1m.values,
            "Return_3M": ret_3m.values,
            "Return_12M": ret_12m.values,
            "RS_1M": rs_1m.values,
            "RS_3M": rs_3m.values,
            "RS_12M": rs_12m.values,
            "Trading_Days": trading_days.values,
        }
    )

    result = result.sort_values("RS_Rating", ascending=False).reset_index(drop=True)
    return result


def format_percent(x):
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


def build_post_content(date_str, index_name, rs_df):
    """블로그(구글 Blogger)에 붙여넣을 한국어 HTML 본문 생성."""
    top_df = rs_df.head(TOP_N).copy()

    # --- 전체 종목 기준 섹터 순위 계산 ---
    all_sector_df = (
        rs_df.groupby("Sector")
        .agg(섹터평균_RS=("RS_Rating", "mean"))
        .reset_index()
    )
    all_sector_df = all_sector_df.sort_values("섹터평균_RS", ascending=False).reset_index(drop=True)
    all_sector_df["순위"] = range(1, len(all_sector_df) + 1)
    sector_rank_map = dict(zip(all_sector_df["Sector"], all_sector_df["순위"]))

    # 섹터명에 순위 표시
    top_df["티커"] = top_df["Ticker"]
    top_df["회사명"] = top_df["Name"]
    top_df["섹터"] = top_df["Sector"].apply(
        lambda s: f"{s}({sector_rank_map.get(s, '?')}순위섹터)"
    )
    top_df["1개월 수익률"] = top_df["Return_1M"].apply(format_percent)
    top_df["3개월 수익률"] = top_df["Return_3M"].apply(format_percent)
    top_df["12개월 수익률"] = top_df["Return_12M"].apply(format_percent)
    top_df["1개월 RS"] = top_df["RS_1M"].astype(int)
    top_df["3개월 RS"] = top_df["RS_3M"].astype(int)
    top_df["12개월 RS"] = top_df["RS_12M"].astype(int)
    top_df["RS 등급(1~99)"] = top_df["RS_Rating"].astype(int)

    display_df = top_df[[
        "티커", "회사명", "섹터",
        "1개월 수익률", "3개월 수익률", "12개월 수익률",
        "1개월 RS", "3개월 RS", "12개월 RS",
        "RS 등급(1~99)",
    ]]

    table_html = display_df.to_html(
        index=False,
        escape=False,
        border=1,
        justify="center"
    )

    # 섹터별 강도 요약 (전체 종목 기준, 순위 포함)
    sector_summary = (
        rs_df.groupby("Sector")
        .agg(
            섹터평균_RS=("RS_Rating", "mean"),
            종목수=("Ticker", "count"),
        )
        .reset_index()
    )
    sector_summary["섹터평균_RS"] = sector_summary["섹터평균_RS"].round(1)
    sector_summary = sector_summary.sort_values("섹터평균_RS", ascending=False).reset_index(drop=True)
    # 섹터명에 순위 표시
    sector_summary["섹터"] = sector_summary.apply(
        lambda row: f"{row['Sector']}({sector_rank_map.get(row['Sector'], '?')}순위섹터)",
        axis=1,
    )
    sector_display = sector_summary[["섹터", "섹터평균_RS", "종목수"]]
    sector_table_html = sector_display.to_html(
        index=False,
        escape=False,
        border=1,
        justify="center",
    )

    intro_html = f"""
<p><strong>{date_str} {index_name} 상대강도 리포트</strong></p>
<p>본 포스트는 최근 약 1년(최대 252 거래일) 동안의 주가 흐름을 바탕으로,
Investor's Business Daily(IBD) 스타일의 상대강도(Relative Strength, RS) 점수를
참고하여 계산한 결과입니다.</p>
<p>RS 계산식은 대략적으로 다음과 같은 가중치를 사용합니다.<br/>
RS ≈ 2 × (현재가 / 63거래일 전 종가) + (현재가 / 126거래일 전 종가)
+ (현재가 / 189거래일 전 종가) + (현재가 / 252거래일 전 종가)</p>
<p>신규상장(IPO) 종목의 경우, 거래일 5일 미만은 RS=1로 고정하고,
데이터가 부족한 기간은 가용 데이터를 기반으로 외삽하여 계산합니다.</p>
<p>이 값을 기준으로 전체 종목을 백분위 순위로 정렬해 1~99 등급 형태의
RS 등급을 만든 뒤, 상위 {TOP_N}개 종목을 추려서 정리했습니다.</p>
<p>아래 첫 번째 표는 상위 종목 리스트이고, 두 번째 표는 전체 종목 기준으로
섹터별 평균 RS 등급과 종목 수를 정리한 것입니다.</p>
"""

    disclaimer_html = """
<p><em>※ 본 글은 특정 종목의 매수/매도 추천이 아니며, 정보 제공만을 목적으로 합니다.
투자 판단의 최종 책임은 투자자 본인에게 있습니다.</em></p>
"""

    content_html = intro_html + table_html + "<br/><br/>" + sector_table_html + disclaimer_html
    return content_html


def save_post_html(output_dir, date_str, index_name, title, content_html):
    """HTML 파일로 저장해서 Blogger에 수동으로 붙여넣기 쉽게 만든다."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = index_name.lower().replace(" ", "_").replace("&", "and")
    filename = f"{date_str}_{safe_name}.html"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!-- TITLE: {title} -->\n")
        f.write(content_html)

    print(f"{index_name} 리포트 HTML 저장됨: {path}")
    return path


def get_blogger_access_token():
    """OAuth2 refresh token으로 새 access token을 발급받는다."""
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": GOOGLE_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def post_to_blogger(title, content_html):
    """Google Blogger API v3로 글을 발행한다."""
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN, BLOGGER_BLOG_ID]):
        raise RuntimeError(
            "Blogger 포스팅에 필요한 환경변수가 설정되지 않았습니다.\n"
            "필요: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN, BLOGGER_BLOG_ID"
        )

    access_token = get_blogger_access_token()

    url = f"https://www.googleapis.com/blogger/v3/blogs/{BLOGGER_BLOG_ID}/posts/"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "kind": "blogger#post",
        "blog": {"id": BLOGGER_BLOG_ID},
        "title": title,
        "content": content_html,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()

    post_data = resp.json()
    post_url = post_data.get("url", "(URL 알 수 없음)")
    print(f"Blogger 포스팅 성공: {post_url}")
    return post_url


def run_universe(index_name, loader_func, output_dir):
    """특정 인덱스(NASDAQ100, S&P 500)에 대해 RS 계산 및 HTML 저장."""
    today = dt.date.today()
    date_str = today.strftime("%Y-%m-%d")

    print(f"[{index_name}] 종목 리스트 불러오는 중...")
    tickers = loader_func()
    print(f"[{index_name}] 종목 수: {len(tickers)}")

    print(f"[{index_name}] 가격 데이터 다운로드 중...")
    df_prices = download_price_data(tickers, LOOKBACK_DAYS)

    print(f"[{index_name}] RS 계산(IBD 스타일) 중...")
    rs_df = compute_rs_ibd_style(df_prices)

    print(f"[{index_name}] 본문 생성 중...")
    content_html = build_post_content(
        date_str=date_str,
        index_name=index_name,
        rs_df=rs_df,
    )

    title = f"{date_str} {index_name} 상대강도 TOP {TOP_N}"

    print(f"[{index_name}] HTML 파일로 저장 중...")
    path = save_post_html(output_dir, date_str, index_name, title, content_html)

    return title, path, content_html


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "output")

    # NASDAQ 100
    nasdaq_title, nasdaq_path, nasdaq_html = run_universe(
        index_name="NASDAQ 100",
        loader_func=load_nasdaq100_universe,
        output_dir=output_dir,
    )

    # S&P 500
    sp500_title, sp500_path, sp500_html = run_universe(
        index_name="S&P 500",
        loader_func=load_sp500_universe,
        output_dir=output_dir,
    )

    print("\n===== 리포트 생성 완료 =====")
    print(f"1) {nasdaq_title} → {nasdaq_path}")
    print(f"2) {sp500_title} → {sp500_path}")

    # Blogger 자동 포스팅
    if BLOGGER_AUTO_POST:
        print("\n===== Blogger 자동 포스팅 시작 =====")
        today_str = dt.date.today().strftime("%Y-%m-%d")

        # NASDAQ 100 포스팅
        try:
            url1 = post_to_blogger(nasdaq_title, nasdaq_html)
            print(f"NASDAQ 100 포스팅 완료: {url1}")
        except Exception as e:
            print(f"NASDAQ 100 포스팅 실패: {e}")

        # S&P 500 포스팅
        try:
            url2 = post_to_blogger(sp500_title, sp500_html)
            print(f"S&P 500 포스팅 완료: {url2}")
        except Exception as e:
            print(f"S&P 500 포스팅 실패: {e}")
    else:
        print("\nBlogger 자동 포스팅이 비활성화되어 있습니다.")
        print("자동 포스팅을 원하면 환경변수 BLOGGER_AUTO_POST=true 로 설정하세요.")


if __name__ == "__main__":
    main()

