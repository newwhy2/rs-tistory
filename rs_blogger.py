import os
import json
import datetime as dt
import argparse
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
TOP_N = 50
QUALITY_TOP_N = 30
OUTPUT_DIR_NAME = "output"
DATA_DIR_NAME = "data"

THEME_ETF_MAP = {
    "Semiconductors": "SMH",
    "Software": "IGV",
    "Cybersecurity": "HACK",
    "Biotechnology": "XBI",
    "Gold Miners": "GDX",
    "Silver Miners": "SIL",
    "Oil & Gas": "XLE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Aerospace & Defense": "ITA",
    "Home Construction": "XHB",
    "Retail": "XRT",
    "Banks": "KRE",
    "Real Estate": "XLRE",
    "Transportation": "IYT",
}


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


def detect_setup(df_prices: pd.DataFrame, ticker: str) -> str:
    """가격 데이터로 기술적 셋업 상태를 감지한다.

    감지 패턴:
    - VCP: 최근 4주간 주간 변동폭(고가-저가 비율)이 점점 줄어듦
    - 타이트: 최근 5일 고저 범위 < 5%
    - 돌파: 오늘 종가 > 최근 20일 최고가
    - 52주高: 현재가가 52주 고가의 95% 이상
    """
    col = df_prices[ticker].dropna()
    if len(col) < 30:
        return ""

    latest_price = col.iloc[-1]
    setups = []

    # --- 52주 신고가 부근 ---
    high_52w = col.tail(252).max()
    if latest_price >= high_52w * 0.95:
        setups.append("52주高")

    # --- 돌파: 종가 > 최근 20일 최고가 (어제까지의 고가 기준) ---
    prev_20d_high = col.iloc[-21:-1].max() if len(col) > 21 else col.iloc[:-1].max()
    if latest_price > prev_20d_high:
        setups.append("돌파")

    # --- 타이트 (최근 5일 고저 범위 < 5%) ---
    recent_5d = col.tail(5)
    range_pct = (recent_5d.max() - recent_5d.min()) / (recent_5d.min() + 1e-9)
    if range_pct < 0.05:
        setups.append("타이트")

    # --- VCP (Volatility Contraction Pattern) ---
    # 최근 4주의 주간 변동폭이 점점 줄어들면 VCP
    if len(col) >= 20:
        weekly_ranges = []
        for i in range(4):
            start = -(i + 1) * 5
            end = -i * 5 if i > 0 else None
            week_data = col.iloc[start:end] if end else col.iloc[start:]
            if len(week_data) > 0:
                w_range = (week_data.max() - week_data.min()) / (week_data.min() + 1e-9)
                weekly_ranges.append(w_range)
        weekly_ranges.reverse()  # 오래된 주 → 최근 주 순서

        if len(weekly_ranges) >= 3:
            contractions = sum(
                1 for i in range(1, len(weekly_ranges))
                if weekly_ranges[i] < weekly_ranges[i - 1]
            )
            if contractions >= 2:
                setups.append("VCP")

    return ",".join(setups) if setups else "-"


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

    # --- D-1 RS 계산 (전일 대비 변동용) ---
    if len(df_prices) >= 254:
        df_prev = df_prices.iloc[:-1]  # 어제까지의 데이터
        prev_latest = df_prev.iloc[-1]
        prev_c63 = df_prev.iloc[-min(63, len(df_prev))]
        prev_c126 = df_prev.iloc[-min(126, len(df_prev))]
        prev_c189 = df_prev.iloc[-min(189, len(df_prev))]
        prev_c252 = df_prev.iloc[-min(252, len(df_prev))]

        prev_ratio_q1 = prev_latest / (prev_c63 + eps)
        prev_ratio_q2 = prev_latest / (prev_c126 + eps)
        prev_ratio_q3 = prev_latest / (prev_c189 + eps)
        prev_ratio_q4 = prev_latest / (prev_c252 + eps)
        prev_rs_raw = 2 * prev_ratio_q1 + prev_ratio_q2 + prev_ratio_q3 + prev_ratio_q4
        rs_prev = (prev_rs_raw.rank(pct=True) * 98 + 1).round(0)
        rs_prev[ipo_mask] = 1
    else:
        rs_prev = rs_rating.copy()  # 데이터 부족 시 변동 없음

    rs_change = rs_rating - rs_prev

    tickers = list(df_prices.columns)
    meta_map = get_company_meta(tickers)

    result = pd.DataFrame(
        {
            "Ticker": tickers,
            "Name": [meta_map.get(t, {}).get("name", t) for t in tickers],
            "Sector": [meta_map.get(t, {}).get("sector", "Unknown") for t in tickers],
            "RS_Raw": rs_raw.values,
            "RS_Rating": rs_rating.values,
            "RS_Change": rs_change.values,
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


def build_post_content(date_str, index_name, rs_df, df_prices=None):
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
        lambda s: f"{s}({sector_rank_map.get(s, '?')}위섹터)"
    )
    top_df["1개월 수익률"] = top_df["Return_1M"].apply(format_percent)
    top_df["3개월 수익률"] = top_df["Return_3M"].apply(format_percent)
    top_df["12개월 수익률"] = top_df["Return_12M"].apply(format_percent)
    top_df["1개월 RS"] = top_df["RS_1M"].astype(int)
    top_df["3개월 RS"] = top_df["RS_3M"].astype(int)
    top_df["12개월 RS"] = top_df["RS_12M"].astype(int)
    top_df["RS 등급"] = top_df["RS_Rating"].astype(int)

    # RS 변동 (전일 대비)
    def format_rs_change(val):
        v = int(val)
        if v > 0:
            return f"<span style='color:red'>+{v}</span>"
        elif v < 0:
            return f"<span style='color:blue'>{v}</span>"
        return "0"
    top_df["RS변동"] = top_df["RS_Change"].apply(format_rs_change)

    # 기술적 셋업 감지
    if df_prices is not None:
        top_df["셋업"] = top_df["Ticker"].apply(lambda t: detect_setup(df_prices, t))
    else:
        top_df["셋업"] = "-"

    display_df = top_df[[
        "티커", "회사명", "섹터",
        "RS 등급", "RS변동", "셋업",
        "1개월 수익률", "3개월 수익률", "12개월 수익률",
        "1개월 RS", "3개월 RS", "12개월 RS",
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
        lambda row: f"{row['Sector']}({sector_rank_map.get(row['Sector'], '?')}위섹터)",
        axis=1,
    )
    sector_display = sector_summary[["섹터", "섹터평균_RS", "종목수"]]
    sector_table_html = sector_display.to_html(
        index=False,
        escape=False,
        border=1,
        justify="center",
    )

    # CSS: 테이블 스타일 (티커 줄바꿈 방지, 모바일 스크롤)
    style_html = """
<style>
.rs-table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.rs-table-wrap table { border-collapse: collapse; width: 100%; font-size: 14px; }
.rs-table-wrap th, .rs-table-wrap td { padding: 6px 8px; white-space: nowrap; }
.rs-table-wrap th { background: #f0f0f0; }
</style>
"""

    intro_html = f"""
<p><strong>{date_str} {index_name} 상대강도 리포트</strong></p>
<p>RS 계산식 (IBD 스타일 가중치):<br/>
RS ≈ 2 × (현재가/63일전) + (현재가/126일전) + (현재가/189일전) + (현재가/252일전)</p>
<p>셋업 범례: <b>VCP</b>=변동성 수축, <b>타이트</b>=5일 범위&lt;5%, <b>돌파</b>=20일 고가 돌파, <b>52주高</b>=52주 고가 부근</p>
"""

    disclaimer_html = """
<p><em>※ 본 글은 특정 종목의 매수/매도 추천이 아니며, 정보 제공만을 목적으로 합니다.
투자 판단의 최종 책임은 투자자 본인에게 있습니다.</em></p>
"""

    wrapped_table = f'<div class="rs-table-wrap">{table_html}</div>'
    wrapped_sector = f'<div class="rs-table-wrap">{sector_table_html}</div>'

    content_html = style_html + intro_html + wrapped_table + "<br/><br/>" + wrapped_sector + disclaimer_html
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

def get_blogger_access_token():
    """OAuth2 refresh token으로 새 access token을 발급받는다."""
    client_id = GOOGLE_CLIENT_ID.strip()
    client_secret = GOOGLE_CLIENT_SECRET.strip()
    refresh_token = GOOGLE_REFRESH_TOKEN.strip()

    # print(f"[DEBUG] client_id: {client_id[:20]}..." if client_id else "[DEBUG] client_id: EMPTY!")

    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"[ERROR] Token refresh failed: {resp.status_code}")
        print(f"[ERROR] Response: {resp.text}")
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

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        post_data = resp.json()
        post_url = post_data.get("url", "(URL 알 수 없음)")
        print(f"Blogger 포스팅 성공: {post_url}")
        return post_url
    except Exception as e:
        print(f"Blogger 포스팅 실패: {e}")
        return None


def build_market_breadth(df_prices: pd.DataFrame, index_name: str) -> str:
    """시장 브리핑용 데이터 생성 (등락, 신고가/신저가, MA 확산, 점수 기반 판정)."""
    if len(df_prices) < 252:
        return ""

    latest = df_prices.iloc[-1]
    prev = df_prices.iloc[-2]

    changes = latest - prev
    adv = int((changes > 0).sum())
    dec = int((changes < 0).sum())
    unch = int((changes == 0).sum())
    total = max(adv + dec + unch, 1)

    adv_pct = adv / total * 100
    dec_pct = dec / total * 100

    high_52w = df_prices.tail(252).max()
    low_52w = df_prices.tail(252).min()
    new_highs = int((latest >= high_52w * 0.99).sum())
    new_lows = int((latest <= low_52w * 1.01).sum())

    ma50 = df_prices.rolling(50).mean().iloc[-1]
    ma200 = df_prices.rolling(200).mean().iloc[-1]
    pct_above_50 = ((latest > ma50).sum() / total) * 100
    pct_above_200 = ((latest > ma200).sum() / total) * 100

    score = 0
    adv_dec_ratio = adv / max(dec, 1)
    nh_nl_ratio = new_highs / max(new_lows, 1)

    if adv_dec_ratio >= 1.3:
        score += 2
    elif adv_dec_ratio >= 1.1:
        score += 1
    elif adv_dec_ratio <= 0.77:
        score -= 2
    elif adv_dec_ratio <= 0.91:
        score -= 1

    if nh_nl_ratio >= 1.5:
        score += 2
    elif nh_nl_ratio >= 1.1:
        score += 1
    elif nh_nl_ratio <= 0.67:
        score -= 2
    elif nh_nl_ratio <= 0.91:
        score -= 1

    if pct_above_50 >= 60:
        score += 1
    elif pct_above_50 <= 40:
        score -= 1

    if pct_above_200 >= 55:
        score += 1
    elif pct_above_200 <= 45:
        score -= 1

    if score >= 4:
        sentiment = "강세 우위 (Bullish Bias)"
        sentiment_color = "#d32f2f"
    elif score <= -4:
        sentiment = "약세 우위 (Bearish Bias)"
        sentiment_color = "#1976d2"
    else:
        sentiment = "중립/혼조 (Neutral)"
        sentiment_color = "#757575"

    html = f"""
<div style="margin:10px 0;padding:15px;border:1px solid #e0e0e0;border-radius:10px;background-color:#f9f9f9;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <h3 style="margin:0;color:#333;">{index_name} Market Breadth</h3>
        <span style="font-size:13px;font-weight:bold;color:{sentiment_color};border:1px solid {sentiment_color};padding:2px 8px;border-radius:12px;">{sentiment}</span>
    </div>

    <div style="margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;font-size:13px;color:#555;margin-bottom:4px;">
            <span>상승 {adv} ({adv_pct:.1f}%)</span>
            <span>하락 {dec} ({dec_pct:.1f}%)</span>
        </div>
        <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;background:#eee;">
            <div style="width:{adv_pct}%;background:#ff5252;"></div>
            <div style="width:{dec_pct}%;background:#448aff;"></div>
        </div>
    </div>

    <div style="display:flex;justify-content:space-around;text-align:center;font-size:14px;">
        <div>
            <span style="display:block;color:#999;font-size:11px;">52주 신고가</span>
            <span style="font-weight:bold;color:#ff5252;">{new_highs}</span>
        </div>
        <div>
            <span style="display:block;color:#999;font-size:11px;">52주 신저가</span>
            <span style="font-weight:bold;color:#448aff;">{new_lows}</span>
        </div>
        <div>
            <span style="display:block;color:#999;font-size:11px;">50일선 상회</span>
            <span style="font-weight:bold;color:#333;">{pct_above_50:.1f}%</span>
        </div>
        <div>
            <span style="display:block;color:#999;font-size:11px;">200일선 상회</span>
            <span style="font-weight:bold;color:#333;">{pct_above_200:.1f}%</span>
        </div>
        <div>
            <span style="display:block;color:#999;font-size:11px;">Breadth Score</span>
            <span style="font-weight:bold;color:{sentiment_color};">{score:+d}</span>
        </div>
    </div>

    <div style="margin-top:12px;padding:10px;background:#fff;border-radius:8px;font-size:12px;color:#555;line-height:1.55;">
        <strong>판정 기준</strong><br/>
        1) Adv/Decl 비율, 2) 52주 High/Low 비율, 3) 50일선 상회 비율, 4) 200일선 상회 비율을 점수화합니다.<br/>
        총점 +4 이상: 강세 우위, -4 이하: 약세 우위, 그 사이는 중립/혼조로 분류합니다.
    </div>
</div>
"""
    return html


def _safe_period_return(series: pd.Series, bars: int) -> float:
    if len(series.dropna()) <= bars:
        return float("nan")
    last = series.iloc[-1]
    base = series.iloc[-(bars + 1)]
    if pd.isna(last) or pd.isna(base) or base == 0:
        return float("nan")
    return (last / base) - 1.0


def build_theme_tracker() -> str:
    """대표 테마 ETF의 기간별 수익률 + RS/RS변동 요약."""
    tickers = list(THEME_ETF_MAP.values())
    if not tickers:
        return ""

    try:
        raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    except Exception:
        return ""
    if raw.empty:
        return ""

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close_df = raw.xs("Close", axis=1, level=0)
        else:
            close_df = raw.xs(raw.columns.levels[0][0], axis=1, level=0)
    else:
        close_df = raw["Close"].to_frame() if "Close" in raw.columns else raw

    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame()

    # ETF 간 상대강도(RS): 3M 수익률 백분위(1~99)
    ret_3m_today = {}
    ret_3m_prev = {}
    for _, etf in THEME_ETF_MAP.items():
        if etf not in close_df.columns:
            continue
        col = close_df[etf].dropna()
        if len(col) >= 64:
            ret_3m_today[etf] = (col.iloc[-1] / col.iloc[-64]) - 1.0
        if len(col) >= 65:
            ret_3m_prev[etf] = (col.iloc[-2] / col.iloc[-65]) - 1.0

    rs_today = {}
    rs_prev = {}
    if ret_3m_today:
        s = pd.Series(ret_3m_today)
        rs_today = ((s.rank(pct=True) * 98) + 1).round(0).astype(int).to_dict()
    if ret_3m_prev:
        s = pd.Series(ret_3m_prev)
        rs_prev = ((s.rank(pct=True) * 98) + 1).round(0).astype(int).to_dict()

    rows = []
    for theme, etf in THEME_ETF_MAP.items():
        if etf not in close_df.columns:
            continue
        col = close_df[etf].dropna()
        if len(col) < 3:
            continue

        ytd_ret = float("nan")
        ytd_window = col[col.index.year == col.index[-1].year]
        if len(ytd_window) > 1 and ytd_window.iloc[0] != 0:
            ytd_ret = (ytd_window.iloc[-1] / ytd_window.iloc[0]) - 1.0

        rows.append(
            {
                "Theme": theme,
                "ETF": etf,
                "RS": rs_today.get(etf, 0),
                "RS_Change": rs_today.get(etf, 0) - rs_prev.get(etf, rs_today.get(etf, 0)),
                "Today": _safe_period_return(col, 1),
                "1W": _safe_period_return(col, 5),
                "1M": _safe_period_return(col, 21),
                "3M": _safe_period_return(col, 63),
                "YTD": ytd_ret,
            }
        )

    if not rows:
        return ""

    df = pd.DataFrame(rows).sort_values("RS", ascending=False).reset_index(drop=True)

    def fmt(v: float) -> str:
        if pd.isna(v):
            return "-"
        color = "#d32f2f" if v > 0 else "#1976d2" if v < 0 else "#757575"
        return f"<span style='color:{color};font-weight:600;'>{v * 100:+.2f}%</span>"

    for col in ["Today", "1W", "1M", "3M", "YTD"]:
        df[col] = df[col].apply(fmt)

    def fmt_rs_change(v: int) -> str:
        if pd.isna(v):
            return "-"
        vi = int(v)
        if vi > 0:
            return f"<span style='color:#d32f2f;font-weight:600;'>+{vi}</span>"
        if vi < 0:
            return f"<span style='color:#1976d2;font-weight:600;'>{vi}</span>"
        return "<span style='color:#757575;'>0</span>"

    df["RS"] = df["RS"].fillna(0).astype(int)
    df["RS변동"] = df["RS_Change"].apply(fmt_rs_change)

    table = df[["Theme", "ETF", "RS", "RS변동", "Today", "1W", "1M", "3M", "YTD"]].to_html(
        index=False, escape=False, border=1, justify="center"
    )

    return (
        '<div style="margin:10px 0;padding:15px;border:1px solid #e0e0e0;border-radius:10px;background-color:#ffffff;">'
        '<h3 style="margin:0 0 10px 0;color:#333;">Theme Tracker (ETF Proxy)</h3>'
        "<p style='margin:0 0 10px 0;color:#666;font-size:12px;'>테마별 대표 ETF 수익률로 상대 강도를 비교합니다.</p>"
        f'<div class="rs-table-wrap">{table}</div>'
        "</div>"
    )


def build_industry_rank_table(df_prices: pd.DataFrame, rs_df: pd.DataFrame) -> str:
    """Sector를 Industry 대체 지표로 사용한 기간별 랭킹."""
    if len(df_prices) < 252 or rs_df.empty:
        return ""

    sector_map = rs_df.set_index("Ticker")["Sector"].to_dict()
    sector_cols = [c for c in df_prices.columns if c in sector_map]
    if not sector_cols:
        return ""

    px = df_prices[sector_cols].copy()
    if px.empty:
        return ""

    # 종목별 가격을 시작점 1로 정규화한 뒤 섹터 평균 곡선을 만든다.
    # 이 방식은 일부 종목 결측이 있어도 섹터 곡선을 안정적으로 계산한다.
    norm = px.copy()
    for col in norm.columns:
        s = norm[col].dropna()
        if s.empty or s.iloc[0] == 0:
            norm[col] = float("nan")
        else:
            norm[col] = norm[col] / s.iloc[0]

    sector_curve = norm.T.groupby(pd.Series(sector_map)).mean().T
    sector_curve = sector_curve.ffill().dropna(how="all")
    if sector_curve.empty:
        return ""

    horizons = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "12M": 252}
    rank_df = pd.DataFrame({"Sector": sector_curve.columns})

    for label, bars in horizons.items():
        if len(sector_curve) <= bars:
            rank_df[label] = "-"
            continue
        period_ret = (sector_curve.iloc[-1] / sector_curve.iloc[-(bars + 1)]) - 1.0
        period_ret = pd.to_numeric(period_ret, errors="coerce")
        rank_raw = period_ret.rank(ascending=False, method="min")
        rank_df[label] = rank_raw.apply(lambda x: int(x) if pd.notna(x) else "-")

    def parse_int(x):
        try:
            return int(x)
        except Exception:
            return None

    rc = []
    for _, row in rank_df.iterrows():
        r1w = parse_int(row["1W"])
        r1m = parse_int(row["1M"])
        if r1w is None or r1m is None:
            rc.append("-")
            continue
        delta = r1m - r1w
        rc.append(f"+{delta}" if delta > 0 else str(delta))
    rank_df["Rank Change"] = rc

    rank_df = rank_df.sort_values(
        by="1D",
        key=lambda s: pd.to_numeric(s, errors="coerce").fillna(9999),
    )

    # 모든 기간 컬럼이 비정상이면 섹션 자체를 숨긴다.
    rank_cols = ["1D", "1W", "1M", "3M", "6M", "12M"]
    valid_count = pd.to_numeric(rank_df[rank_cols].stack(), errors="coerce").notna().sum()
    if int(valid_count) == 0:
        return ""

    table = rank_df[["Sector", "1D", "1W", "1M", "3M", "6M", "12M", "Rank Change"]].to_html(
        index=False, escape=False, border=1, justify="center"
    )
    return (
        '<div style="margin:10px 0;padding:15px;border:1px solid #e0e0e0;border-radius:10px;background-color:#ffffff;">'
        '<h3 style="margin:0 0 10px 0;color:#333;">Industry Rank (Sector Proxy)</h3>'
        "<p style='margin:0 0 10px 0;color:#666;font-size:12px;'>섹터 평균 수익률 순위를 기간별로 비교합니다.</p>"
        f'<div class="rs-table-wrap">{table}</div>'
        "</div>"
    )


def _band_score(value: float, thresholds: list[tuple[float, int]]) -> int:
    if pd.isna(value):
        return 0
    for th, score in thresholds:
        if value >= th:
            return score
    return 0


def _safe_debt_ratio(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    # yfinance debtToEquity가 120(%) 형식으로 오는 경우를 비율(1.2)로 정규화
    return value / 100.0 if value > 10 else value


def _extract_revenue_cagr_3y(ticker_obj) -> float:
    frames = []
    for attr in ["income_stmt", "financials"]:
        try:
            df = getattr(ticker_obj, attr, None)
            if callable(df):
                df = df()
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
        except Exception:
            continue
    for method in ["get_income_stmt", "get_financials"]:
        try:
            fn = getattr(ticker_obj, method, None)
            if callable(fn):
                df = fn()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    frames.append(df)
        except Exception:
            continue

    for frame in frames:
        row_key = None
        for idx in frame.index.astype(str):
            if "total revenue" in idx.lower().replace("_", " "):
                row_key = idx
                break
        if row_key is None:
            continue

        rev = pd.to_numeric(frame.loc[row_key], errors="coerce").dropna()
        if len(rev) < 4:
            continue
        rev = rev.sort_index()
        base = float(rev.iloc[-4])
        latest = float(rev.iloc[-1])
        if base <= 0 or latest <= 0:
            return float("nan")
        return (latest / base) ** (1.0 / 3.0) - 1.0
    return float("nan")


def build_quality_universe(nasdaq_rs: pd.DataFrame, sp500_rs: pd.DataFrame) -> pd.DataFrame:
    """NASDAQ100 + S&P500 결합 후 중복 티커 제거."""
    merged = pd.concat([nasdaq_rs, sp500_rs], ignore_index=True)
    merged = merged.sort_values("RS_Rating", ascending=False)
    merged = merged.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
    return merged


def fetch_quality_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Quality 스코어링용 펀더멘털/유동성 데이터 수집."""
    rows = []
    for t in tickers:
        revenue_growth = float("nan")
        earnings_growth = float("nan")
        roe = float("nan")
        operating_margins = float("nan")
        debt_to_equity = float("nan")
        market_cap = float("nan")
        avg_vol_10d = float("nan")
        revenue_cagr_3y = float("nan")
        current_price = float("nan")

        try:
            tk = yf.Ticker(t)
            info = tk.get_info() or {}
            revenue_growth = info.get("revenueGrowth", float("nan"))
            earnings_growth = info.get("earningsGrowth", float("nan"))
            roe = info.get("returnOnEquity", float("nan"))
            operating_margins = info.get("operatingMargins", float("nan"))
            debt_to_equity = _safe_debt_ratio(info.get("debtToEquity", float("nan")))
            market_cap = info.get("marketCap", float("nan"))
            avg_vol_10d = info.get("averageDailyVolume10Day", float("nan"))
            current_price = (
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or float("nan")
            )
            revenue_cagr_3y = _extract_revenue_cagr_3y(tk)
        except Exception:
            pass

        rows.append(
            {
                "Ticker": t,
                "RevenueGrowth": revenue_growth,
                "EarningsGrowth": earnings_growth,
                "RevenueCagr3Y": revenue_cagr_3y,
                "ReturnOnEquity": roe,
                "OperatingMargins": operating_margins,
                "DebtToEquity": debt_to_equity,
                "MarketCap": market_cap,
                "AvgDailyVolume10Day": avg_vol_10d,
                "CurrentPrice": current_price,
            }
        )

    df = pd.DataFrame(rows)

    # 최근 20일 평균 거래대금(달러) 추정: 20일 평균 거래량 * 최근 종가
    try:
        vol_raw = yf.download(tickers, period="3mo", auto_adjust=False, progress=False)
        if isinstance(vol_raw.columns, pd.MultiIndex):
            close_df = (
                vol_raw.xs("Close", axis=1, level=0)
                if "Close" in vol_raw.columns.get_level_values(0)
                else pd.DataFrame()
            )
            volume_df = (
                vol_raw.xs("Volume", axis=1, level=0)
                if "Volume" in vol_raw.columns.get_level_values(0)
                else pd.DataFrame()
            )
        else:
            close_df = vol_raw["Close"].to_frame() if "Close" in vol_raw.columns else pd.DataFrame()
            volume_df = vol_raw["Volume"].to_frame() if "Volume" in vol_raw.columns else pd.DataFrame()

        if not close_df.empty and not volume_df.empty:
            dollar_vol = close_df * volume_df
            dv20 = dollar_vol.rolling(20).mean().iloc[-1]
            if isinstance(dv20, pd.Series):
                df["DollarVol20"] = df["Ticker"].map(dv20.to_dict())
            else:
                only_ticker = tickers[0] if tickers else None
                df["DollarVol20"] = dv20 if only_ticker else float("nan")
        else:
            df["DollarVol20"] = float("nan")
    except Exception:
        df["DollarVol20"] = float("nan")

    # volume만 있고 dollar volume이 비면 currentPrice로 대체 계산
    fallback_dollar = df["AvgDailyVolume10Day"] * df["CurrentPrice"]
    df["DollarVol20"] = df["DollarVol20"].fillna(fallback_dollar)
    return df


def compute_quality_score(
    df_candidates: pd.DataFrame,
    df_prices: pd.DataFrame,
    sector_rank_map: dict[str, int],
) -> pd.DataFrame:
    """정성(quality) 스코어를 계산하고 랭킹 테이블을 반환."""
    quality = df_candidates.copy()
    fundamentals = fetch_quality_fundamentals(quality["Ticker"].tolist())
    quality = quality.merge(fundamentals, on="Ticker", how="left")

    sector_count = max(len(sector_rank_map), 1)
    top_cut = max(1, int(sector_count * 0.3 + 0.999))
    mid_cut = max(top_cut + 1, int(sector_count * 0.7 + 0.999))

    liq_threshold = quality["DollarVol20"].dropna().quantile(0.4)

    growth_scores = []
    profitability_scores = []
    leadership_scores = []
    chart_scores = []
    liquidity_scores = []
    setup_texts = []
    confidence_tags = []

    required_cols = [
        "RevenueGrowth",
        "EarningsGrowth",
        "RevenueCagr3Y",
        "ReturnOnEquity",
        "OperatingMargins",
        "DebtToEquity",
        "MarketCap",
        "AvgDailyVolume10Day",
        "DollarVol20",
    ]

    for _, row in quality.iterrows():
        # 1) 성장 품질 (35)
        score_growth = 0
        score_growth += _band_score(row.get("RevenueGrowth"), [(0.15, 12), (0.05, 8), (0.0, 4)])
        score_growth += _band_score(row.get("EarningsGrowth"), [(0.20, 12), (0.05, 8), (0.0, 4)])
        score_growth += _band_score(row.get("RevenueCagr3Y"), [(0.12, 11), (0.05, 7), (0.0, 3)])

        # 2) 수익성/재무 (25)
        score_profit = 0
        score_profit += _band_score(row.get("ReturnOnEquity"), [(0.17, 10), (0.10, 7), (0.05, 4)])
        score_profit += _band_score(row.get("OperatingMargins"), [(0.15, 8), (0.08, 5), (0.03, 2)])
        debt = row.get("DebtToEquity")
        if pd.notna(debt):
            if debt < 0.7:
                score_profit += 7
            elif debt <= 1.5:
                score_profit += 4

        # 3) 수급/리더십 (20)
        score_lead = 0
        rs_rating = row.get("RS_Rating")
        rs_change = row.get("RS_Change")
        if pd.notna(rs_rating):
            if rs_rating >= 90:
                score_lead += 10
            elif rs_rating >= 80:
                score_lead += 7
            elif rs_rating >= 70:
                score_lead += 4
        if pd.notna(rs_change):
            if rs_change > 0:
                score_lead += 4
            elif rs_change == 0:
                score_lead += 2
        s_rank = sector_rank_map.get(row.get("Sector"), sector_count)
        if s_rank <= top_cut:
            score_lead += 6
        elif s_rank <= mid_cut:
            score_lead += 3

        # 4) 차트 품질 (15)
        setup = "-"
        if row["Ticker"] in df_prices.columns:
            setup = detect_setup(df_prices, row["Ticker"])
        setup_texts.append(setup)
        score_chart = 0
        if setup and setup != "-":
            parts = [p.strip() for p in str(setup).split(",") if p.strip()]
            if "VCP" in parts:
                score_chart += 6
            if "돌파" in parts:
                score_chart += 5
            if "타이트" in parts:
                score_chart += 2
            if "52주高" in parts:
                score_chart += 2
        score_chart = min(score_chart, 15)

        # 5) 규모/유동성 (5)
        score_liq = 0
        market_cap = row.get("MarketCap")
        if pd.notna(market_cap):
            if market_cap > 10_000_000_000:
                score_liq += 3
            elif market_cap >= 3_000_000_000:
                score_liq += 2
        dv20 = row.get("DollarVol20")
        if pd.notna(liq_threshold) and pd.notna(dv20) and dv20 >= liq_threshold:
            score_liq += 2

        # 결측 신뢰도
        missing_ratio = float(pd.isna(row[required_cols]).sum()) / float(len(required_cols))
        confidence_tags.append("Low" if missing_ratio >= 0.5 else "High")

        growth_scores.append(score_growth)
        profitability_scores.append(score_profit)
        leadership_scores.append(score_lead)
        chart_scores.append(score_chart)
        liquidity_scores.append(score_liq)

    quality["Setup"] = setup_texts
    quality["ScoreGrowth"] = growth_scores
    quality["ScoreProfitability"] = profitability_scores
    quality["ScoreLeadership"] = leadership_scores
    quality["ScoreChart"] = chart_scores
    quality["ScoreLiquidity"] = liquidity_scores
    quality["DataConfidence"] = confidence_tags
    quality["QualityScore"] = (
        quality["ScoreGrowth"]
        + quality["ScoreProfitability"]
        + quality["ScoreLeadership"]
        + quality["ScoreChart"]
        + quality["ScoreLiquidity"]
    ).clip(upper=100)

    quality = quality.sort_values(
        by=["QualityScore", "RS_Rating"],
        ascending=[False, False],
    ).reset_index(drop=True)
    quality["Rank"] = range(1, len(quality) + 1)
    return quality


def build_quality_post(date_str: str, quality_df: pd.DataFrame) -> tuple[str, str]:
    """Quality 리더십 포스트의 제목/본문 HTML 생성."""
    title = f"{date_str} 미국 주식 Quality 리더십 리포트 (S&P500+NASDAQ100)"
    top_df = quality_df.head(QUALITY_TOP_N).copy()

    if top_df.empty:
        body = "<p>오늘 생성 가능한 Quality 데이터가 없습니다.</p>"
        return title, body

    top_df["순위"] = top_df["Rank"].astype(int)
    top_df["티커"] = top_df["Ticker"]
    top_df["회사명"] = top_df["Name"]
    top_df["섹터"] = top_df["Sector"]
    top_df["Quality점수"] = top_df["QualityScore"].astype(int)
    top_df["일간보정"] = top_df.get("daily_adjust", 0).astype(int)
    top_df["주간보정"] = top_df.get("weekly_adjust", 0).astype(int)
    top_df["총보정"] = top_df.get("total_adjust_clipped", 0).astype(int)
    top_df["최종점수"] = top_df.get("FinalQualityScore", top_df["QualityScore"]).astype(int)
    top_df["성장점수"] = top_df["ScoreGrowth"].astype(int)
    top_df["재무점수"] = top_df["ScoreProfitability"].astype(int)
    top_df["수급점수"] = top_df["ScoreLeadership"].astype(int)
    top_df["차트점수"] = top_df["ScoreChart"].astype(int)
    top_df["유동성점수"] = top_df["ScoreLiquidity"].astype(int)
    top_df["RS등급"] = top_df["RS_Rating"].fillna(0).astype(int)
    top_df["셋업"] = top_df["Setup"].fillna("-")
    top_df["DataConfidence"] = top_df["DataConfidence"].fillna("Low")

    display_df = top_df[
        [
            "순위", "티커", "회사명", "섹터",
            "Quality점수", "일간보정", "주간보정", "총보정", "최종점수",
            "성장점수", "재무점수", "수급점수", "차트점수", "유동성점수",
            "RS등급", "셋업", "DataConfidence",
        ]
    ]
    table_html = display_df.to_html(index=False, escape=False, border=1, justify="center")

    style_html = """
<style>
.rs-table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.rs-table-wrap table { border-collapse: collapse; width: 100%; font-size: 13px; }
.rs-table-wrap th, .rs-table-wrap td { padding: 5px 8px; white-space: nowrap; border: 1px solid #ddd; }
.rs-table-wrap th { background: #f5f5f5; font-weight: bold; }
</style>
"""
    intro_html = f"""
<p><strong>{date_str} QUALITY LEADERSHIP BRIEF</strong></p>
<p>정량 RS 지표와 함께, 정성(quality) 우선순위 지표를 점수화해 상위 종목을 정리한 별도 리포트입니다.</p>
<p>대상 유니버스: S&P 500 + NASDAQ 100 (중복 제거), 노출: Top {QUALITY_TOP_N}</p>
"""
    formula_html = """
<div style="margin:10px 0;padding:15px;border:1px solid #e0e0e0;border-radius:10px;background-color:#fafafa;">
  <h3 style="margin:0 0 10px 0;color:#333;">Quality 점수 산식 (총 100점, 완전 공개)</h3>
  <ol style="margin:0;padding-left:18px;line-height:1.7;">
    <li><strong>이익/매출 성장 품질 (35점)</strong>: 매출성장(12) + EPS성장(12) + 3년 매출CAGR(11)</li>
    <li><strong>수익성/재무 건전성 (25점)</strong>: ROE(10) + 영업이익률(8) + 부채비율(7)</li>
    <li><strong>수급/리더십/RS (20점)</strong>: RS등급(10) + RS변화(4) + 섹터랭크(6)</li>
    <li><strong>차트 구조 품질 (15점)</strong>: VCP(+6), 돌파(+5), 타이트(+2), 52주高(+2), 최대 15</li>
    <li><strong>규모/유동성 (5점)</strong>: 시총(3) + 최근20일 거래대금 상위60%(2)</li>
  </ol>
  <p style="font-size:12px;color:#666;margin:10px 0 0 0;">
    결측치 정책: 해당 지표는 0점 처리. 필수 지표 결측 비율 50% 이상이면 DataConfidence=Low.
  </p>
</div>
"""
    adjustment_html = """
<div style="margin:10px 0;padding:15px;border:1px solid #e0e0e0;border-radius:10px;background-color:#f8fbff;">
  <h3 style="margin:0 0 10px 0;color:#333;">정성 보정 규칙 (±5)</h3>
  <ul style="margin:0;padding-left:18px;line-height:1.7;">
    <li>일간 Agent 신호: 방향(+/-)과 신뢰도(confidence) 기반으로 종목당 -3 ~ +3</li>
    <li>주간 Deep 컨텍스트: 섹터/테마 확증 기반으로 종목당 -2 ~ +2</li>
    <li>총보정 = clip(일간보정 + 주간보정, -5, +5)</li>
    <li>최종점수 = Quality점수 + 총보정</li>
  </ul>
</div>
"""
    disclaimer_html = """
<p style="font-size:12px;color:#888;margin-top:20px;">
<em>※ 본 글은 특정 종목의 매수/매도 추천이 아니며, 정보 제공만을 목적으로 합니다. 투자 판단의 책임은 투자자에게 있습니다.</em>
</p>
"""

    content_html = (
        style_html
        + intro_html
        + formula_html
        + adjustment_html
        + f'<h3>Quality Top {QUALITY_TOP_N}</h3>'
        + f'<div class="rs-table-wrap">{table_html}</div>'
        + disclaimer_html
    )
    return title, content_html


def post_quality_report(
    date_str: str,
    quality_df: pd.DataFrame,
    output_dir: str,
    quality_filename: str = "quality_with_adjustment",
    publish: bool = True,
):
    """Quality 리포트 저장 및 (옵션) Blogger 포스팅."""
    title, content_html = build_quality_post(date_str, quality_df)
    quality_path = os.path.join(output_dir, f"{date_str}_{quality_filename}.html")
    with open(quality_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- TITLE: {title} -->\n")
        f.write(content_html)
    print(f"Quality 리포트 HTML 저장됨: {quality_path}")

    post_url = None
    if BLOGGER_AUTO_POST and publish:
        post_url = post_to_blogger(title, content_html)
        if not post_url:
            raise RuntimeError("QUALITY_POST_FAILED: Blogger 포스팅 실패")
    return quality_path, post_url, title


def run_agent_qual_signals(date_str: str, tickers: list[str]) -> pd.DataFrame:
    """일간 Agent 신호(뉴스 기반)를 수집해 표준 스키마로 반환."""
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "signal_type", "direction", "confidence", "source_url", "summary"])

    pos_kw = [
        "beats", "beat", "upgrade", "raised guidance", "partnership", "approved",
        "record", "launch", "strong demand", "buyback", "acquisition",
    ]
    neg_kw = [
        "miss", "downgrade", "cuts guidance", "probe", "lawsuit", "recall",
        "delay", "weak demand", "fraud", "layoff", "bankruptcy",
    ]

    rows = []
    # 비용/시간 안정성을 위해 RS/Quality 후보 중 상위 80개만 신호 수집
    for t in tickers[:80]:
        try:
            news = yf.Ticker(t).get_news()
        except Exception:
            continue
        if not news:
            continue

        # 최신 3건 기반으로 요약 점수
        score = 0.0
        count = 0
        best_url = ""
        best_title = ""
        for item in news[:3]:
            title = str(item.get("title", "") or "")
            link = str(item.get("link", "") or "")
            txt = title.lower()
            local = 0.0
            for kw in pos_kw:
                if kw in txt:
                    local += 1.0
            for kw in neg_kw:
                if kw in txt:
                    local -= 1.0
            if local != 0:
                count += 1
                score += local
            if not best_url and link:
                best_url = link
            if not best_title and title:
                best_title = title

        if count == 0:
            continue

        avg = score / max(count, 1)
        direction = 1 if avg > 0 else -1 if avg < 0 else 0
        confidence = min(1.0, 0.4 + min(abs(avg), 2.0) * 0.3)
        rows.append(
            {
                "date": date_str,
                "ticker": t,
                "signal_type": "news_sentiment",
                "direction": direction,
                "confidence": round(float(confidence), 3),
                "source_url": best_url,
                "summary": best_title[:220] if best_title else "headline sentiment signal",
            }
        )

    return pd.DataFrame(rows, columns=["date", "ticker", "signal_type", "direction", "confidence", "source_url", "summary"])


def generate_weekly_deep_context(
    date_str: str,
    quality_universe: pd.DataFrame,
    sector_rank_map: dict[str, int],
) -> pd.DataFrame:
    """주간 Deep 컨텍스트(섹터/테마 확증) 생성."""
    if quality_universe.empty:
        return pd.DataFrame(columns=["week", "theme_or_sector", "thesis", "evidence", "risk", "confidence", "source_url"])

    monday = (dt.datetime.strptime(date_str, "%Y-%m-%d").date() - dt.timedelta(days=dt.datetime.strptime(date_str, "%Y-%m-%d").weekday()))
    week_str = monday.strftime("%Y-%m-%d")

    sector_mean = (
        quality_universe.groupby("Sector")
        .agg(avg_rs=("RS_Rating", "mean"), mean_3m=("Return_3M", "mean"), n=("Ticker", "count"))
        .reset_index()
    )
    sector_mean["rank"] = sector_mean["Sector"].map(sector_rank_map).fillna(len(sector_rank_map) + 1)
    sector_mean = sector_mean.sort_values("rank")

    rows = []
    top3 = sector_mean.head(3)
    bot3 = sector_mean.tail(3)
    for _, r in top3.iterrows():
        rows.append(
            {
                "week": week_str,
                "theme_or_sector": r["Sector"],
                "thesis": "섹터 리더십 유지 가능성",
                "evidence": f"RS 평균 {r['avg_rs']:.1f}, 3M 평균수익률 {r['mean_3m'] * 100:+.2f}%",
                "risk": "실적 시즌 변동성/밸류에이션 부담",
                "confidence": 0.75,
                "source_url": "https://finance.yahoo.com/",
            }
        )
    for _, r in bot3.iterrows():
        rows.append(
            {
                "week": week_str,
                "theme_or_sector": r["Sector"],
                "thesis": "섹터 약세/리스크 확대",
                "evidence": f"RS 평균 {r['avg_rs']:.1f}, 3M 평균수익률 {r['mean_3m'] * 100:+.2f}%",
                "risk": "숏커버링/정책 변화 시 급반등 가능",
                "confidence": 0.72,
                "source_url": "https://finance.yahoo.com/",
            }
        )
    return pd.DataFrame(rows, columns=["week", "theme_or_sector", "thesis", "evidence", "risk", "confidence", "source_url"])


def run_deep_research_weekly_context(date_str: str) -> pd.DataFrame:
    """최근 주간 Deep 컨텍스트 파일 로드."""
    data_dir = os.path.join(os.path.dirname(__file__), DATA_DIR_NAME)
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "deep_research_weekly.json")
    if not os.path.exists(path):
        print("DEEP_CONTEXT_FETCH_FAILED: weekly context file not found")
        return pd.DataFrame(columns=["week", "theme_or_sector", "thesis", "evidence", "risk", "confidence", "source_url"])
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            records = payload.get("records", [])
        else:
            records = payload
        return pd.DataFrame(records)
    except Exception as e:
        print(f"DEEP_CONTEXT_FETCH_FAILED: {e}")
        return pd.DataFrame(columns=["week", "theme_or_sector", "thesis", "evidence", "risk", "confidence", "source_url"])


def save_weekly_deep_context(deep_df: pd.DataFrame):
    data_dir = os.path.join(os.path.dirname(__file__), DATA_DIR_NAME)
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "deep_research_weekly.json")
    payload = {
        "updated_at": dt.datetime.utcnow().isoformat() + "Z",
        "records": deep_df.to_dict(orient="records"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Deep context saved: {path}")
    return path


def compute_qual_adjustment(
    quality_df: pd.DataFrame,
    qual_signal_df: pd.DataFrame,
    deep_theme_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """일간 Agent + 주간 Deep 컨텍스트를 점수 보정값으로 변환."""
    date_str = dt.date.today().strftime("%Y-%m-%d")
    base = pd.DataFrame({"ticker": quality_df["Ticker"].tolist()})
    base["date"] = date_str
    base["daily_adjust"] = 0
    base["weekly_adjust"] = 0

    # 일간 Agent 신호: confidence와 방향으로 -3 ~ +3
    if qual_signal_df is not None and not qual_signal_df.empty:
        sig = qual_signal_df.copy()
        sig["ticker"] = sig["ticker"].astype(str)

        def to_daily(row):
            d = int(row.get("direction", 0))
            c = float(row.get("confidence", 0.0))
            if d == 0 or c < 0.30:
                return 0
            if c >= 0.85:
                m = 3
            elif c >= 0.65:
                m = 2
            else:
                m = 1
            return m if d > 0 else -m

        sig["daily_score"] = sig.apply(to_daily, axis=1)
        sig_g = sig.groupby("ticker", as_index=False)["daily_score"].sum()
        sig_g["daily_adjust"] = sig_g["daily_score"].clip(lower=-3, upper=3)
        base = base.merge(sig_g[["ticker", "daily_adjust"]], on="ticker", how="left", suffixes=("", "_new"))
        base["daily_adjust"] = base["daily_adjust_new"].fillna(base["daily_adjust"]).astype(int)
        base = base.drop(columns=["daily_adjust_new"])

    # 주간 Deep 컨텍스트: 섹터/테마 매칭으로 -2 ~ +2
    if deep_theme_df is not None and not deep_theme_df.empty:
        local = deep_theme_df.copy()
        local["theme_or_sector"] = local["theme_or_sector"].astype(str)
        local["thesis"] = local["thesis"].astype(str)
        local["confidence"] = pd.to_numeric(local.get("confidence", 0.0), errors="coerce").fillna(0.0)

        sector_scores = {}
        for _, r in local.iterrows():
            theme = r["theme_or_sector"]
            conf = float(r["confidence"])
            thesis = r["thesis"].lower()
            sign = 1
            if any(k in thesis for k in ["약세", "리스크", "risk", "bear", "weak"]):
                sign = -1
            mag = 2 if conf >= 0.75 else 1 if conf >= 0.55 else 0
            sector_scores[theme] = max(-2, min(2, sector_scores.get(theme, 0) + sign * mag))

        weekly = []
        for t in base["ticker"].tolist():
            s = sector_map.get(t, "")
            weekly.append(sector_scores.get(s, 0))
        base["weekly_adjust"] = pd.Series(weekly, index=base.index).astype(int)

    base["total_adjust"] = base["daily_adjust"] + base["weekly_adjust"]
    base["total_adjust_clipped"] = base["total_adjust"].clip(lower=-5, upper=5).astype(int)
    return base[["date", "ticker", "daily_adjust", "weekly_adjust", "total_adjust_clipped"]]


def merge_quality_with_adjustment(
    quality_df: pd.DataFrame,
    qual_adjustment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Quality 점수에 보정값을 결합해 최종 점수 생성."""
    merged = quality_df.copy()
    if qual_adjustment_df is None or qual_adjustment_df.empty:
        merged["daily_adjust"] = 0
        merged["weekly_adjust"] = 0
        merged["total_adjust_clipped"] = 0
    else:
        merged = merged.merge(
            qual_adjustment_df,
            left_on="Ticker",
            right_on="ticker",
            how="left",
        )
        for c in ["daily_adjust", "weekly_adjust", "total_adjust_clipped"]:
            merged[c] = merged[c].fillna(0).astype(int)
        if "ticker" in merged.columns:
            merged = merged.drop(columns=["ticker"])
        if "date" in merged.columns:
            merged = merged.drop(columns=["date"])

    merged["FinalQualityScore"] = (merged["QualityScore"] + merged["total_adjust_clipped"]).clip(lower=0, upper=105)
    merged = merged.sort_values(by=["FinalQualityScore", "RS_Rating"], ascending=[False, False]).reset_index(drop=True)
    merged["Rank"] = range(1, len(merged) + 1)
    return merged


def get_sector_avg_return(df_prices, rs_df):
    """섹터별 당일 평균 수익률 계산."""
    latest = df_prices.iloc[-1]
    prev = df_prices.iloc[-2]
    returns = (latest - prev) / (prev + 1e-9)
    
    # Ticker별 섹터 매핑
    sector_map = rs_df.set_index("Ticker")["Sector"].to_dict()
    
    # 데이터프레임 생성
    df_ret = pd.DataFrame({"Return": returns})
    df_ret["Sector"] = df_ret.index.map(sector_map)
    df_ret = df_ret.dropna()
    
    # 섹터별 평균
    sector_g = df_ret.groupby("Sector")["Return"].mean().reset_index()
    sector_g = sector_g.sort_values("Return", ascending=False)
    
    return sector_g


def build_combined_post(date_str, nasdaq_rs, sp500_rs, nasdaq_prices, sp500_prices):
    """통합 리포트 HTML 생성 (Market Brief 포함)."""
    
    # --- Market Breadth (S&P 500 기준) ---
    breadth_html = build_market_breadth(sp500_prices, "S&P 500")
    theme_html = build_theme_tracker()
    industry_rank_html = build_industry_rank_table(sp500_prices, sp500_rs)
    
    # --- Sector Performance ---
    sector_perf = get_sector_avg_return(sp500_prices, sp500_rs)
    top3_sectors = sector_perf.head(3)
    bottom3_sectors = sector_perf.tail(3).sort_values("Return")
    
    def format_sec_ret(row):
        r = row["Return"] * 100
        color = "red" if r > 0 else "blue"
        return f"<li>{row['Sector']}: <span style='color:{color}'>{r:+.2f}%</span></li>"
        
    sector_html = f"""
<div style="margin:10px 0;padding:15px;border:1px solid #e0e0e0;border-radius:10px;background-color:#ffffff;">
    <h3 style="margin:0 0 10px 0;color:#333;">🏭 Sector Performance</h3>
    <div style="display:flex;justify-content:space-between;">
        <div style="width:48%;">
            <strong style="color:#d32f2f;">▲ 강세 섹터</strong>
            <ul style="padding-left:20px;margin:5px 0;font-size:13px;">
                {''.join(top3_sectors.apply(format_sec_ret, axis=1))}
            </ul>
        </div>
        <div style="width:48%;">
            <strong style="color:#1976d2;">▼ 약세 섹터</strong>
            <ul style="padding-left:20px;margin:5px 0;font-size:13px;">
                {''.join(bottom3_sectors.apply(format_sec_ret, axis=1))}
            </ul>
        </div>
    </div>
</div>
"""

    # --- 기존 코드: 섹터 순위 맵 ---
    all_sector_df = (
        sp500_rs.groupby("Sector")
        .agg(섹터평균_RS=("RS_Rating", "mean"))
        .reset_index()
    )
    all_sector_df = all_sector_df.sort_values("섹터평균_RS", ascending=False).reset_index(drop=True)
    all_sector_df["순위"] = range(1, len(all_sector_df) + 1)
    sector_rank_map = dict(zip(all_sector_df["Sector"], all_sector_df["순위"]))

    # --- 테이블 생성 함수 ---
    def format_table(top_df, all_prices_local):
        top_df = top_df.copy()
        top_df["티커"] = top_df["Ticker"]
        top_df["소속"] = top_df["Universe"]
        top_df["회사명"] = top_df["Name"]
        top_df["섹터"] = top_df["Sector"].apply(
            lambda s: f"{s}({sector_rank_map.get(s, '?')}위)"
        )
        top_df["RS 등급"] = top_df["RS_Rating"].fillna(0).astype(int)

        def format_rs_change(val):
            if pd.isna(val) or val == "":
                return "-"
            try:
                v = int(float(val))
            except ValueError:
                return "-"
                
            if v > 0: return f"<span style='color:#d32f2f'>+{v}</span>"
            elif v < 0: return f"<span style='color:#1976d2'>{v}</span>"
            return "-"
        top_df["RS변동"] = top_df["RS_Change"].apply(format_rs_change)

        top_df["셋업"] = top_df["Ticker"].apply(
            lambda t: detect_setup(all_prices_local, t) if (all_prices_local is not None and t in all_prices_local.columns) else "-"
        )

        top_df["1개월 수익률"] = top_df["Return_1M"].apply(format_percent)
        top_df["3개월 수익률"] = top_df["Return_3M"].apply(format_percent)
        top_df["12개월 수익률"] = top_df["Return_12M"].apply(format_percent)
        top_df["1개월 RS"] = top_df["RS_1M"].fillna(0).astype(int)
        top_df["3개월 RS"] = top_df["RS_3M"].fillna(0).astype(int)
        top_df["12개월 RS"] = top_df["RS_12M"].fillna(0).astype(int)

        display_df = top_df[[
            "티커", "소속", "회사명", "섹터",
            "RS 등급", "RS변동", "셋업",
            "1개월 수익률", "3개월 수익률", "12개월 수익률",
            "1개월 RS", "3개월 RS", "12개월 RS",
        ]]
        return display_df.to_html(index=False, escape=False, border=1, justify="center")

    # --- NASDAQ + S&P 통합 테이블 ---
    nasdaq_union = nasdaq_rs.copy()
    nasdaq_union["Universe"] = "NASDAQ 100"
    sp500_union = sp500_rs.copy()
    sp500_union["Universe"] = "S&P 500"

    combined_union = pd.concat([nasdaq_union, sp500_union], ignore_index=True)
    combined_union = combined_union.sort_values("RS_Rating", ascending=False)
    combined_union = combined_union.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
    combined_top = combined_union.head(TOP_N).copy()

    all_prices_for_table = pd.concat([nasdaq_prices, sp500_prices], axis=1)
    all_prices_for_table = all_prices_for_table.loc[:, ~all_prices_for_table.columns.duplicated()]
    combined_table = format_table(combined_top, all_prices_for_table)

    # --- 섹터 전체 요약 ---
    sector_summary = (
        sp500_rs.groupby("Sector")
        .agg(
            섹터평균_RS=("RS_Rating", "mean"),
            종목수=("Ticker", "count"),
        )
        .reset_index()
    )
    sector_summary["섹터평균_RS"] = sector_summary["섹터평균_RS"].round(1)
    sector_summary = sector_summary.sort_values("섹터평균_RS", ascending=False).reset_index(drop=True)
    sector_summary["섹터"] = sector_summary.apply(
        lambda row: f"{row['Sector']}({sector_rank_map.get(row['Sector'], '?')}위)", axis=1
    )
    sector_table_html = sector_summary[["섹터", "섹터평균_RS", "종목수"]].to_html(
        index=False, escape=False, border=1, justify="center"
    )

    # --- 최종 조립 ---
    style_html = """
<style>
.rs-table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.rs-table-wrap table { border-collapse: collapse; width: 100%; font-size: 13px; }
.rs-table-wrap th, .rs-table-wrap td { padding: 5px 8px; white-space: nowrap; border: 1px solid #ddd; }
.rs-table-wrap th { background: #f5f5f5; font-weight: bold; }
</style>
"""
    intro_html = f"""
<p><strong>{date_str} DAILY MARKET BRIEF</strong></p>
<p>오늘의 시장 브리핑과 상대강도(RS) 분석 리포트입니다.</p>
<p style="font-size:12px;color:#666;">
강세/약세 표기는 시장 확산 지표(상승/하락 비율, 52주 신고가/신저가 비율, 50일/200일선 상회 비율) 점수 합산 기준으로 분류했습니다.
</p>
"""
    disclaimer_html = """
<p style="font-size:12px;color:#888;margin-top:20px;">
<em>※ 본 글은 특정 종목의 매수/매도 추천이 아니며, 정보 제공만을 목적으로 합니다. 투자 판단의 책임은 투자자에게 있습니다.</em>
</p>
"""

    content_html = (
        style_html
        + intro_html
        + breadth_html
        + theme_html
        + sector_html
        + industry_rank_html
        + "<hr style='margin:20px 0;border:0;border-top:1px solid #eee;'/>"
        + f'<h3>🚀 통합 상대강도 TOP {TOP_N} (NASDAQ 100 + S&P 500)</h3>'
        + f'<div class="rs-table-wrap">{combined_table}</div>'
        + "<br/>"
        + '<h3>📊 섹터별 강도 요약 (S&P 500 기준)</h3>'
        + f'<div class="rs-table-wrap">{sector_table_html}</div>'
        + disclaimer_html
    )
    return content_html


def run_universe(index_name, loader_func, output_dir=None): # output_dir 호환성 유지
    """특정 인덱스에 대해 종목 로드, 가격 다운로드, RS 계산. (데이터만 반환)"""
    # output_dir 인자는 레거시 호출 호환성을 위해 남겨둠 (사용 안 함)
    print(f"[{index_name}] 종목 리스트 불러오는 중...")
    tickers = loader_func()
    print(f"[{index_name}] 종목 수: {len(tickers)}")

    print(f"[{index_name}] 가격 데이터 다운로드 중...")
    df_prices = download_price_data(tickers, LOOKBACK_DAYS)

    print(f"[{index_name}] RS 계산(IBD 스타일) 중...")
    rs_df = compute_rs_ibd_style(df_prices)

    return rs_df, df_prices


def main():
    parser = argparse.ArgumentParser(description="RS + Quality reporting pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "quant", "quality", "hybrid", "weekly-deep"],
        default="all",
        help="Run mode: all|quant|quality|hybrid|weekly-deep",
    )
    parser.add_argument(
        "--weekly-deep",
        action="store_true",
        help="(deprecated) same as --mode weekly-deep",
    )
    args = parser.parse_args()
    mode = "weekly-deep" if args.weekly_deep else args.mode

    today = dt.date.today()
    date_str = today.strftime("%Y-%m-%d")
    root_dir = os.path.dirname(__file__)
    output_dir = os.path.join(root_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 수집
    nasdaq_rs, nasdaq_prices = run_universe("NASDAQ 100", load_nasdaq100_universe)
    sp500_rs, sp500_prices = run_universe("S&P 500", load_sp500_universe)
    quality_universe = build_quality_universe(nasdaq_rs, sp500_rs)

    sector_rank_df = (
        quality_universe.groupby("Sector")
        .agg(섹터평균_RS=("RS_Rating", "mean"))
        .reset_index()
        .sort_values("섹터평균_RS", ascending=False)
        .reset_index(drop=True)
    )
    sector_rank_df["순위"] = range(1, len(sector_rank_df) + 1)
    sector_rank_map = dict(zip(sector_rank_df["Sector"], sector_rank_df["순위"]))

    # 주간 모드: deep context만 생성/저장
    if mode == "weekly-deep":
        print("\n===== Weekly Deep Context 생성 시작 =====")
        deep_df = generate_weekly_deep_context(date_str, quality_universe, sector_rank_map)
        save_weekly_deep_context(deep_df)
        print("Weekly Deep Context 생성 완료")
        return

    if mode in ["all", "quant"]:
        print("\n통합 리포트 생성 중...")
        combined_html = build_combined_post(
            date_str, nasdaq_rs, sp500_rs, nasdaq_prices, sp500_prices
        )
        combined_title = f"{date_str} 미국 주식 시장 브리핑 & 상대강도 리포트"
        path = os.path.join(output_dir, f"{date_str}_combined.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"<!-- TITLE: {combined_title} -->\n")
            f.write(combined_html)
        print(f"통합 리포트 HTML 저장됨: {path}")

        if BLOGGER_AUTO_POST:
            print("\n===== Blogger 자동 포스팅 시작 (Quant) =====")
            post_to_blogger(combined_title, combined_html)

    needs_quality_base = mode in ["all", "quality", "hybrid"]
    if needs_quality_base:
        print("\n===== Quality 리포트 생성 시작 =====")
        all_prices = pd.concat([nasdaq_prices, sp500_prices], axis=1)
        all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]
        quality_df = compute_quality_score(
            df_candidates=quality_universe,
            df_prices=all_prices,
            sector_rank_map=sector_rank_map,
        )

        base_quality_df = quality_df.assign(
            daily_adjust=0,
            weekly_adjust=0,
            total_adjust_clipped=0,
            FinalQualityScore=quality_df["QualityScore"],
        )

        post_quality_report(
            date_str=date_str,
            quality_df=base_quality_df,
            output_dir=output_dir,
            quality_filename="quality",
            publish=(mode in ["all", "quality"]),
        )

    if mode in ["all", "hybrid"]:
        # 일간 Agent 신호
        try:
            qual_signal_df = run_agent_qual_signals(date_str, quality_df["Ticker"].tolist())
        except Exception as e:
            print(f"AGENT_SIGNAL_FETCH_FAILED: {e}")
            qual_signal_df = pd.DataFrame(columns=["date", "ticker", "signal_type", "direction", "confidence", "source_url", "summary"])
        if qual_signal_df.empty:
            print("AGENT_SIGNAL_FETCH_FAILED: no usable daily signals")

        qual_signal_path = os.path.join(output_dir, f"{date_str}_qual_signals.json")
        with open(qual_signal_path, "w", encoding="utf-8") as f:
            json.dump(qual_signal_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        print(f"Agent signals saved: {qual_signal_path}")

        # 주간 Deep 컨텍스트 로드
        deep_theme_df = run_deep_research_weekly_context(date_str)

        # 보정 계산/결합
        ticker_sector_map = dict(zip(quality_df["Ticker"], quality_df["Sector"]))
        qual_adjustment_df = compute_qual_adjustment(
            quality_df=quality_df,
            qual_signal_df=qual_signal_df,
            deep_theme_df=deep_theme_df,
            sector_map=ticker_sector_map,
        )
        adjusted_quality_df = merge_quality_with_adjustment(quality_df, qual_adjustment_df)

        try:
            post_quality_report(
                date_str=date_str,
                quality_df=adjusted_quality_df,
                output_dir=output_dir,
                quality_filename="quality_with_adjustment",
                publish=True,
            )
            if BLOGGER_AUTO_POST:
                print("Quality 리포트 포스팅 완료 (Hybrid)")
        except Exception as e:
            print(f"QUALITY_POST_FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

