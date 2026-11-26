import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TARGET_SYMBOL = "^TWII"

KNOWN_NAMES = {
    "^TWII": "加權指數",
    "0050.TW": "元大台灣50",
    "0050": "元大台灣50",
    "2330.TW": "台積電",
    "2330": "台積電",
    "2317.TW": "鴻海",
    "2317": "鴻海",
    "2454.TW": "聯發科",
    "2454": "聯發科",
    "2308.TW": "台達電",
    "2308": "台達電",
    "2382.TW": "廣達",
    "2382": "廣達",
    "0056.TW": "元大高股息",
    "0056": "元大高股息",
    "00878.TW": "國泰永續高股息",
    "00878": "國泰永續高股息",
    "00919.TW": "群益台灣精選高息",
    "00919": "群益台灣精選高息",
    "00715L.TW": "期街口布蘭特正2",
    "00715L": "期街口布蘭特正2",
    "NVDA": "輝達",
    "TSM": "台積電ADR",
    "AAPL": "蘋果",
    "MSFT": "微軟",
    "AMD": "超微",
    "GOOGL": "Alphabet",
    "AMZN": "亞馬遜",
    "META": "Meta",
    "TSLA": "特斯拉",
    "AVGO": "博通",
    "MU": "美光",
    "ASML": "艾司摩爾",
}

NAME_TO_SYMBOL = {
    "加權": "^TWII",
    "大盤": "^TWII",
    "台股": "^TWII",
    "加權指數": "^TWII",
    "台積電": "2330.TW",
    "鴻海": "2317.TW",
    "聯發科": "2454.TW",
    "台達電": "2308.TW",
    "廣達": "2382.TW",
    "台灣50": "0050.TW",
    "元大台灣50": "0050.TW",
    "高股息": "0056.TW",
    "元大高股息": "0056.TW",
    "國泰永續高股息": "00878.TW",
    "群益台灣精選高息": "00919.TW",
    "輝達": "NVDA",
    "蘋果": "AAPL",
    "微軟": "MSFT",
    "特斯拉": "TSLA",
    "美光": "MU",
}

MA_PERIODS = [5, 20]


def _extract_series(raw_df, col_name):
    if raw_df is None or raw_df.empty:
        return None
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    if col_name not in raw_df.columns:
        return None
    s = raw_df[col_name]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.astype(float)


def _clean_symbol_for_display(symbol):
    clean = symbol.strip().upper()
    for suff in (".TW", ".TWO"):
        if clean.endswith(suff):
            clean = clean[: -len(suff)]
    return clean


def _resolve_user_query(raw_query):
    q = raw_query.strip()
    if not q:
        return DEFAULT_TARGET_SYMBOL
    if q in NAME_TO_SYMBOL:
        return NAME_TO_SYMBOL[q]
    q_upper = q.upper()
    if q_upper in NAME_TO_SYMBOL:
        return NAME_TO_SYMBOL[q_upper]
    return q_upper


def _fetch_stock_name(symbol_or_ticker, default_code):
    if default_code in KNOWN_NAMES:
        return KNOWN_NAMES[default_code]
    if symbol_or_ticker in KNOWN_NAMES:
        return KNOWN_NAMES[symbol_or_ticker]
    try:
        t_info = yf.Ticker(symbol_or_ticker).info
        name = t_info.get("shortName") or t_info.get("longName")
        if name:
            return name.replace("TAIWAN SEMICONDUCTOR MANUFACTURING", "台積電")
    except Exception:
        pass
    return ""


def _download_with_tw_fallback(symbol, interval, period):
    resolved_input = _resolve_user_query(symbol)
    cleaned = _clean_symbol_for_display(resolved_input)
    raw = yf.download(resolved_input, interval=interval, period=period, progress=False)
    if raw is not None and not raw.empty:
        stock_name = _fetch_stock_name(resolved_input, cleaned)
        disp_title = f"{stock_name} ({cleaned})" if stock_name else cleaned
        return raw, resolved_input, cleaned, disp_title

    if cleaned.isdigit() or (len(cleaned) >= 4 and cleaned[:4].isdigit()):
        for suffix in (".TW", ".TWO"):
            candidate = f"{cleaned}{suffix}"
            cand_df = yf.download(candidate, interval=interval, period=period, progress=False)
            if cand_df is not None and not cand_df.empty:
                stock_name = _fetch_stock_name(candidate, cleaned)
                disp_title = f"{stock_name} ({cleaned})" if stock_name else cleaned
                return cand_df, candidate, cleaned, disp_title

    return None, resolved_input, cleaned, cleaned


def _align_daily_to_target(daily_df, target_index, default_map):
    target_naive = pd.to_datetime(target_index).tz_localize(None).normalize()
    src_naive = pd.to_datetime(daily_df.index).tz_localize(None).normalize()
    mapped = daily_df.copy()
    mapped.index = src_naive
    mapped = mapped[~mapped.index.duplicated(keep="last")]
    aligned = mapped.reindex(target_naive, method="ffill").ffill()

    out = {}
    for col, fallback in default_map.items():
        if col in aligned.columns:
            out[col] = aligned[col].fillna(fallback).values
        else:
            out[col] = np.full(len(target_index), fallback)
    return out


def get_market_data(ticker=DEFAULT_TARGET_SYMBOL, interval="1d", period="8y"):
    raw, resolved_symbol, clean_code, disp_title = _download_with_tw_fallback(
        ticker, interval=interval, period=period
    )
    if raw is None or raw.empty:
        raise ValueError(f"找不到「{ticker}」的行情資料，請確認股票代號或名稱是否正確。")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=raw.index)
    df["Open"] = _extract_series(raw, "Open")
    df["High"] = _extract_series(raw, "High")
    df["Low"] = _extract_series(raw, "Low")
    df["Close"] = _extract_series(raw, "Close")
    df["Volume"] = _extract_series(raw, "Volume").fillna(0.0)
    df = df.dropna(subset=["Close", "Open", "High", "Low"])
    if df.empty:
        raise ValueError(f"{disp_title} 有效 K 線資料為空。")

    df.attrs["symbol"] = resolved_symbol
    df.attrs["clean_code"] = clean_code
    df.attrs["disp_title"] = disp_title

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    for p in MA_PERIODS:
        df[f"MA{p}"] = close.rolling(p).mean()

    roll_high_250 = high.rolling(250, min_periods=60).max()
    df["Drawdown_Pct"] = ((close - roll_high_250) / roll_high_250.replace(0, np.nan) * 100).fillna(0.0)

    roll_high_60 = high.rolling(60, min_periods=20).max()
    df["Pullback12_Pct"] = ((close - roll_high_60) / roll_high_60.replace(0, np.nan) * 100).fillna(0.0)

    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df["Hist"] = (exp12 - exp26) - (exp12 - exp26).ewm(span=9, adjust=False).mean()

    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
    df["K"] = rsv.ewm(com=2, adjust=False).mean().fillna(50.0)
    df["D"] = df["K"].ewm(com=2, adjust=False).mean().fillna(50.0)
    df["J"] = (3 * df["K"] - 2 * df["D"]).fillna(50.0)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"] = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).fillna(50.0)

    try:
        vix_raw = yf.download("^VIX", interval="1d", period="max", progress=False)
        vix_s = _extract_series(vix_raw, "Close")
        if vix_s is not None and not vix_s.empty:
            vix_daily = pd.DataFrame({"VIX": vix_s})
            vix_daily["VIX_MA60"] = vix_daily["VIX"].rolling(60).mean().fillna(20.0)
            vix_daily["VIX_Bias60"] = (
                (vix_daily["VIX"] - vix_daily["VIX_MA60"]) / vix_daily["VIX_MA60"].replace(0, np.nan) * 100
            ).fillna(0.0)
            aligned_v = _align_daily_to_target(vix_daily, df.index, {"VIX": 20.0, "VIX_Bias60": 0.0})
            for k, arr in aligned_v.items():
                df[k] = arr
        else:
            df["VIX"], df["VIX_Bias60"] = 20.0, 0.0
    except Exception:
        df["VIX"], df["VIX_Bias60"] = 20.0, 0.0

    return df


def _is_bottom_rebound_bar(df, idx):
    if idx < 3:
        return False
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    close = float(row["Close"])
    prev_close = float(prev["Close"])
    ma5, ma20 = row.get("MA5", np.nan), row.get("MA20", np.nan)
    pb60 = float(row.get("Pullback12_Pct", 0.0))
    dd250 = float(row.get("Drawdown_Pct", 0.0))
    meaningful_pullback = (pb60 <= -4.0) or (dd250 <= -10.0 and pb60 <= -6.0)
    if not meaningful_pullback:
        return False
    if pd.isna(ma5) or pd.isna(ma20) or close <= float(ma5) or close <= float(ma20):
        return False
    if close <= prev_close:
        return False
    hist = float(row.get("Hist", 0.0))
    hist_prev = float(prev.get("Hist", 0.0))
    k, d = float(row.get("K", 50.0)), float(row.get("D", 50.0))
    rsi = float(row.get("RSI14", 50.0))
    return bool((hist > hist_prev and hist_prev < 0) or (k > d and k >= 35) or (rsi >= 50))


def compute_synergy_signal(df):
    if df is None or df.empty:
        raise ValueError("行情資料為空，無法計算訊號。")

    n = len(df)
    purple = np.zeros(n, dtype=bool)
    for i in range(3, n):
        purple[i] = _is_bottom_rebound_bar(df, i)

    purple_filtered = np.zeros(n, dtype=bool)
    last_purple, last_low = -10_000, np.inf
    for i in range(n):
        if purple[i]:
            low_now = float(df["Low"].iloc[i])
            new_low = low_now < last_low * 0.98
            if i - last_purple > 7 or new_low:
                purple_filtered[i] = True
                last_purple = i
                last_low = low_now if new_low else min(last_low, low_now)

    vix_s = df.get("VIX", pd.Series(20.0, index=df.index)).astype(float)
    vix_b60 = df.get("VIX_Bias60", pd.Series(0.0, index=df.index)).astype(float)
    vix_panic_peak = (vix_s >= 23.0) | (vix_b60 >= 18.0)
    vix_retreat_bar = (vix_s < vix_s.rolling(5, min_periods=1).max() * 0.96) | (vix_s < vix_s.shift(1))
    vix_turn_point = vix_panic_peak & vix_retreat_bar
    vix_near_window = vix_turn_point.rolling(11, min_periods=1, center=True).max().astype(bool)

    df["Is_VIX_Purple_Synergy"] = purple_filtered & vix_near_window.values
    return df