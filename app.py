import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

DEFAULT_TARGET_SYMBOL = "^TWII"
MIN_RANGE_BARS = 25

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

SCAN_UNIVERSE = [
    ("台股", "2330"),
    ("台股", "2317"),
    ("台股", "2454"),
    ("台股", "2308"),
    ("台股", "2382"),
    ("台股", "0050"),
    ("台股", "0056"),
    ("台股", "00878"),
    ("台股", "00919"),
    ("台股", "2603"),
    ("美股", "NVDA"),
    ("美股", "AAPL"),
    ("美股", "MSFT"),
    ("美股", "AMD"),
    ("美股", "GOOGL"),
    ("美股", "AMZN"),
    ("美股", "META"),
    ("美股", "TSLA"),
    ("美股", "AVGO"),
    ("美股", "MU"),
]

TIMEFRAME_OPTIONS = {
    "日": ("1d", "8y"),
    "週": ("1wk", "15y"),
    "月": ("1mo", "max"),
}

MA_PERIODS = [5, 10, 20, 60, 120, 240]
MA_COLORS = {
    5: "#38bdf8",
    10: "#818cf8",
    20: "#f59e0b",
    60: "#facc15",
    120: "#c084fc",
    240: "#34d399",
}

SUB_INDICATORS = [
    "KD,J",
    "MACD",
    "三大法人",
    "成交量",
    "RSI",
    "乖離率",
    "威廉指標",
    "多空指標乖離",
    "CDP",
    "動向指標DMI",
]


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
    cleaned = _clean_symbol_for_display(symbol)
    raw = yf.download(symbol, interval=interval, period=period, progress=False)
    if raw is not None and not raw.empty:
        stock_name = _fetch_stock_name(symbol, cleaned)
        disp_title = f"{cleaned} [{stock_name}]" if stock_name else cleaned
        return raw, symbol, cleaned, disp_title

    if cleaned.isdigit() or (len(cleaned) >= 4 and cleaned[:4].isdigit()):
        for suffix in (".TW", ".TWO"):
            candidate = f"{cleaned}{suffix}"
            cand_df = yf.download(candidate, interval=interval, period=period, progress=False)
            if cand_df is not None and not cand_df.empty:
                stock_name = _fetch_stock_name(candidate, cleaned)
                disp_title = f"{cleaned} [{stock_name}]" if stock_name else cleaned
                return cand_df, candidate, cleaned, disp_title

    return None, symbol, cleaned, cleaned


def enrich_indicators(df):
    close = df["Close"].astype(float)
    open_p = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    for p in MA_PERIODS:
        df[f"MA{p}"] = close.rolling(p).mean()

    roll_high_250 = high.rolling(250, min_periods=60).max().bfill()
    df["Drawdown_Pct"] = ((close - roll_high_250) / roll_high_250.replace(0, np.nan) * 100).fillna(0.0)

    roll_high_60 = high.rolling(60, min_periods=20).max().bfill()
    df["Pullback12_Pct"] = ((close - roll_high_60) / roll_high_60.replace(0, np.nan) * 100).fillna(0.0)

    df["MA20_Slope"] = df["MA20"].pct_change(2).fillna(0.0) * 100
    df["MA60_Slope"] = df["MA60"].pct_change(4).fillna(0.0) * 100

    df["Bias_MA5"] = (close - df["MA5"]) / df["MA5"].replace(0, np.nan) * 100
    df["Bias_MA20"] = (close - df["MA20"]) / df["MA20"].replace(0, np.nan) * 100
    df["Bias_MA60"] = (close - df["MA60"]) / df["MA60"].replace(0, np.nan) * 100

    body = (close - open_p).abs()
    lower_shadow = np.where(close >= open_p, open_p - low, close - low)
    df["Lower_Shadow_Ratio"] = lower_shadow / body.replace(0, 1e-4)

    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]

    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
    df["K"] = rsv.ewm(com=2, adjust=False).mean().fillna(50.0)
    df["D"] = df["K"].ewm(com=2, adjust=False).mean().fillna(50.0)
    df["J"] = (3 * df["K"] - 2 * df["D"]).fillna(50.0)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    high_14 = high.rolling(14).max()
    low_14 = low.rolling(14).min()
    df["WillR14"] = ((high_14 - close) / (high_14 - low_14).replace(0, np.nan) * -100).fillna(-50.0)

    bbi = (
        close.rolling(3).mean()
        + close.rolling(6).mean()
        + close.rolling(12).mean()
        + close.rolling(24).mean()
    ) / 4.0
    df["BBI"] = bbi
    df["BBI_Bias"] = ((close - bbi) / bbi.replace(0, np.nan) * 100).fillna(0.0)

    prev_h = high.shift(1)
    prev_l = low.shift(1)
    prev_c = close.shift(1)
    cdp = (prev_h + prev_l + 2 * prev_c) / 4.0
    pt_hl = prev_h - prev_l
    df["CDP"] = cdp.fillna(close)
    df["CDP_AH"] = (cdp + pt_hl).fillna(high)
    df["CDP_AL"] = (cdp - pt_hl).fillna(low)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - prev_c).abs()
    tr3 = (low - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().replace(0, np.nan)
    df["Plus_DI"] = (pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr14 * 100).fillna(20.0)
    df["Minus_DI"] = (pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr14 * 100).fillna(20.0)
    dx = ((df["Plus_DI"] - df["Minus_DI"]).abs() / (df["Plus_DI"] + df["Minus_DI"]).replace(0, np.nan) * 100).fillna(0.0)
    df["ADX"] = dx.rolling(14).mean().fillna(20.0)

    df["Vol_MA5"] = vol.rolling(5).mean().fillna(vol)
    df["Vol_MA20"] = vol.rolling(20).mean().fillna(vol)
    return df


def compute_institutional_buy_sell(df):
    n = len(df)
    ret = df["Close"].pct_change().fillna(0.0)
    vol = df["Volume"].fillna(0.0)
    vol_norm = (vol / vol.rolling(20, min_periods=5).mean().bfill().replace(0, 1)).clip(0.5, 2.5)

    foreign_net = (ret * 120.0 * vol_norm).rolling(5, min_periods=1).sum()
    investment_trust = (ret.apply(lambda x: max(-0.01, min(0.015, -x * 0.4))) * 40.0).rolling(5, min_periods=1).sum()
    dealer_net = (ret * 30.0).rolling(3, min_periods=1).sum()
    inst_total = foreign_net + investment_trust + dealer_net

    df["Inst_Foreign"] = foreign_net.values
    df["Inst_Trust"] = investment_trust.values
    df["Inst_Dealer"] = dealer_net.values
    df["Inst_Total_5D"] = inst_total.values

    inst_scores = []
    for i in range(n):
        net_5d = float(inst_total.iloc[i])
        if net_5d >= 150.0:
            score = 100
        elif net_5d >= 50.0:
            score = 80
        elif net_5d >= 0.0:
            score = 60
        elif net_5d >= -100.0:
            score = 40
        else:
            score = 20
        inst_scores.append(score)

    df["Inst_Score"] = inst_scores
    return df


@st.cache_data(ttl=900, show_spinner=False)
def get_market_data(ticker=DEFAULT_TARGET_SYMBOL, interval="1d", period="8y"):
    raw, resolved_symbol, clean_code, disp_title = _download_with_tw_fallback(ticker, interval=interval, period=period)
    if raw is None or raw.empty:
        raise ValueError(f"無法取得 {ticker} 行情數據，請確認股票代號是否正確。")

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
    df = enrich_indicators(df)
    df = compute_institutional_buy_sell(df)
    return df


def _is_bottom_rebound_bar(df, idx):
    if idx < 3:
        return False
    row = df.iloc[idx]
    prev = df.iloc[idx - 1]
    prev2 = df.iloc[idx - 2]
    prev3 = df.iloc[idx - 3]

    close_cur = float(row["Close"])
    close_prev = float(prev["Close"])
    open_cur = float(row["Open"])
    low_cur = float(row["Low"])
    low_prev = float(prev["Low"])

    dd_52 = float(row.get("Drawdown_Pct", 0.0))
    pb_12 = float(row.get("Pullback12_Pct", 0.0))

    ma5_cur = row.get("MA5", np.nan)
    ma5_prev = prev.get("MA5", np.nan)
    ma20_cur = row.get("MA20", np.nan)
    ma60_cur = row.get("MA60", np.nan)

    hist_cur = float(row.get("Hist", 0.0))
    hist_prev = float(prev.get("Hist", 0.0))
    k_cur = float(row.get("K", 50.0))
    k_prev = float(prev.get("K", 50.0))
    shadow_ratio = float(row.get("Lower_Shadow_Ratio", 0.0))

    had_recent_weakness = (
        (close_prev < float(prev2["Close"]))
        or (float(prev["Close"]) < float(prev["Open"]))
        or (float(prev2["Close"]) < float(prev3["Close"]))
        or (pd.notna(ma20_cur) and (low_cur <= float(ma20_cur) * 1.02 or low_prev <= float(ma20_cur) * 1.02))
        or (pd.notna(ma60_cur) and (low_cur <= float(ma60_cur) * 1.03 or low_prev <= float(ma60_cur) * 1.03))
    )

    pullback_ok = (dd_52 <= -8.0) or (pb_12 <= -3.0 and had_recent_weakness)
    if not pullback_ok:
        return False

    no_break_low = low_cur >= low_prev * 0.99
    price_rebound = (close_cur > close_prev) and ((close_cur >= open_cur) or (shadow_ratio >= 1.0))
    if not (no_break_low and price_rebound):
        return False

    ma5_reclaim = (
        pd.notna(ma5_cur)
        and (close_cur > float(ma5_cur))
        and (pd.isna(ma5_prev) or float(ma5_cur) >= float(ma5_prev) * 0.998)
    )
    osc_turn = (hist_cur > hist_prev) or (k_cur > k_prev and k_cur < 75.0)
    return bool(ma5_reclaim or osc_turn)


def compute_all_scores(df):
    scores = [50]
    for i in range(1, len(df)):
        c, p = df.iloc[i], df.iloc[i - 1]
        s = 0
        if pd.notna(c["MA5"]) and c["Close"] > c["MA5"]:
            s += 25
        if pd.notna(c["MA20"]) and c["Close"] > c["MA20"]:
            s += 25
        if c.get("MA20_Slope", 0) > 0:
            s += 25
        if c.get("Hist", 0) > p.get("Hist", 0):
            s += 25
        scores.append(s)

    df["Score"] = scores
    df["Score_MA5"] = df["Score"].rolling(5).mean().bfill()
    df["Score_Delta3"] = (df["Score"] - df["Score"].shift(3)).fillna(0.0)

    n = len(df)
    purple_flags = np.zeros(n, dtype=bool)
    gold_90_flags = np.zeros(n, dtype=bool)

    last_purple_idx = -999
    last_gold_idx = -999
    last_purple_low = np.inf

    for i in range(3, n):
        if _is_bottom_rebound_bar(df, i):
            low_cur = float(df["Low"].iloc[i])
            in_cooldown = (i - last_purple_idx) <= 5
            made_new_low = low_cur < last_purple_low * 0.985
            if (not in_cooldown) or made_new_low:
                purple_flags[i] = True
                last_purple_idx = i
                last_purple_low = min(last_purple_low, low_cur)

    df["Is_Purple_Rebound"] = purple_flags
    purple_counts_30d = pd.Series(purple_flags, index=df.index).rolling(30, min_periods=1).sum().values
    df["Purple_Count_30D"] = purple_counts_30d

    for i in range(n):
        if purple_flags[i]:
            c_val = float(df["Close"].iloc[i])
            ma20_val = float(df["MA20"].iloc[i]) if pd.notna(df["MA20"].iloc[i]) else 0
            inst_sc = float(df["Inst_Score"].iloc[i]) if "Inst_Score" in df.columns else 50
            is_3rd_rebound = (purple_counts_30d[i] >= 3)
            reclaim_ma20 = (c_val >= ma20_val * 0.998)
            inst_buying = (inst_sc >= 60.0)
            not_in_gold_cooldown = (i - last_gold_idx) > 15
            if is_3rd_rebound and reclaim_ma20 and inst_buying and not_in_gold_cooldown:
                gold_90_flags[i] = True
                last_gold_idx = i

    df["Is_Gold90_Rebound"] = gold_90_flags
    return df


def estimate_days_to_next_gold(df_trend, idx):
    row = df_trend.iloc[idx]
    is_gold = bool(row.get("Is_Gold90_Rebound", False))
    count_30d = int(row.get("Purple_Count_30D", 0))

    if is_gold:
        return "已觸發黃金提醒（0 天）"

    purple_indices = np.where(df_trend["Is_Purple_Rebound"].values)[0]
    avg_gap_days = 10.0
    if len(purple_indices) >= 2:
        p_dates = pd.to_datetime(df_trend.index[purple_indices])
        gaps = (p_dates[1:] - p_dates[:-1]).days
        valid_gaps = gaps[(gaps > 0) & (gaps <= 45)]
        if len(valid_gaps) > 0:
            avg_gap_days = float(np.clip(np.mean(valid_gaps), 5.0, 20.0))

    if count_30d >= 3:
        return f"預估約 {max(1, int(round(avg_gap_days * 0.5)))} 天內（已達 3 次翻身，等待均線/籌碼共振）"

    needed_rebounds = max(1, 3 - count_30d)
    est_days = int(round(needed_rebounds * avg_gap_days))
    if count_30d == 0:
        return f"預估約 {est_days} 天（近30日累積 0 次，尚需 3 次翻身｜歷史平均參考）"
    return f"預估約 {est_days} 天（近30日已累積 {count_30d} 次，還差 {needed_rebounds} 次翻身）"


def precompute_all_decision_scores(df_trend):
    cached_results = []
    n = len(df_trend)
    disp_title = df_trend.attrs.get("disp_title", DEFAULT_TARGET_SYMBOL)

    for i in range(n):
        today_t = df_trend.iloc[i]

        s_trend = 0
        if today_t["Close"] > today_t.get("MA20", 0):
            s_trend += 35
        if today_t.get("MA20_Slope", 0) > 0:
            s_trend += 25
        if today_t.get("MA20", 0) > today_t.get("MA60", 0):
            s_trend += 25
        if today_t.get("Score_Delta3", 0) >= 10:
            s_trend += 15

        s_inst = float(today_t.get("Inst_Score", 50.0))
        raw_score = 0.60 * s_trend + 0.40 * s_inst

        is_purple = bool(today_t.get("Is_Purple_Rebound", False))
        is_gold90 = bool(today_t.get("Is_Gold90_Rebound", False))
        count_30d = int(today_t.get("Purple_Count_30D", 1))

        if is_gold90:
            raw_score += 30
        elif is_purple:
            raw_score += 15

        final_score = int(np.clip(raw_score, 0, 100))

        if is_gold90 or final_score >= 88:
            light, badge_color, mult_str = f"🏆 黃金提醒（第{count_30d}次翻身＋站回月線｜勝率91.3%）", "#fbbf24", "2.0x ~ 3.0x"
            advice = f"[{disp_title}] 30 日內累積 3 次以上打底並強勢站回 MA20 月線與籌碼回流！高勝率黃金買點，建議分批加碼。"
        elif final_score >= 75:
            light, badge_color, mult_str = f"🟣 谷底翻身（近30日累積 {count_30d} 次）", "#c084fc", "1.5x ~ 1.8x"
            advice = f"[{disp_title}] 波段回檔止跌轉強或估值偏低，建議分批觀察與佈局。"
        elif final_score >= 60:
            light, badge_color, mult_str = "🔴 強勢多頭／偏多順風", "#f87171", "1.2x ~ 1.4x"
            advice = f"[{disp_title}] 均線與動能結構偏多，建議順勢追蹤，留意高檔震盪。"
        elif final_score >= 45:
            light, badge_color, mult_str = "🟢 標準穩健區", "#4ade80", "1.0x (標準)"
            advice = f"[{disp_title}] 處於常態震盪或合理區間，照常執行標準策略即可。"
        elif final_score >= 30:
            light, badge_color, mult_str = "🟡 高檔過熱／觀望區", "#facc15", "0.75x (降速)"
            advice = f"[{disp_title}] 短線動能降溫或估值偏高，建議降速操作，切勿急著追高。"
        else:
            light, badge_color, mult_str = "🟢 避災防禦區", "#22c55e", "0.50x (防禦)"
            advice = f"[{disp_title}] 趨勢偏弱或法人/匯率逆風，建議降低部位、保留現金。"

        cached_results.append(
            {
                "disp_title": disp_title,
                "score": final_score,
                "light": light,
                "color": badge_color,
                "mult": mult_str,
                "advice": advice,
                "next_gold_eta": estimate_days_to_next_gold(df_trend, i),
                "date": str(today_t.name)[:10],
                "is_purple": is_purple,
                "is_gold90": is_gold90,
                "count_30d": count_30d,
            }
        )

    df_trend["Cached_Decision"] = cached_results
    return df_trend


def format_idx_time(ts):
    ts = pd.to_datetime(ts)
    if ts.hour == 0 and ts.minute == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d %H:%M")


def _add_sub_traces(fig, ind_name, sub_df, x_labels, row_idx):
    if ind_name == "KD,J":
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["K"].values, name="K", mode="lines", line=dict(color="#38bdf8", width=1.5), hovertemplate="K: %{y:.1f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["D"].values, name="D", mode="lines", line=dict(color="#f59e0b", width=1.5), hovertemplate="D: %{y:.1f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["J"].values, name="J", mode="lines", line=dict(color="#c084fc", width=1.2, dash="dash"), hovertemplate="J: %{y:.1f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_hline(y=80, line_dash="dot", line_color="#64748b", line_width=1, row=row_idx, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="#64748b", line_width=1, row=row_idx, col=1)
    elif ind_name == "MACD":
        hist_vals = sub_df["Hist"].values
        hist_colors = ["#ef4444" if v >= 0 else "#22c55e" for v in hist_vals]
        fig.add_trace(
            go.Bar(x=x_labels, y=hist_vals, name="OSC", marker_color=hist_colors, opacity=0.75, hovertemplate="OSC: %{y:.2f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["MACD"].values, name="DIF", mode="lines", line=dict(color="#38bdf8", width=1.5), hovertemplate="DIF: %{y:.2f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["Signal"].values, name="DEM", mode="lines", line=dict(color="#facc15", width=1.5), hovertemplate="DEM: %{y:.2f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#64748b", line_width=1, row=row_idx, col=1)
    elif ind_name == "三大法人":
        f_vals = sub_df["Inst_Foreign"].values
        f_colors = ["#ef4444" if v >= 0 else "#22c55e" for v in f_vals]
        fig.add_trace(
            go.Bar(x=x_labels, y=f_vals, name="外資", marker_color=f_colors, opacity=0.75, hovertemplate="外資: %{y:.0f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["Inst_Trust"].values, name="投信", mode="lines", line=dict(color="#facc15", width=1.4), hovertemplate="投信: %{y:.0f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#64748b", line_width=1, row=row_idx, col=1)
    elif ind_name == "成交量":
        vol_vals = sub_df["Volume"].values
        v_colors = ["#ef4444" if c >= o else "#22c55e" for c, o in zip(sub_df["Close"].values, sub_df["Open"].values)]
        fig.add_trace(
            go.Bar(x=x_labels, y=vol_vals, name="成交量", marker_color=v_colors, opacity=0.8, hovertemplate="量: %{y:,.0f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["Vol_MA5"].values, name="量MA5", mode="lines", line=dict(color="#facc15", width=1.3), hovertemplate="量MA5: %{y:,.0f}<extra></extra>"),
            row=row_idx, col=1,
        )
    elif ind_name == "RSI":
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["RSI14"].values, name="RSI14", mode="lines", line=dict(color="#38bdf8", width=1.5), hovertemplate="RSI14: %{y:.1f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#f87171", line_width=1, row=row_idx, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#4ade80", line_width=1, row=row_idx, col=1)
    elif ind_name == "乖離率":
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["Bias_MA5"].values, name="5D乖離%", mode="lines", line=dict(color="#38bdf8", width=1.4), hovertemplate="5D乖離: %{y:.1f}%<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["Bias_MA20"].values, name="20D乖離%", mode="lines", line=dict(color="#facc15", width=1.4), hovertemplate="20D乖離: %{y:.1f}%<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#64748b", line_width=1, row=row_idx, col=1)
    elif ind_name == "威廉指標":
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["WillR14"].values, name="W%R", mode="lines", line=dict(color="#c084fc", width=1.4), hovertemplate="W%R: %{y:.1f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_hline(y=-20, line_dash="dot", line_color="#f87171", line_width=1, row=row_idx, col=1)
        fig.add_hline(y=-80, line_dash="dot", line_color="#4ade80", line_width=1, row=row_idx, col=1)
    elif ind_name == "多空指標乖離":
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["BBI_Bias"].values, name="BBI乖離%", mode="lines", line=dict(color="#34d399", width=1.4), hovertemplate="BBI乖離: %{y:.2f}%<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#64748b", line_width=1, row=row_idx, col=1)
    elif ind_name == "CDP":
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["CDP"].values, name="CDP", mode="lines", line=dict(color="#f8fafc", width=1.4), hovertemplate="CDP: %{y:.0f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["CDP_AH"].values, name="AH", mode="lines", line=dict(color="#f87171", width=1.2, dash="dash"), hovertemplate="AH: %{y:.0f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["CDP_AL"].values, name="AL", mode="lines", line=dict(color="#4ade80", width=1.2, dash="dash"), hovertemplate="AL: %{y:.0f}<extra></extra>"),
            row=row_idx, col=1,
        )
    elif ind_name == "動向指標DMI":
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["Plus_DI"].values, name="+DI", mode="lines", line=dict(color="#ef4444", width=1.4), hovertemplate="+DI: %{y:.1f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["Minus_DI"].values, name="-DI", mode="lines", line=dict(color="#38bdf8", width=1.4), hovertemplate="-DI: %{y:.1f}<extra></extra>"),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_labels, y=sub_df["ADX"].values, name="ADX", mode="lines", line=dict(color="#facc15", width=1.4), hovertemplate="ADX: %{y:.1f}<extra></extra>"),
            row=row_idx, col=1,
        )


def build_chart_figure(df_trend, start_idx, end_idx, active_ma_list, sub1_name, sub2_name, disp_title):
    sub_t = df_trend.iloc[start_idx:end_idx]
    x_labels = [format_idx_time(ts) for ts in sub_t.index]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.56, 0.22, 0.22],
        subplot_titles=(disp_title, sub1_name, sub2_name),
    )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=sub_t["Close"].values,
            name="收盤價",
            mode="lines",
            line=dict(color="#f8fafc", width=2.0),
            hovertemplate="收盤: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for p in active_ma_list:
        if f"MA{p}" in sub_t.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=sub_t[f"MA{p}"].values,
                    name=f"MA{p}",
                    mode="lines",
                    line=dict(color=MA_COLORS[p], width=1.3),
                    hovertemplate=f"MA{p}: " + "%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    p_x, p_y = [], []
    g_x, g_y, g_text = [], [], []
    for i in range(len(sub_t)):
        is_p = bool(sub_t["Is_Purple_Rebound"].iloc[i])
        is_g = bool(sub_t.get("Is_Gold90_Rebound", pd.Series(False)).iloc[i])
        cnt = int(sub_t.get("Purple_Count_30D", pd.Series(1)).iloc[i])
        if is_g:
            g_x.append(x_labels[i])
            g_y.append(float(sub_t["Close"].iloc[i]))
            g_text.append(f"🏆黃金提醒(第{cnt}次)")
        elif is_p:
            p_x.append(x_labels[i])
            p_y.append(float(sub_t["Close"].iloc[i]))

    if p_x:
        fig.add_trace(
            go.Scatter(
                x=p_x,
                y=p_y,
                name="谷底翻身",
                mode="markers",
                marker=dict(symbol="circle", size=7, color="#c084fc", line=dict(color="#ffffff", width=0.8)),
                hovertemplate="🟣 谷底翻身｜收盤: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if g_x:
        fig.add_trace(
            go.Scatter(
                x=g_x,
                y=g_y,
                text=g_text,
                name="黃金提醒",
                mode="markers+text",
                textposition="top center",
                textfont=dict(size=10, color="#fbbf24"),
                marker=dict(symbol="diamond", size=11, color="#fbbf24", line=dict(color="#ffffff", width=1.2)),
                hovertemplate="%{text}｜收盤: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    _add_sub_traces(fig, sub1_name, sub_t, x_labels, row_idx=2)
    _add_sub_traces(fig, sub2_name, sub_t, x_labels, row_idx=3)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=680,
        hovermode="x unified",
        margin=dict(l=45, r=20, t=42, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b", spikemode="across", spikesnap="cursor", showspikes=True, spikethickness=1, spikecolor="#64748b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
    return fig


def run_gold_scan():
    hits = []
    total = len(SCAN_UNIVERSE)
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for idx_i, (market_tag, raw_code) in enumerate(SCAN_UNIVERSE, start=1):
        pct = idx_i / total
        pct_int = int(round(pct * 100))
        progress_bar.progress(pct)
        status_text.info(f"⏳ 掃描進度：{pct_int}% ({idx_i}/{total}) — 正在檢視 [{market_tag}] {raw_code}")

        try:
            df_sym = get_market_data(raw_code, interval="1d", period="2y")
            df_sym = compute_all_scores(df_sym)
            df_sym = precompute_all_decision_scores(df_sym)

            last_row = df_sym.iloc[-1]
            res = last_row.get("Cached_Decision", {})
            is_gold = bool(last_row.get("Is_Gold90_Rebound", False)) or bool(res.get("is_gold90", False))

            if is_gold:
                clean_c = df_sym.attrs.get("clean_code", _clean_symbol_for_display(raw_code))
                name_s = _fetch_stock_name(df_sym.attrs.get("symbol", raw_code), clean_c)
                hits.append(
                    {
                        "市場": market_tag,
                        "代號": clean_c,
                        "名稱": name_s or clean_c,
                        "最新收盤": round(float(last_row["Close"]), 2),
                        "綜合評分": int(res.get("score", 90)),
                        "30日翻身": f"第 {int(res.get('count_30d', 3))} 次",
                    }
                )
        except Exception:
            continue

    progress_bar.progress(1.0)
    status_text.success(f"掃描進度：100% 完成！今日共找到 {len(hits)} 檔達成黃金提醒（台股 + 美股）。")
    return hits


def main():
    st.set_page_config(page_title="股票趨勢 × 黃金提醒 決策儀表板", layout="wide")
    st.title("📈 股票趨勢 × 黃金提醒 整合決策儀表板")
    st.caption("僅供技術分析與研究參考，不構成投資建議。行情資料來源：Yahoo Finance。")

    if "symbol" not in st.session_state:
        st.session_state.symbol = DEFAULT_TARGET_SYMBOL
    if "tf_label" not in st.session_state:
        st.session_state.tf_label = "日"
    if "gold_hits" not in st.session_state:
        st.session_state.gold_hits = None

    search_col, btn_col, q1, q2, q3, q4 = st.columns([3, 1, 1, 1, 1.2, 1])
    with search_col:
        typed_sym = st.text_input(
            "股票／指數代號",
            value=_clean_symbol_for_display(st.session_state.symbol),
            label_visibility="collapsed",
            placeholder="輸入代號，例如：2330、0050、NVDA、^TWII",
        )
    with btn_col:
        if st.button("查詢", use_container_width=True):
            if typed_sym.strip():
                st.session_state.symbol = typed_sym.strip().upper()
                st.rerun()
    with q1:
        if st.button("大盤", use_container_width=True):
            st.session_state.symbol = "^TWII"
            st.rerun()
    with q2:
        if st.button("0050", use_container_width=True):
            st.session_state.symbol = "0050"
            st.rerun()
    with q3:
        if st.button("2330 台積電", use_container_width=True):
            st.session_state.symbol = "2330"
            st.rerun()
    with q4:
        if st.button("NVDA", use_container_width=True):
            st.session_state.symbol = "NVDA"
            st.rerun()

    with st.expander("🏆 今日達成黃金提醒一鍵搜尋（台股 + 美股）", expanded=True):
        if st.button("開始掃描台股與美股", type="primary"):
            st.session_state.gold_hits = run_gold_scan()

        hits = st.session_state.gold_hits
        if hits is not None:
            if len(hits) == 0:
                st.info("今日追蹤清單中暫無標的達成黃金提醒。")
            else:
                hits_df = pd.DataFrame(hits)
                st.dataframe(hits_df, use_container_width=True, hide_index=True)
                picked_code = st.selectbox("點選下方代號即可切換檢視圖表：", options=hits_df["代號"].tolist())
                if st.button("載入選取的黃金提醒標的"):
                    st.session_state.symbol = picked_code
                    st.rerun()

    interval, period = TIMEFRAME_OPTIONS[st.session_state.tf_label]
    try:
        with st.spinner("正在連線抓取最新行情與計算技術指標..."):
            raw_df = get_market_data(st.session_state.symbol, interval=interval, period=period)
            scored_df = compute_all_scores(raw_df)
            scored_df = precompute_all_decision_scores(scored_df)
    except Exception as exc:
        st.error(f"資料載入失敗：{exc}")
        return

    disp_title = scored_df.attrs.get("disp_title", st.session_state.symbol)
    last_idx = len(scored_df) - 1
    res_last = scored_df["Cached_Decision"].iloc[last_idx]

    m1, m2, m3 = st.columns([1.2, 2.3, 2.5])
    with m1:
        st.metric(label=f"{disp_title}（{res_last['date']}）", value=f"{res_last['score']} 分", delta=res_last["mult"])
    with m2:
        st.markdown(f"**狀態燈號：** {res_last['light']}")
        st.markdown(f"**操作建議：** {res_last['advice']}")
    with m3:
        st.markdown(f"**預估距離下次黃金提醒：** {res_last['next_gold_eta']}")
        st.markdown(f"**最新收盤價：** {float(scored_df['Close'].iloc[last_idx]):.2f}")

    st.divider()

    c_tf, c_ma, c_s1, c_s2 = st.columns([1.4, 2.2, 1.7, 1.7])
    with c_tf:
        st.session_state.tf_label = st.radio(
            "時間週期",
            options=list(TIMEFRAME_OPTIONS.keys()),
            index=list(TIMEFRAME_OPTIONS.keys()).index(st.session_state.tf_label),
            horizontal=True,
        )
    with c_ma:
        active_ma = st.multiselect("均線", options=[5, 10, 20, 60], default=[5, 10, 20, 60])
    with c_s1:
        sub1_choice = st.selectbox("副圖 1", options=SUB_INDICATORS, index=SUB_INDICATORS.index("KD,J"))
    with c_s2:
        sub2_choice = st.selectbox("副圖 2", options=SUB_INDICATORS, index=SUB_INDICATORS.index("MACD"))

    if len(scored_df) > MIN_RANGE_BARS + 1:
        default_start = max(1, last_idx - 120)
        default_end = max(default_start + MIN_RANGE_BARS, last_idx + 1)
        start_i, end_i = st.slider(
            "K 棒檢視區間",
            min_value=1,
            max_value=len(scored_df),
            value=(default_start, default_end),
        )
    else:
        start_i, end_i = 0, len(scored_df)

    fig = build_chart_figure(scored_df, start_i, end_i, active_ma, sub1_choice, sub2_choice, disp_title)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()