from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np

from market_engine import get_market_data, compute_synergy_signal

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Trend Signal API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TF_MAP = {
    "1d": ("1d", "2y"),
    "1wk": ("1wk", "5y"),
    "1mo": ("1mo", "10y"),
}


@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/chart")
def api_chart(symbol: str = Query("^TWII"), tf: str = Query("1d")):
    interval, period = TF_MAP.get(tf, ("1d", "2y"))
    try:
        df = compute_synergy_signal(get_market_data(symbol, interval=interval, period=period))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    sub = df.tail(120)
    items = []
    for ts, row in sub.iterrows():
        is_buy = bool(row.get("Is_VIX_Purple_Synergy", False))
        vix_val = float(row.get("VIX", 20.0))
        k_val = float(row.get("K", 50.0))
        d_val = float(row.get("D", 50.0))
        pb60 = float(row.get("Pullback12_Pct", 0.0))
        dd250 = float(row.get("Drawdown_Pct", 0.0))
        vix_b60 = float(row.get("VIX_Bias60", 0.0))

        vix_hot = (vix_val >= 23.0) or (vix_b60 >= 18.0)
        pullback_ready = (pb60 <= -4.0) or (dd250 <= -10.0 and pb60 <= -6.0)

        if is_buy:
            eta, bg, fg = "🟢 【今日出現翻身買點！】恐慌已過且股價站穩均線", "#064e3b", "#6ee7b7"
        elif vix_hot and pullback_ready:
            est = 1 if k_val >= 30 else 3
            eta, bg, fg = f"🟡 【黃金觀察期】預估約 {est}~5 天內出現買點（恐慌與回檔皆已到位，待站穩均線）", "#451a03", "#fcd34d"
        elif vix_hot or pb60 <= -3.0:
            est = max(5, min(20, int(round((23.0 - min(vix_val, 23.0)) * 2 + 6))))
            eta, bg, fg = f"🟠 【波動升溫中】預估距離下次買點約 {est} 天（持續觀察）", "#1e293b", "#fdba74"
        else:
            eta, bg, fg = "⚪ 【平常心持有】目前恐慌與回檔尚未同步到位", "#1e293b", "#e2e8f0"

        items.append({
            "date": ts.strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
            "ma5": round(float(row["MA5"]) if not np.isnan(row["MA5"]) else float(row["Close"]), 2),
            "ma20": round(float(row["MA20"]) if not np.isnan(row["MA20"]) else float(row["Close"]), 2),
            "k": round(k_val, 1),
            "d": round(d_val, 1),
            "vix": round(vix_val, 1),
            "is_buy": is_buy,
            "buy_eta": eta,
            "signal_bg": bg,
            "signal_fg": fg,
        })

    return {
        "title": f"{df.attrs.get('disp_title', symbol)} 趨勢看盤",
        "items": items,
    }