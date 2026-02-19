"""
==========================================================
  Whale Radar Web — FastAPI Backend
  POST /api/login   パスワード認証 → トークン発行
  GET  /api/alerts  最新スキャン結果 (JSON)
  GET  /api/status  スキャン状態・次回実行時刻
  GET  /            index.html 配信
==========================================================
"""

import asyncio
import datetime
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from scanner import (
    check_insider_trades,
    check_options_volume,
    check_spoofing_from_chain,
    format_insider_alert_jp,
    format_sentiment_jp,
    format_spoofing_history_jp,
    format_whale_alert_history_jp,
    get_sentiment_direction,
    get_stock_price,
    make_fingerprint,
)

# ============================================================
# .env 読み込み
# ============================================================
load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# ============================================================
# 設定 (すべて .env から)
# ============================================================
WATCHLIST = [s.strip() for s in _env("WATCHLIST", "NVDA,AMD,SMCI,PLTR,ARM,MSTR,COIN,MARA,TSLA,SOFI,RKLB,GRRR,DJT,HOOD,AAPL,AMZN,META,GOOGL,MSFT,NFLX").split(",") if s.strip()]
OPTION_VOLUME_THRESHOLD = _env_float("OPTION_VOLUME_THRESHOLD", 3.0)
SPOOFING_MULTIPLIER = _env_float("SPOOFING_MULTIPLIER", 10.0)
MIN_SPOOFING_CONTRACTS = _env_int("MIN_SPOOFING_CONTRACTS", 500)
SCAN_INTERVAL = _env_int("SCAN_INTERVAL", 300)
SLEEP_BETWEEN_TICKERS = _env_float("SLEEP_BETWEEN_TICKERS", 2.0)
SLEEP_BETWEEN_REQUESTS = _env_float("SLEEP_BETWEEN_REQUESTS", 1.5)
CORS_ORIGINS = [s.strip() for s in _env("CORS_ORIGINS", "*").split(",") if s.strip()]
WEBHOOK_URL = _env("WEBHOOK_URL", "")
PORT = _env_int("PORT", 8000)
APP_PASSWORD = _env("APP_PASSWORD", "")

# ============================================================
# 認証トークン (サーバー起動ごとにシークレットを生成)
# ============================================================
_SERVER_SECRET = secrets.token_hex(32)


def _make_token(password: str) -> str:
    """パスワードからHMACトークンを生成する。"""
    return hmac.new(
        _SERVER_SECRET.encode(), password.encode(), hashlib.sha256
    ).hexdigest()


# 起動時に正しいトークンを計算 (APP_PASSWORD が空なら認証なし)
_VALID_TOKEN = _make_token(APP_PASSWORD) if APP_PASSWORD else ""


def _check_auth(request: Request) -> bool:
    """リクエストの認証トークンを検証する。
    APP_PASSWORD が空 → 常にTrue (認証なし)。"""
    if not APP_PASSWORD:
        return True
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return hmac.compare_digest(auth[7:], _VALID_TOKEN)
    return False

# ============================================================
# ロガー
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("whale_radar_web")

# ============================================================
# グローバルステート
# ============================================================
MAX_HISTORY = 50

scan_state: dict = {
    "scan_count": 0,
    "last_scan_at": None,
    "next_scan_at": None,
    "is_scanning": False,
    "watchlist_count": len(WATCHLIST),
}

# 直近50件のアラート履歴
alert_history: deque[dict] = deque(maxlen=MAX_HISTORY)

# 最新スキャンの銘柄別データ
latest_tickers: dict = {}

# 最新集計
latest_summary: dict = {
    "total_alerts": 0,
    "whale_count": 0,
    "spoofing_count": 0,
    "insider_count": 0,
}

# 重複抑制キャッシュ
last_fingerprints: dict[str, set[str]] = {}


# ============================================================
# スキャン実行 (1サイクル)
# ============================================================

def run_full_scan() -> None:
    """全銘柄をスキャンして結果をグローバルステートに格納する。"""
    global latest_tickers, latest_summary, last_fingerprints

    tickers_data: dict = {}
    cycle_whale: list[dict] = []
    cycle_spoofing: list[dict] = []
    cycle_insider: list[dict] = []

    scan_start = time.time()

    for idx, ticker in enumerate(WATCHLIST, start=1):
        logger.info(f"  [{idx}/{len(WATCHLIST)}] {ticker} スキャン中...")
        ticker_info: dict = {"price": None, "alerts": []}

        # --- 株価取得 ---
        try:
            price_data = get_stock_price(ticker)
            if price_data:
                ticker_info["price"] = price_data
        except Exception as e:
            logger.warning(f"{ticker} 株価取得エラー: {e}")

        # --- A) オプション出来高 ---
        try:
            whale_alerts = check_options_volume(ticker, OPTION_VOLUME_THRESHOLD)
        except Exception as e:
            whale_alerts = []
            logger.warning(f"{ticker} オプションエラー: {e}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

        # --- B) 異常板検知 ---
        try:
            spoofing_alerts = check_spoofing_from_chain(
                ticker, SPOOFING_MULTIPLIER, MIN_SPOOFING_CONTRACTS
            )
        except Exception as e:
            spoofing_alerts = []
            logger.warning(f"{ticker} 異常板エラー: {e}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

        # --- C) インサイダー ---
        try:
            insider_alerts = check_insider_trades(ticker)
        except Exception as e:
            insider_alerts = []
            logger.warning(f"{ticker} インサイダーエラー: {e}")

        # --- 重複チェック ---
        ticker_all = whale_alerts + spoofing_alerts + insider_alerts
        current_fps = set(make_fingerprint(a) for a in ticker_all)
        prev_fps = last_fingerprints.get(ticker, set())

        is_changed = current_fps != prev_fps
        last_fingerprints[ticker] = current_fps

        # --- センチメント ---
        current_price = ticker_info["price"]["price"] if ticker_info["price"] else None
        sentiment = get_sentiment_direction(whale_alerts, current_price)
        sentiment_text = format_sentiment_jp(ticker, whale_alerts, ticker_info["price"])

        # --- タイムスタンプ・日本語テキスト付加 ---
        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        for a in whale_alerts:
            a["detected_at"] = now_str
            a["sentiment"] = sentiment
            a["text_jp"] = format_whale_alert_history_jp(a)

        for a in spoofing_alerts:
            a["detected_at"] = now_str
            a["text_jp"] = format_spoofing_history_jp(a, current_price)

        for a in insider_alerts:
            a["detected_at"] = now_str
            a["text_jp"] = format_insider_alert_jp(a)

        # --- 変化があったアラートのみ履歴に追加 ---
        if is_changed:
            for a in ticker_all:
                alert_history.append(a)

        ticker_info["whale_alerts"] = whale_alerts
        ticker_info["spoofing_alerts"] = spoofing_alerts
        ticker_info["insider_alerts"] = insider_alerts
        ticker_info["sentiment"] = sentiment
        ticker_info["sentiment_text"] = sentiment_text
        ticker_info["alert_count"] = len(ticker_all)
        ticker_info["changed"] = is_changed

        tickers_data[ticker] = ticker_info

        cycle_whale.extend(whale_alerts)
        cycle_spoofing.extend(spoofing_alerts)
        cycle_insider.extend(insider_alerts)

        logger.info(
            f"  {ticker}: whale={len(whale_alerts)} "
            f"spoof={len(spoofing_alerts)} "
            f"insider={len(insider_alerts)}"
            f"{' (変更なし)' if not is_changed else ''}"
        )

        if idx < len(WATCHLIST):
            time.sleep(SLEEP_BETWEEN_TICKERS)

    elapsed = round(time.time() - scan_start, 1)

    # グローバルステート更新 (アトミックに差し替え — 途中状態をAPIに見せない)
    latest_tickers.update(tickers_data)
    for k in list(latest_tickers):
        if k not in tickers_data:
            latest_tickers.pop(k, None)
    latest_summary["total_alerts"] = len(cycle_whale) + len(cycle_spoofing) + len(cycle_insider)
    latest_summary["whale_count"] = len(cycle_whale)
    latest_summary["spoofing_count"] = len(cycle_spoofing)
    latest_summary["insider_count"] = len(cycle_insider)

    logger.info(
        f"  完了 ({elapsed}s) — "
        f"whale={len(cycle_whale)} spoofing={len(cycle_spoofing)} "
        f"insider={len(cycle_insider)}"
    )


# ============================================================
# バックグラウンドスキャンループ
# ============================================================

async def scan_loop() -> None:
    """定期的にスキャンを実行するバックグラウンドタスク。
    どんな例外が発生しても自動復旧して監視を継続する。"""
    await asyncio.sleep(2)

    while True:
        try:
            scan_state["is_scanning"] = True
            scan_state["scan_count"] += 1
            num = scan_state["scan_count"]
            logger.info(f"━━━━ SCAN #{num} 開始 ({len(WATCHLIST)}銘柄) ━━━━")

            try:
                await asyncio.to_thread(run_full_scan)
                scan_state["last_scan_at"] = datetime.datetime.now().isoformat()
            except Exception as e:
                logger.error(f"スキャン実行エラー: {e}", exc_info=True)

            next_time = datetime.datetime.now() + datetime.timedelta(seconds=SCAN_INTERVAL)
            scan_state["next_scan_at"] = next_time.isoformat()
            scan_state["is_scanning"] = False

            logger.info(f"━━━━ SCAN #{num} 完了 — 次回: {next_time.strftime('%H:%M:%S')} ━━━━")
            await asyncio.sleep(SCAN_INTERVAL)

        except asyncio.CancelledError:
            logger.info("スキャンループ: キャンセルされました")
            raise
        except Exception as e:
            scan_state["is_scanning"] = False
            logger.error(f"スキャンループ致命的エラー — 30秒後に再開: {e}", exc_info=True)
            await asyncio.sleep(30)


# ============================================================
# FastAPI アプリケーション
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scan_loop())
    yield
    task.cancel()


app = FastAPI(
    title="Whale Radar Web",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ============================================================
# エンドポイント
# ============================================================

INDEX_HTML = Path(__file__).parent / "index.html"


@app.get("/")
async def serve_dashboard():
    """ダッシュボード HTML を配信 (認証不要 — ログイン画面を含む)。"""
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.post("/api/login")
async def login(request: Request):
    """パスワードを検証してトークンを返す。"""
    try:
        body = await request.json()
        password = body.get("password", "")
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid request"})

    if not APP_PASSWORD:
        return JSONResponse(content={"ok": True, "token": "open"})

    if hmac.compare_digest(password, APP_PASSWORD):
        return JSONResponse(content={"ok": True, "token": _VALID_TOKEN})

    return JSONResponse(status_code=401, content={"ok": False, "error": "パスワードが違います"})


@app.post("/api/verify")
async def verify_token(request: Request):
    """保存済みトークンが有効かどうかを返す。"""
    if not APP_PASSWORD:
        return JSONResponse(content={"valid": True})
    try:
        body = await request.json()
        token = body.get("token", "")
    except Exception:
        return JSONResponse(content={"valid": False})
    return JSONResponse(content={"valid": hmac.compare_digest(token, _VALID_TOKEN)})


@app.get("/api/alerts")
async def get_alerts():
    """最新のスキャン結果をJSON返却。"""
    return JSONResponse(content={
        "scan_count": scan_state["scan_count"],
        "last_scan_at": scan_state["last_scan_at"],
        "next_scan_at": scan_state["next_scan_at"],
        "is_scanning": scan_state["is_scanning"],
        "tickers": latest_tickers,
        "summary": latest_summary,
        "history": list(alert_history),
    })


@app.get("/api/status")
async def get_status():
    """スキャン状態・次回実行時刻・銘柄数を返す。"""
    return JSONResponse(content={
        "scan_count": scan_state["scan_count"],
        "is_scanning": scan_state["is_scanning"],
        "last_scan_at": scan_state["last_scan_at"],
        "next_scan_at": scan_state["next_scan_at"],
        "watchlist_count": len(WATCHLIST),
        "watchlist": WATCHLIST,
        "scan_interval_sec": SCAN_INTERVAL,
        "history_count": len(alert_history),
        "summary": latest_summary,
    })


# ============================================================
# 直接実行
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
