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
import json
import logging
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from zoneinfo import ZoneInfo

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

# ============================================================
# 米国市場 営業時間判定 (desktop版と同じロジック)
# ============================================================
US_MARKET_HOLIDAYS_2025 = {
    (1, 1), (1, 20), (2, 17), (4, 18), (5, 26),
    (6, 19), (7, 4), (9, 1), (11, 27), (12, 25),
}
US_MARKET_HOLIDAYS_2026 = {
    (1, 1), (1, 19), (2, 16), (4, 3), (5, 25),
    (6, 19), (7, 3), (9, 7), (11, 26), (12, 25),
}
_ALL_HOLIDAYS = US_MARKET_HOLIDAYS_2025 | US_MARKET_HOLIDAYS_2026


def is_us_market_open() -> bool:
    """米国市場がオープン中かどうかを判定する (ET基準)。"""
    et = datetime.datetime.now(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    if (et.month, et.day) in _ALL_HOLIDAYS:
        return False
    market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= et <= market_close


def get_market_status_text() -> str:
    """現在の市場状態テキストを返す。"""
    et = datetime.datetime.now(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return "MARKET CLOSED (Weekend / 週末休場)"
    if (et.month, et.day) in _ALL_HOLIDAYS:
        return "MARKET CLOSED (Holiday / 祝日休場)"
    hour = et.hour
    if hour < 9 or (hour == 9 and et.minute < 30):
        return "MARKET CLOSED (Pre-Market / 開場前)"
    if hour >= 16:
        return "MARKET CLOSED (After Hours / 閉場後)"
    return "MARKET OPEN"


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
_DEFAULT_WATCHLIST = [s.strip() for s in _env("WATCHLIST", "NVDA,AMD,SMCI,PLTR,ARM,MSTR,COIN,MARA,TSLA,SOFI,RKLB,GRRR,DJT,HOOD,AAPL,AMZN,META,GOOGL,MSFT,NFLX").split(",") if s.strip()]

# ウォッチリスト永続化ファイル
_WATCHLIST_FILE = Path(__file__).parent / "watchlist.json"


def _load_watchlist() -> list[str]:
    """watchlist.json があればそこから読み込み、なければ .env デフォルトを使う。"""
    if _WATCHLIST_FILE.exists():
        try:
            data = json.loads(_WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return list(_DEFAULT_WATCHLIST)


def _save_watchlist() -> None:
    """現在の WATCHLIST を watchlist.json に保存する。"""
    _WATCHLIST_FILE.write_text(
        json.dumps(WATCHLIST, ensure_ascii=False), encoding="utf-8"
    )


WATCHLIST: list[str] = _load_watchlist()
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
LIFETIME_PASSWORD = _env("LIFETIME_PASSWORD", "")
LIFETIME_MAX_USES = _env_int("LIFETIME_MAX_USES", 2)

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
_LIFETIME_TOKEN = _make_token(LIFETIME_PASSWORD) if LIFETIME_PASSWORD else ""

# ============================================================
# 買い切りパスワード使用回数管理 (入力回数ベース)
# ============================================================
_LIFETIME_FILE = Path(__file__).parent / "lifetime_uses.json"


def _load_lifetime_count() -> int:
    """使用回数を読み込む。"""
    if _LIFETIME_FILE.exists():
        try:
            data = json.loads(_LIFETIME_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data.get("count", 0)
        except Exception:
            pass
    return 0


def _save_lifetime_count(count: int) -> None:
    _LIFETIME_FILE.write_text(
        json.dumps({"count": count}, ensure_ascii=False), encoding="utf-8"
    )


_lifetime_use_count: int = _load_lifetime_count()


def _check_auth(request: Request) -> bool:
    """リクエストの認証トークンを検証する。
    APP_PASSWORD が空 → 常にTrue (認証なし)。"""
    if not APP_PASSWORD:
        return True
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        if hmac.compare_digest(token, _VALID_TOKEN):
            return True
        if _LIFETIME_TOKEN and hmac.compare_digest(token, _LIFETIME_TOKEN):
            return True
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

# ゲートタイマー (IPアドレス → 初回アクセス日時)
# { "1.2.3.4": {"start": timestamp, "date": "2026-02-19"} }
GATE_PREVIEW_SEC = 300  # 5分
_gate_tracker: dict[str, dict] = {}

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
    """全銘柄をスキャンして結果をグローバルステートに格納する。
    1銘柄完了ごとに即座にAPIに反映し、フロントで順次表示される。"""
    global last_fingerprints

    scan_start = time.time()
    scan_state["scanning_progress"] = f"0/{len(WATCHLIST)}"

    for idx, ticker in enumerate(WATCHLIST, start=1):
        logger.info(f"  [{idx}/{len(WATCHLIST)}] {ticker} スキャン中...")
        scan_state["scanning_progress"] = f"{idx}/{len(WATCHLIST)} ({ticker})"
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

        # --- 即座にグローバルステートに反映 (APIで即見える) ---
        latest_tickers[ticker] = ticker_info

        # サマリーもリアルタイム更新
        w = sum(len(v.get("whale_alerts", [])) for v in latest_tickers.values())
        s = sum(len(v.get("spoofing_alerts", [])) for v in latest_tickers.values())
        i = sum(len(v.get("insider_alerts", [])) for v in latest_tickers.values())
        latest_summary["whale_count"] = w
        latest_summary["spoofing_count"] = s
        latest_summary["insider_count"] = i
        latest_summary["total_alerts"] = w + s + i

        logger.info(
            f"  {ticker}: whale={len(whale_alerts)} "
            f"spoof={len(spoofing_alerts)} "
            f"insider={len(insider_alerts)}"
            f"{' (変更なし)' if not is_changed else ''}"
        )

        if idx < len(WATCHLIST):
            time.sleep(SLEEP_BETWEEN_TICKERS)

    # ウォッチリストから外された銘柄を除去
    for k in list(latest_tickers):
        if k not in WATCHLIST:
            latest_tickers.pop(k, None)

    elapsed = round(time.time() - scan_start, 1)
    scan_state["scanning_progress"] = ""
    logger.info(
        f"  完了 ({elapsed}s) — "
        f"whale={latest_summary['whale_count']} "
        f"spoofing={latest_summary['spoofing_count']} "
        f"insider={latest_summary['insider_count']}"
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
BGM_FILE = Path(__file__).parent / "bgm.mp3"


@app.get("/")
async def serve_dashboard():
    """ダッシュボード HTML を配信 (認証不要 — ログイン画面を含む)。"""
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/bgm.mp3")
async def serve_bgm():
    """BGM MP3 を配信する。"""
    if BGM_FILE.exists():
        return FileResponse(BGM_FILE, media_type="audio/mpeg")
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.post("/api/login")
async def login(request: Request):
    """パスワードを検証してトークンを返す。
    通常パスワード → 無制限
    買い切りパスワード → 入力回数制限 (LIFETIME_MAX_USES)"""
    try:
        body = await request.json()
        password = body.get("password", "")
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid request"})

    if not APP_PASSWORD:
        return JSONResponse(content={"ok": True, "token": "open"})

    # 通常パスワード (サブスク用)
    if hmac.compare_digest(password, APP_PASSWORD):
        return JSONResponse(content={"ok": True, "token": _VALID_TOKEN})

    # 買い切りパスワード (入力回数ベース)
    if LIFETIME_PASSWORD and hmac.compare_digest(password, LIFETIME_PASSWORD):
        global _lifetime_use_count

        # 上限チェック
        if _lifetime_use_count >= LIFETIME_MAX_USES:
            logger.warning(f"買い切りパスワード上限超過 (使用済み: {_lifetime_use_count}/{LIFETIME_MAX_USES})")
            return JSONResponse(status_code=401, content={
                "ok": False,
                "error": "このパスワードは使用回数の上限に達しました"
            })

        # カウント加算 & 保存
        _lifetime_use_count += 1
        _save_lifetime_count(_lifetime_use_count)
        logger.info(f"買い切りパスワード使用: {_lifetime_use_count}/{LIFETIME_MAX_USES}")
        return JSONResponse(content={"ok": True, "token": _LIFETIME_TOKEN, "lifetime": True})

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
    valid = hmac.compare_digest(token, _VALID_TOKEN)
    if not valid and _LIFETIME_TOKEN:
        valid = hmac.compare_digest(token, _LIFETIME_TOKEN)
    return JSONResponse(content={"valid": valid})


def _get_client_ip(request: Request) -> str:
    """プロキシ対応でクライアントIPを取得する。"""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/api/gate")
async def gate_status(request: Request):
    """IPアドレスベースでゲート状態を返す。
    - locked=false: プレビュー中 (remaining=残り秒数)
    - locked=true:  5分経過、ロック済み
    日付が変わればリセットされる。"""
    ip = _get_client_ip(request)
    today = datetime.date.today().isoformat()
    now = time.time()

    entry = _gate_tracker.get(ip)

    # 日付が変わった or 初回 → リセット
    if not entry or entry.get("date") != today:
        entry = {"start": now, "date": today}
        _gate_tracker[ip] = entry

    elapsed = int(now - entry["start"])
    remaining = max(0, GATE_PREVIEW_SEC - elapsed)

    return JSONResponse(content={
        "locked": remaining <= 0,
        "remaining": remaining,
        "preview_sec": GATE_PREVIEW_SEC,
    })


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
        "scanning_progress": scan_state.get("scanning_progress", ""),
        "last_scan_at": scan_state["last_scan_at"],
        "next_scan_at": scan_state["next_scan_at"],
        "watchlist_count": len(WATCHLIST),
        "watchlist": WATCHLIST,
        "scan_interval_sec": SCAN_INTERVAL,
        "history_count": len(alert_history),
        "summary": latest_summary,
        "market_open": is_us_market_open(),
        "market_status": get_market_status_text(),
    })


@app.post("/api/watchlist/add")
async def add_ticker(request: Request):
    """ウォッチリストに銘柄を追加する。"""
    try:
        body = await request.json()
        ticker = body.get("ticker", "").strip().upper()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid request"})

    if not ticker or not ticker.isalpha() or len(ticker) > 10:
        return JSONResponse(status_code=400, content={"ok": False, "error": "無効なティッカー"})

    if ticker in WATCHLIST:
        return JSONResponse(content={"ok": False, "error": f"{ticker} は既に登録されています"})

    WATCHLIST.append(ticker)
    scan_state["watchlist_count"] = len(WATCHLIST)
    _save_watchlist()
    logger.info(f"銘柄追加: {ticker} (合計 {len(WATCHLIST)} 銘柄)")
    return JSONResponse(content={"ok": True, "watchlist": WATCHLIST})


@app.post("/api/watchlist/remove")
async def remove_ticker(request: Request):
    """ウォッチリストから銘柄を削除する。"""
    try:
        body = await request.json()
        ticker = body.get("ticker", "").strip().upper()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid request"})

    if ticker not in WATCHLIST:
        return JSONResponse(content={"ok": False, "error": f"{ticker} はリストにありません"})

    WATCHLIST.remove(ticker)
    scan_state["watchlist_count"] = len(WATCHLIST)
    latest_tickers.pop(ticker, None)
    _save_watchlist()
    logger.info(f"銘柄削除: {ticker} (残り {len(WATCHLIST)} 銘柄)")
    return JSONResponse(content={"ok": True, "watchlist": WATCHLIST})


@app.post("/api/clear/predictions")
async def clear_predictions():
    """センチメント予測データをクリアする。"""
    for info in latest_tickers.values():
        info["sentiment"] = None
        info["sentiment_text"] = ""
        info["whale_alerts"] = []
    latest_summary["whale_count"] = 0
    latest_summary["total_alerts"] = (
        latest_summary["spoofing_count"] + latest_summary["insider_count"]
    )
    logger.info("実績データをクリアしました")
    return JSONResponse(content={"ok": True})


@app.post("/api/clear/history")
async def clear_history():
    """アラート履歴をクリアする。"""
    alert_history.clear()
    for info in latest_tickers.values():
        info["whale_alerts"] = []
        info["spoofing_alerts"] = []
        info["insider_alerts"] = []
        info["alert_count"] = 0
        info["changed"] = False
    latest_summary["whale_count"] = 0
    latest_summary["spoofing_count"] = 0
    latest_summary["insider_count"] = 0
    latest_summary["total_alerts"] = 0
    last_fingerprints.clear()
    logger.info("アラート履歴をクリアしました")
    return JSONResponse(content={"ok": True})


# ============================================================
# 直接実行
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
