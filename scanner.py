"""
==========================================================
  Whale Radar Web — Scanner Module
  whale_radar_main.py から移植したスキャンロジック。
  winsound / pygame / customtkinter 依存なし。
==========================================================
"""

import datetime
import statistics

import requests
import yfinance as yf
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ============================================================
# 株価取得
# ============================================================

def get_stock_price(ticker_symbol: str) -> dict | None:
    """yfinance で現在株価・前日比を取得する。"""
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.fast_info
        price = info.get("lastPrice", 0) or info.get("last_price", 0)
        prev = info.get("previousClose", 0) or info.get("previous_close", 0)
        if price and prev and prev > 0:
            change = price - prev
            change_pct = (change / prev) * 100
            return {
                "price": round(price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
            }
        elif price:
            return {"price": round(price, 2), "change": 0, "change_pct": 0}
    except Exception:
        pass
    return None


# ============================================================
# A) オプション出来高チェック (check_options_volume)
# ============================================================

def check_options_volume(ticker_symbol: str, threshold: float) -> list[dict]:
    """
    オプションの出来高/建玉比が閾値を超える異常な銘柄を検出する。
    Vol > 1000 かつ OI > 100 かつ Vol/OI >= threshold でアラート。
    """
    alerts = []
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        if not expirations:
            return alerts
        for exp in expirations[:2]:
            try:
                chain = ticker.option_chain(exp)
            except Exception:
                continue
            for option_type, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
                if df.empty:
                    continue
                df = df.copy()
                df["volume"] = df["volume"].fillna(0)
                df["openInterest"] = df["openInterest"].fillna(0)
                for _, row in df.iterrows():
                    vol = row["volume"]
                    oi = row["openInterest"]
                    if vol > 1000 and oi > 100 and (vol / oi) >= threshold:
                        alerts.append({
                            "ticker": ticker_symbol,
                            "type": "WHALE_OPTIONS",
                            "option_type": option_type,
                            "expiry": exp,
                            "strike": row["strike"],
                            "volume": int(vol),
                            "open_interest": int(oi),
                            "ratio": round(vol / oi, 1),
                        })
                    elif vol >= 50000 and oi <= 100:
                        alerts.append({
                            "ticker": ticker_symbol,
                            "type": "WHALE_OPTIONS",
                            "option_type": option_type,
                            "expiry": exp,
                            "strike": row["strike"],
                            "volume": int(vol),
                            "open_interest": int(oi),
                            "ratio": round(vol / oi, 1) if oi > 0 else 999.9,
                        })
    except Exception:
        pass
    alerts.sort(key=lambda a: a["volume"], reverse=True)
    return alerts[:5]


# ============================================================
# B) 異常板検知 (check_spoofing_from_chain)
# ============================================================

def check_spoofing_from_chain(
    ticker_symbol: str,
    spoofing_multiplier: float = 10.0,
    min_contracts: int = 500,
) -> list[dict]:
    """
    オプションチェーンの OI から異常な壁（板）を検出する。
    OI が中央値の spoofing_multiplier 倍以上でアラート。
    """
    alerts = []
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        if not expirations:
            return alerts

        for exp in expirations[:2]:
            try:
                chain = ticker.option_chain(exp)
            except Exception:
                continue

            for option_type, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
                if df.empty or len(df) < 5:
                    continue
                df = df.copy()
                df["openInterest"] = df["openInterest"].fillna(0)
                oi_values = df["openInterest"].tolist()
                oi_nonzero = [v for v in oi_values if v > 0]
                if len(oi_nonzero) < 3:
                    continue
                median_oi = statistics.median(oi_nonzero)
                if median_oi <= 0:
                    continue

                for _, row in df.iterrows():
                    oi = row["openInterest"]
                    if oi < min_contracts:
                        continue
                    ratio = oi / median_oi
                    if ratio < spoofing_multiplier:
                        continue

                    bid = row.get("bid", 0) or 0
                    ask = row.get("ask", 0) or 0
                    strikes = df["strike"].tolist()
                    mid_strike = statistics.median(strikes)
                    wall_side = "ASK" if row["strike"] >= mid_strike else "BID"
                    wall_price = ask if wall_side == "ASK" else bid

                    alerts.append({
                        "ticker": ticker_symbol,
                        "type": "SPOOFING_WALL",
                        "option_type": option_type,
                        "expiry": exp,
                        "strike": row["strike"],
                        "wall_side": wall_side,
                        "wall_price": wall_price,
                        "wall_size": int(oi),
                        "size_ratio": round(ratio, 1),
                    })

        alerts.sort(key=lambda a: a["size_ratio"], reverse=True)
    except Exception:
        pass
    return alerts[:5]


# ============================================================
# C) インサイダー取引チェック (check_insider_trades)
# ============================================================

def check_insider_trades(ticker_symbol: str) -> list[dict]:
    """finviz から直近7日以内のインサイダー購入を検出する。"""
    alerts = []
    url = f"https://finviz.com/quote.ashx?t={ticker_symbol}&ty=c&ta=0&p=d"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return alerts
        soup = BeautifulSoup(resp.text, "html.parser")
        insider_table = None
        for table in soup.find_all("table"):
            header_row = table.find("tr")
            if header_row and "Insider" in header_row.get_text():
                insider_table = table
                break
        if not insider_table:
            insider_table = soup.find("table", class_="body-table")
        if not insider_table:
            insider_table = soup.find("table", {"id": "insider-table"})
        if not insider_table:
            return alerts
        rows = insider_table.find_all("tr")[1:]
        today = datetime.date.today()
        for row in rows[:10]:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue
            try:
                insider_name = cols[0].get_text(strip=True)
                relationship = cols[1].get_text(strip=True)
                trade_date = cols[2].get_text(strip=True)
                transaction = cols[3].get_text(strip=True)
                value = cols[4].get_text(strip=True) if len(cols) > 4 else "N/A"
                shares = cols[5].get_text(strip=True) if len(cols) > 5 else "N/A"
            except (IndexError, AttributeError):
                continue
            if "Purchase" in transaction or "Buy" in transaction:
                try:
                    parsed_date = datetime.datetime.strptime(
                        trade_date, "%b %d"
                    ).date()
                    parsed_date = parsed_date.replace(year=today.year)
                    if (today - parsed_date).days > 7:
                        continue
                except ValueError:
                    pass
                alerts.append({
                    "ticker": ticker_symbol,
                    "type": "INSIDER_BUY",
                    "insider_name": insider_name,
                    "relationship": relationship,
                    "date": trade_date,
                    "transaction": transaction,
                    "value": value,
                    "shares": shares,
                })
    except Exception:
        pass
    return alerts


# ============================================================
# OTM重み付けセンチメント判定 (get_sentiment_direction)
# ============================================================

def get_sentiment_direction(
    whale_alerts: list[dict],
    current_price: float | None,
) -> str | None:
    """
    株価との位置関係（OTM/ITM）を考慮して方向感を判定する。
    Returns: "up" / "down" / None
    """
    if not whale_alerts:
        return None

    if current_price is None or current_price <= 0:
        call_vol = sum(
            a["volume"] for a in whale_alerts if a.get("option_type") == "CALL"
        )
        put_vol = sum(
            a["volume"] for a in whale_alerts if a.get("option_type") == "PUT"
        )
        if call_vol >= put_vol * 1.5:
            return "up"
        elif put_vol >= call_vol * 1.5:
            return "down"
        return None

    bull_score = 0.0
    bear_score = 0.0

    for a in whale_alerts:
        vol = a["volume"]
        strike = a["strike"]
        opt_type = a["option_type"]

        if opt_type == "CALL":
            if strike > current_price:
                # OTM CALL: 上昇への強い賭け（順張り）
                bull_score += vol * 1.5
            else:
                # ITM CALL: ヘッジや利確の可能性（ダマシ警戒）
                bull_score += vol * 0.5
        elif opt_type == "PUT":
            if strike < current_price:
                # OTM PUT: 下落への強い賭け（順張り）
                bear_score += vol * 1.5
            else:
                # ITM PUT: ヘッジや利確の可能性（ダマシ警戒）
                bear_score += vol * 0.5

    if bull_score >= bear_score * 1.5:
        return "up"
    elif bear_score >= bull_score * 1.5:
        return "down"

    return None


# ============================================================
# 日本語フォーマット関数群
# ============================================================

def _get_direction_label(option_type: str, ratio: float) -> str:
    is_extreme = (ratio >= 10) or (ratio == 999.9)
    if option_type == "CALL":
        return "\U0001f4c8 強い上方シグナル(CALL)" if is_extreme else "\U0001f4c8 上方シグナル(CALL)"
    else:
        return "\U0001f4c9 強い下方シグナル(PUT)" if is_extreme else "\U0001f4c9 下方シグナル(PUT)"


def format_whale_alert_history_jp(alert: dict) -> str:
    """Whale Options アラートの履歴表示用テキスト。"""
    direction = _get_direction_label(alert["option_type"], alert["ratio"])
    ratio_str = (
        f'通常の{alert["ratio"]}倍'
        if alert["ratio"] != 999.9
        else "通常の∞倍"
    )
    return (
        f"\U0001f6a8 【{alert['ticker']}】 {direction} "
        f"${alert['strike']}  Vol:{alert['volume']:,} ({ratio_str})"
    )


def _get_wall_commentary(alert: dict, current_price: float | None = None) -> str:
    if current_price is None or current_price <= 0:
        return ""
    strike = alert["strike"]
    opt_type = alert["option_type"]
    if opt_type == "PUT" and strike > current_price:
        return "\n  \U0001f449 現在値より高い不自然なPUT配置。一時的な下押し煽りの可能性を示唆"
    elif opt_type == "CALL" and strike < current_price:
        return "\n  \U0001f449 現在値より低い不自然なCALL配置。一時的な高値掴み誘発の可能性を示唆"
    else:
        return "\n  \U0001f449 通常の牽制（けんせい）目的の厚い板"


def format_spoofing_history_jp(
    alert: dict,
    current_price: float | None = None,
    near_price: bool = False,
) -> str:
    """異常板アラートの履歴表示用テキスト。"""
    wall_dir = "売壁" if alert["wall_side"] == "ASK" else "買壁"
    itm_comment = _get_wall_commentary(alert, current_price)
    prefix = "\U0001f525\u26a0\ufe0f" if near_price else "\u26a0\ufe0f"
    return (
        f"{prefix} 【{alert['ticker']}】 異常板({alert['option_type']}) "
        f"${alert['strike']} {wall_dir} "
        f"板サイズ:{alert['wall_size']:,}枚 (板厚比: {alert['size_ratio']}倍) "
        f"※未約定の指値"
        f"{itm_comment}"
    )


def format_insider_alert_jp(alert: dict) -> str:
    """インサイダー購入アラートのテキスト。"""
    return (
        f"\U0001f575\ufe0f 【{alert['ticker']}】 内部者が自社株購入! "
        f"{alert['insider_name']} ({alert['relationship']}) "
        f"{alert['date']} {alert['value']}"
    )


def format_sentiment_jp(
    ticker: str,
    whale_alerts: list[dict],
    price_data: dict | None = None,
) -> str | None:
    """
    GUIのアラート履歴に表示する日本語の結論テキスト。
    補正スコア（OTM重み付け）を可視化する。
    """
    if not whale_alerts:
        return None

    current_price = price_data.get("price") if price_data else None

    call_vol = sum(
        a["volume"] for a in whale_alerts if a.get("option_type") == "CALL"
    )
    put_vol = sum(
        a["volume"] for a in whale_alerts if a.get("option_type") == "PUT"
    )
    total = call_vol + put_vol
    if total == 0:
        return None

    if current_price and current_price > 0:
        price_str = f"現在値: ${current_price:,.2f}"

        bull_score = 0.0
        bear_score = 0.0
        for a in whale_alerts:
            vol = a["volume"]
            strike = a["strike"]
            opt_type = a["option_type"]
            if opt_type == "CALL":
                bull_score += vol * 1.5 if strike > current_price else vol * 0.5
            elif opt_type == "PUT":
                bear_score += vol * 1.5 if strike < current_price else vol * 0.5

        if bull_score >= bear_score * 3:
            verdict = "\U0001f4c8 上方向優勢（OTM重み付け・強）"
        elif bull_score >= bear_score * 1.5:
            verdict = "\U0001f4c8 上方向優勢（OTM重み付け）"
        elif bear_score >= bull_score * 3:
            verdict = "\U0001f4c9 下押し圧力に注意（OTM重み付け・強）"
        elif bear_score >= bull_score * 1.5:
            verdict = "\U0001f4c9 下押し圧力に注意（OTM重み付け）"
        else:
            verdict = "\u2694\ufe0f 上下拮抗（方向感なし）"

        detail = (
            f"（CALL:{call_vol:,} vs PUT:{put_vol:,} | "
            f"補正スコア 牛:{bull_score:.0f} 熊:{bear_score:.0f}）"
        )
        return (
            f"  \U0001f4ca 【{ticker}】 {price_str} | 結論: {verdict}\n"
            f"      {detail}"
        )
    else:
        price_str = "現在値: ---"
        if call_vol >= put_vol * 3:
            verdict = "\U0001f4c8 上方向優勢（出来高ベース・CALL圧倒的）"
        elif call_vol >= put_vol * 1.5:
            verdict = "\U0001f4c8 上方向優勢（出来高ベース）"
        elif put_vol >= call_vol * 3:
            verdict = "\U0001f4c9 下押し圧力に注意（出来高ベース・PUT圧倒的）"
        elif put_vol >= call_vol * 1.5:
            verdict = "\U0001f4c9 下押し圧力に注意（出来高ベース）"
        else:
            verdict = "\u2694\ufe0f 上下拮抗（方向感なし）"

        detail = f"（CALL:{call_vol:,} vs PUT:{put_vol:,}）"
        return (
            f"  \U0001f4ca 【{ticker}】 {price_str} | 結論: {verdict}\n"
            f"      {detail}"
        )


# ============================================================
# フィンガープリント (重複抑制)
# ============================================================

def make_fingerprint(alert: dict) -> str:
    """アラートの内容を比較用文字列に変換する。"""
    atype = alert.get("type", "")
    if atype == "WHALE_OPTIONS":
        return (
            f"W|{alert['ticker']}|{alert['option_type']}|"
            f"{alert['strike']}|{alert['expiry']}|{alert['volume']}"
        )
    elif atype == "SPOOFING_WALL":
        return (
            f"S|{alert['ticker']}|{alert['option_type']}|"
            f"{alert['strike']}|{alert['expiry']}|{alert['wall_size']}"
        )
    elif atype == "INSIDER_BUY":
        return (
            f"I|{alert['ticker']}|{alert['insider_name']}|"
            f"{alert['date']}|{alert['value']}"
        )
    return str(alert)
