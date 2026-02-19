# Whale Radar Web

米国株オプション大口監視システム「Whale Radar」のWeb版ダッシュボード。

## 構成

| ファイル | 役割 |
|---------|------|
| `scanner.py` | スキャンロジック（desktop版から移植） |
| `main.py` | FastAPI バックエンド + バックグラウンドスキャン |
| `index.html` | サイバーパンク風ダッシュボード（単一HTML） |
| `.env.example` | 環境変数テンプレート |
| `requirements.txt` | Python 依存パッケージ |
| `render.yaml` | Render デプロイ設定 |

## API エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| GET | `/` | ダッシュボード HTML |
| GET | `/api/alerts` | 最新スキャン結果（JSON） |
| GET | `/api/status` | スキャン状態・次回実行時刻 |

## ローカル起動

```bash
cd web
cp .env.example .env   # 必要に応じて値を編集
pip install -r requirements.txt
python main.py
# → http://localhost:8000
```

## 設定変更

`.env` を直接編集してサーバー再起動するだけ。フロントに設定UIはない。

```env
# ウォッチリスト
WATCHLIST=NVDA,AMD,SMCI,PLTR,ARM,MSTR,COIN,MARA,TSLA,SOFI,RKLB,GRRR

# オプション出来高の異常判定閾値 (Vol/OI比)
OPTION_VOLUME_THRESHOLD=3.0

# スキャン間隔 (秒)
SCAN_INTERVAL=300

# 銘柄間のスリープ (秒)
SLEEP_BETWEEN_TICKERS=2.0

# API リクエスト間のスリープ (秒)
SLEEP_BETWEEN_REQUESTS=1.5

# CORS 許可ドメイン
CORS_ORIGINS=https://bota-labo.blog

# Discord Webhook (空なら送信しない)
WEBHOOK_URL=
```

## デプロイ

### Render

1. GitHub にプッシュ
2. Render → New Web Service → リポジトリを接続
3. Root Directory を `web` に設定
4. Environment Variables に `.env` の内容を設定
5. `render.yaml` が自動認識される

### Railway

1. GitHub にプッシュ
2. Railway → New Project → Deploy from GitHub
3. Root Directory を `web` に設定
4. Variables に `.env` の内容を設定
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## WordPress 埋め込み

```html
<iframe src="https://your-app.onrender.com/"
        width="100%" height="800" frameborder="0">
</iframe>
```

## 技術スタック

- **Backend**: FastAPI + uvicorn
- **Data**: yfinance + finviz (BeautifulSoup)
- **Frontend**: Vanilla HTML/CSS/JS（外部ライブラリなし）
- **UI**: Cyberpunk theme（desktop版カラーパレット準拠）
