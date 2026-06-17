# Nifty Quant Lab

A quantitative research and signal-generation platform for NSE/Nifty equities. Covers the full pipeline from raw market data to actionable swing-trade signals, with a Streamlit dashboard, REST API, and Telegram alerts.

---

## Features

- **Data pipeline** — EOD OHLCV download via yfinance + NSE scraper, stored in MySQL
- **Indicators engine** — RSI, MACD, Bollinger Bands, ATR, EMA stack, volume profile, and more
- **Swing scanner** — Multi-factor signal scoring across the Nifty universe (buy/sell/neutral)
- **Support & Resistance** — Pivot-based and price-cluster S/R detection
- **Open Interest analytics** — Futures OI, PCR, max-pain, and rollover tracking
- **Volatility analytics** — Historical and implied volatility surface
- **REST API** — FastAPI endpoints for market data, indicators, scanner results, and OI
- **Streamlit dashboard** — Interactive charts, scanner table, OI heatmap
- **Daily reports** — Excel/PDF report generation with Telegram delivery
- **Scheduler** — APScheduler-driven EOD automation (download → indicators → scan → report)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Database | MySQL 8 + SQLAlchemy 2 + Alembic |
| Cache | Redis |
| Data | yfinance, NSE scraper (httpx) |
| Numerics | NumPy, Pandas, SciPy |
| ML (Phase 3) | scikit-learn, LightGBM, CVXPY |
| Scheduler | APScheduler |
| Alerts | Telegram Bot API |
| Testing | pytest, pytest-asyncio, pytest-cov |

---

## Quick Start

### Prerequisites

- Python 3.11+
- MySQL 8
- Redis

### 1. Clone & install

```bash
git clone https://github.com/Raj1984/nifty-quant-lab.git
cd nifty-quant-lab
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set DB_HOST, DB_PASSWORD, TELEGRAM_BOT_TOKEN, etc.
```

### 3. First-time setup (creates tables, downloads 10Y history, runs initial scan)

```bash
python main.py setup
```

---

## CLI Commands

```
python main.py <command>

  api            Start FastAPI server (default: http://localhost:8000)
  setup          Full first-time setup: DB → data → indicators → scan
  scan           Run swing scanner and print signal summary
  download       Download latest EOD data
  indicators     Compute technical indicators for all symbols
  report         Generate and send daily report via Telegram
  dashboard      Launch Streamlit dashboard (http://localhost:8501)
```

---

## Docker

```bash
docker-compose up -d
```

Starts MySQL, Redis, the API server, and the Streamlit dashboard.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/market/{symbol}` | Latest OHLCV + indicators |
| GET | `/market/{symbol}/historical` | Historical OHLCV |
| GET | `/market/{symbol}/indicators` | Computed indicators |
| GET | `/market/{symbol}/sr` | Support & resistance levels |
| GET | `/scanner` | Latest scan results |
| POST | `/scanner/run` | Trigger a live scan |
| GET | `/oi/...` | Open interest routes |

Full docs at `http://localhost:8000/docs` when the API is running.

---

## Project Structure

```
nifty_quant_lab/
├── analytics/          # S/R, OI, futures, volatility
├── api/                # FastAPI app + OI routes
├── config/             # Settings (pydantic) + APScheduler
├── dashboard/          # Streamlit app + pages
├── data/               # Downloader, yfinance & NSE providers
├── database/           # SQLAlchemy models, connection, upsert
├── indicators/         # Indicator engine + service
├── reports/            # Daily report generator
├── signals/            # Swing scanner + OI service
├── telegram/           # Alert sender
├── tests/              # Unit & integration tests
├── utils/              # Logger
├── alembic/            # DB migrations
├── main.py             # Unified CLI entrypoint
└── docker-compose.yml
```

---

## Testing

```bash
pytest                        # all tests with coverage
pytest -m unit                # unit tests only (no network/DB)
pytest -m integration         # requires live DB
```

Current status: **132 tests passing**, 67% coverage.

---

## Roadmap

- **Phase 2** — Live order execution via Zerodha KiteConnect / Angel One SmartAPI
- **Phase 3** — ML-based signal ranking (LightGBM), portfolio optimisation (CVXPY), AI commentary (LLM)
