---
title: Credit Risk Portfolio Analytics
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app/app.py
pinned: false
---

# Credit Risk Analytics

Project for credit risk analytics.

## Setup

Create and activate the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure an API key for the portfolio assistant by copying `.env.example` to
`.env` and replacing the placeholder with a Gemini or OpenAI API key:

```bash
cp .env.example .env
```

Keep the key in `.env`; do not commit or share it. The app also supports keys
provided through Streamlit secrets.

Run the app from the project root (the directory containing `app/` and `src/`):

```bash
streamlit run app/app.py
```

## Project structure

- `app/`: application entry point
- `src/`: data ingestion, modeling, anomaly detection, and AI assistant modules
- `data/raw/`: source CSV files
- `data/processed/`: processed data outputs
- `docs/`: project documentation
- `tests/`: automated tests
