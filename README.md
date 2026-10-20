# StockWise AI

**StockWise AI** — AI-powered inventory expiry prediction and smart selling recommendation application.

> Tagline: *Predict Smarter. Waste Less. Sell Better.*

## Phase 1 — Project Foundation and Professional Dashboard UI

This phase delivers a polished offline dashboard UI with:

- Professional Streamlit layout (light warm-gray theme, navy typography, teal/indigo accents)
- Sidebar navigation (Overview, Inventory, Sales History, Predictions, Alerts & Recommendations, Reports, Settings)
- Reusable metric cards (Products Tracked, High-Risk Products, Potentially Unsold Units, Potential Cost Exposure)
- Plotly charts (Sales vs Forecast, Expiry Risk Distribution)
- Sample "Needs Attention" products table with search and risk filtering
- Clear illustrative-data disclosures (sample values are **not** real predictions)

All data shown in Phase 1 is illustrative sample data for UI testing only.

## Technology Stack

- Python 3.11+
- Streamlit
- Pandas, NumPy
- Plotly
- Scikit-learn (reserved for future ML phases)
- SQLite (reserved for future persistence phases)

No paid APIs, API keys, cloud services, or external AI services are used. The app runs locally and offline after installation.

## Local Execution

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
StockWise-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
│   └── styles.css
├── components/
│   ├── sidebar.py
│   ├── metric_cards.py
│   └── charts.py
├── pages/
│   ├── 1_Inventory.py
│   ├── 2_Sales_History.py
│   ├── 3_Predictions.py
│   ├── 4_Alerts_Recommendations.py
│   ├── 5_Reports.py
│   └── 6_Settings.py
├── services/
├── database/
├── ml/
├── utils/
├── tests/
├── data/
└── reports/
```

Only Phase 1 files are implemented; remaining directories are reserved for later phases.

## Security

No API keys, passwords, tokens, or private data are stored in this repository. See `.gitignore` for excluded local data and environment files.
