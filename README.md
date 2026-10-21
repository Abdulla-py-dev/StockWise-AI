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
- SQLite (local persistence, Phase 2)

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

## Phase 2 — Inventory Management & SQLite

This phase adds a persistent local inventory management system on top of the Phase 1 dashboard.

- **Local SQLite database** under `database/` (`stockwise.db`), created automatically on startup via `database/db.py` and `database/schema.sql`. Existing databases are never overwritten.
- **Inventory CRUD**: add, view, edit, and delete products through `pages/1_Inventory.py`, with all SQL kept in `services/inventory_service.py`.
- **Validation**: SKU uniqueness, non-negative quantity/unit cost, required fields, and ISO `YYYY-MM-DD` expiry dates (`utils/validators.py`).
- **Search & filters**: search by SKU/name, filter by category and expiry status, "expiring soon" filter, sorting by name/SKU/category/quantity/unit cost/expiry date, and pagination.
- **CSV import**: preview before import, row-level validation errors, duplicate-SKU detection within the file and against the database, transactional import, summary of imported vs. rejected rows.
- **Dynamic dashboard metrics**: the Overview metric cards now read real values from SQLite (products tracked, high-risk products, total units in stock, potential cost exposure). Charts remain clearly labeled illustrative until a later phase.
- **Automated tests** in `tests/` using a temporary database (never the real inventory database).

All data stays local; no paid APIs, keys, or cloud services are used. `*.db`, `*.sqlite`, and `*.sqlite3` are gitignored.

## Security

No API keys, passwords, tokens, or private data are stored in this repository. See `.gitignore` for excluded local data and environment files.
