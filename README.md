# ⚡ EV Adoption in India — Interactive Dashboard

A polished Streamlit dashboard that analyses electric-vehicle trends, manufacturer market share,
charging-infrastructure coverage, and future outlook across India using five real-world datasets.

---

## 📁 Project Structure

```
ev_adoption_india/
├── app.py                                  ← Single Python file: all code lives here
├── requirements.txt
├── README.md
├── EV Maker by Place.csv                   ← Manufacturer name, city, state
├── ev_cat_01-24.csv                        ← Monthly EV registrations by vehicle class (2001–2024)
├── ev_sales_by_makers_and_cat_15-24.csv    ← Annual EV sales by maker & category (2015–2024)
├── OperationalPC.csv                       ← Operational public charging stations by state
└── Vehicle Class - All.csv                 ← All-time registrations by vehicle class
```

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Run the dashboard

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

> All five CSV files must be in the **same directory** as `app.py`.

---

## 🧩 Dashboard Sections

| Tab | What you'll find |
|-----|-----------------|
| **Overview** | Four headline KPIs: total registrations, leading manufacturer, top vehicle class, and state with the most charging stations |
| **📈 Trends** | Annual bar chart, year-over-year growth rate, monthly time-series, and a heatmap of categories vs years |
| **🚗 Categories** | Pie and bar charts of vehicle-class share; interactive category selector; stacked bar of sales by category (2015–2024) |
| **🏭 Manufacturers** | Market-share pie (top 15), all-time sales bar chart (top 10), annual growth lines, category mix per manufacturer |
| **🔌 Charging** | State-by-state bar charts, top/bottom 5 tables, infrastructure gap KPIs |
| **📍 Maker Locations** | Manufacturers per state, top manufacturing cities, interactive treemap (State → City → Maker), full directory |
| **🔮 Forecast** | Polynomial-regression forecast with configurable horizon (1–10 years) and training-start year; category-level forecast; clearly labelled as estimates |
| **💡 Recommendations** | Data-backed insights on adoption acceleration, charging gaps, policy priorities |

---

## 🔧 Sidebar Filters

All charts respect these sidebar filters:

- **Year range** — slide to focus on a specific period
- **Vehicle Category** — filter by EV category (2W, 3W, 4W, LMV, …)
- **Manufacturer (Top 30)** — multi-select from the highest-selling makers
- **State (Charging)** — narrow the charging-infrastructure view

---

## 🗂️ Dataset Source

> **Kaggle dataset link:** https://www.kaggle.com/datasets/srinrealyf/india-ev-market-data

The five CSV files used in this project can be downloaded from the Kaggle dataset above.
Place all downloaded CSV files in the **same directory** as `app.py` before running the dashboard.

---

## 📊 Data Details & Cleaning

| File | Key cleaning steps |
|------|--------------------|
| `ev_cat_01-24.csv` | Row where `Date == "0"` removed; dates parsed as `DD/MM/YY`; all numeric columns coerced; `Total` column derived |
| `ev_sales_by_makers_and_cat_15-24.csv` | Quoted maker names stripped; year columns converted to integers; `Total` derived |
| `Vehicle Class - All.csv` | Indian-format numbers (`1,02,965`) stripped of commas before conversion |
| `OperationalPC.csv` | BOM stripped; trailing blank rows dropped |
| `EV Maker by Place.csv` | BOM stripped; duplicate entries kept (same maker, different locations) |

---

## 🔮 Forecast Methodology

- **Model:** Polynomial regression (degree 2) via `scikit-learn`
- **Input:** Annual total EV registrations from `ev_cat_01-24.csv`
- **Training window:** Configurable — defaults to 2015 onwards
- **Horizon:** 1–10 years beyond the latest available year
- ⚠️ All forecast values are clearly labelled as **estimates**. They do not factor in policy changes,
  subsidy expiry, supply-chain disruptions, or demand shocks.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.32 | Web dashboard framework |
| `pandas` | ≥ 2.0 | Data loading & manipulation |
| `numpy` | ≥ 1.26 | Numerical operations |
| `plotly` | ≥ 5.20 | Interactive charts & Visualizations |
| `scikit-learn` | ≥ 1.4 | Polynomial regression for forecasting |

---

## 📝 License

This project is for educational and analytical purposes.  
Data sourced from public datasets (Vahan Dashboard / EVREPORTER).
