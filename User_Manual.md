# Data Validator Pro — User Manual

**Version 3.0 (Enhanced Scalable Edition)**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation & Setup](#2-installation--setup)
3. [Architecture & Project Structure](#3-architecture--project-structure)
4. [How It Works — The Validation Pipeline](#4-how-it-works--the-validation-pipeline)
5. [What It Tests — Complete Test Reference](#5-what-it-tests--complete-test-reference)
6. [GUI Walkthrough](#6-gui-walkthrough)
7. [Example Workflow](#7-example-workflow)
8. [Further Development Route](#8-further-development-route)

---

## 1. Overview

Data Validator Pro is a desktop application (Tkinter GUI) that ingests a tabular dataset and automatically runs a battery of statistical diagnostics depending on its structure. It auto-detects whether the data is **cross-sectional**, **time-series**, or **panel**, then branches into the appropriate test suite. Results are saved as a structured HTML report with embedded plots and a companion JSON file.

Key design goals:

- **Zero-config**: date columns, data type, and frequency are inferred automatically — but the user can override any decision via pop-up dialogs.
- **Scalable**: chunked reads, dtype downcasting, vectorised operations, and explicit `gc.collect()` calls keep memory manageable on multi-GB datasets.
- **Checkpoint-safe**: partial results are written to disk after each major pipeline stage so a crash never loses everything.

---

## 2. Installation & Setup

### Requirements

Python ≥ 3.9 is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
statsmodels>=0.14
scipy>=1.10
openpyxl>=3.0
psutil>=5.9
```

Optional: `ttkbootstrap` for a modern-looking GUI theme (the app detects it automatically and falls back to plain Tkinter if absent).

### Launching the App

```bash
python app.py
```

This opens the **Data Validator Pro** window with six tabs: Input, Preview, Log, Results, History, and Settings.

---

## 3. Architecture & Project Structure

```
├── app.py                  # Tkinter GUI — threading, tabs, override dialogs
├── main.py                 # StepwiseValidator — orchestrates the entire pipeline
├── requirements.txt
│
├── utils/
│   ├── read_data.py            # CSV / Excel / JSON / Parquet reader (auto-delimiter, downcasting)
│   ├── detect_date_column.py   # Heuristic date-column detector (keyword + parse scoring)
│   ├── ask_data_type.py        # Auto-classifies into cross_sectional / time_series / panel
│   ├── check_missing_duplicate.py  # Missing-value & duplicate-row diagnostics
│   └── convert_str_to_category.py  # String → pandas Categorical converter
│
├── cross_sectional/
│   ├── min_max.py              # Per-column min/max
│   ├── average.py              # Per-column mean
│   ├── median.py               # Per-column median
│   ├── outliers.py             # IQR / Z-score outlier detection
│   ├── correlation.py          # Pearson & Spearman correlation matrices
│   ├── normality.py            # Jarque-Bera & Shapiro-Wilk tests
│   ├── multicollinearity.py    # VIF & condition number
│   ├── heteroskedasticity.py   # Breusch-Pagan & White tests
│   └── plots.py                # Scatter, grouped bar, box plots
│
├── time_series/
│   ├── detect_range.py         # Date range & frequency inference
│   ├── frequency.py            # Detailed frequency regularity analysis
│   ├── missing_dates.py        # Gap detection in the date index
│   ├── stl_decomposition.py    # STL decomposition + seasonal strength (F_S)
│   ├── autocorrelation.py      # ACF, PACF, Ljung-Box Q test
│   ├── stationarity.py         # ADF, KPSS, Phillips-Perron unit-root tests
│   ├── volatility.py           # Engle ARCH-LM test
│   ├── structural_breaks.py    # Chow, CUSUM, Zivot-Andrews, Bai-Perron, Welch mean-shift
│   └── plots.py                # Line plots, STL component plots, ACF/PACF plots
│
├── panel/
│   ├── detect_range.py         # Per-entity date ranges
│   ├── frequency.py            # Per-entity frequency + suspicious gap detection
│   ├── balance.py              # Panel balance check + within/between variance decomposition
│   └── plots.py                # Per-entity series, faceted panel, snapshot bar charts
│
├── overrides/
│   ├── manager.py              # Key-value store for user override parameters
│   ├── registry.py             # Predefined dialog templates (date column, frequency, STL period, etc.)
│   └── dialogs.py              # Tkinter dialog windows triggered from the background thread
│
└── reports/
    └── html_report_v2.py       # Assembles results dict → self-contained HTML report
```

**Data flow:**

```
app.py  ──spawns thread──▶  StepwiseValidator.run_all_steps()
                                │
                    ┌───────────┼───────────────┐
                    ▼           ▼               ▼
             cross_sectional  time_series     panel
                    │           │               │
                    └───────────┴───────────────┘
                                │
                         generate_html_report()
                                │
                         results/<timestamp>/report.html
```

---

## 4. How It Works — The Validation Pipeline

The `StepwiseValidator` class in `main.py` runs five universal pre-processing steps, then branches into one of three analysis suites.

### 4.1 Universal Steps (always run)

| Step | Method | What it does |
|------|--------|-------------|
| 1 | `step_read_data()` | Reads CSV / Excel / JSON / Parquet with auto-delimiter detection, encoding sniffing, and dtype downcasting for memory efficiency. |
| 2 | `step_detect_date_col()` | Scores every column on date-keyword matches and parse success. Selects the best candidate (or asks the user if ambiguous). |
| 3 | `step_missing_duplicates()` | Counts missing cells per column, complete-row ratio, and duplicate rows. Supports entity+date keyed duplicate detection for panel data. |
| 4 | `step_convert_categories()` | Converts `object` / `string` columns (excluding date columns) to `pd.Categorical` for memory savings and downstream grouping. |
| 5 | `step_detect_data_type()` | Classifies the dataset: **cross-sectional** (no dates), **time-series** (one date column with unique dates), or **panel** (one date column with repeated dates + a categorical entity column). |

After step 3, a **checkpoint** is written to `partial_results.json` so that even if the analysis branch crashes, the data-quality results survive.

### 4.2 Branching Logic

- **No date column detected** → `cross_sectional`
- **One date column, dates are unique** → `time_series`
- **One date column, dates repeat + categorical entity column exists** → `panel`

The user can also force the type manually via the GUI dropdown.

---

## 5. What It Tests — Complete Test Reference

### 5.1 Cross-Sectional Tests

#### Descriptive Statistics
- **Min / Max** — per-column extrema.
- **Mean** — arithmetic mean per numeric column.
- **Median** — 50th percentile per numeric column.

#### Outlier Detection
- **IQR method (default)**: fences at Q1 − 1.5·IQR and Q3 + 1.5·IQR.
- **Z-score method** (optional): flags observations where |z| > 3.

#### Correlation Analysis
- **Pearson** r — linear association on pairwise-complete observations.
- **Spearman** ρ — rank correlation (robust to non-linearity).

#### Normality Tests

| Test | Null hypothesis | Mathematical basis |
|------|----------------|-------------------|
| **Jarque-Bera** | The data is normally distributed. | JB = (n/6)[S² + (K−3)²/4], where S = skewness, K = kurtosis. Under H₀, JB ~ χ²(2). |
| **Shapiro-Wilk** | The data is normally distributed. | W = (Σ aᵢ x₍ᵢ₎)² / Σ(xᵢ − x̄)². Sub-sampled to 5000 observations for large n. |

#### Multicollinearity

| Test | What it measures | Interpretation |
|------|-----------------|---------------|
| **VIF** (Variance Inflation Factor) | VIFⱼ = 1/(1 − Rⱼ²), where Rⱼ² is the R² from regressing Xⱼ on all other regressors. | VIF > 10 signals severe multicollinearity. |
| **Condition Number** | κ = σ_max / σ_min of the design matrix (singular values). | κ > 30 indicates ill-conditioning. |

#### Heteroskedasticity

| Test | Null hypothesis | Framework |
|------|----------------|-----------|
| **Breusch-Pagan** | Error variance is constant (homoskedastic). | Regresses squared OLS residuals on the original regressors. LM = nR² ~ χ²(k). |
| **White** | Error variance is constant. | Same idea but includes cross-products and squares of regressors — detects non-linear heteroskedasticity. |

Both tests use the first numeric column as Y and the remaining numeric columns as X (default behavior; override-able).

#### Plots
- Scatter plots (pairwise numeric), grouped bar charts (first categorical × numeric), box plots.

---

### 5.2 Time-Series Tests

All tests below are run **per numeric column** in the dataset.

#### Temporal Structure
- **Date range & frequency inference** — detects start/end dates and infers the dominant frequency (D, W, M, Q, A).
- **Frequency regularity analysis** — counts unique deltas, flags suspicious gaps (> 2× median delta).
- **Missing dates** — enumerates all gaps in the date index relative to the inferred frequency.

#### STL Decomposition

Seasonal-Trend decomposition using LOESS (Cleveland et al., 1990). Decomposes Yₜ = Tₜ + Sₜ + Rₜ (trend + seasonal + residual).

**Seasonal strength** is computed via the Hyndman & Athanasopoulos (2021) formula:

```
F_S = max(0,  1 − Var(R) / Var(S + R))
```

where F_S ∈ [0, 1]; a value of 1 means variance is entirely explained by the seasonal component.

The seasonal period is inferred from frequency (D→7, W→52, M→12, Q→4) or requested from the user via an override dialog.

#### Autocorrelation Diagnostics

| Diagnostic | Description |
|-----------|-------------|
| **ACF** | Sample autocorrelation function up to lag min(12, n/2). |
| **PACF** | Partial autocorrelation — isolates the direct effect of lag k after removing intermediate lags. |
| **Ljung-Box Q** | Q = n(n+2) Σ_{k=1}^{h} ρ̂²_k / (n−k). Under H₀ (no autocorrelation up to lag h), Q ~ χ²(h). |

#### Stationarity / Unit-Root Tests

| Test | H₀ | H₁ | Key detail |
|------|----|----|-----------|
| **ADF** (Augmented Dickey-Fuller) | Unit root (non-stationary) | Stationary | Lag length chosen by AIC. |
| **KPSS** | Level-stationary | Unit root | Note reversed hypotheses vs. ADF — use both for confirmation. |
| **Phillips-Perron** | Unit root | Stationary | Newey-West long-run variance correction — robust to serial correlation without explicit lag selection. Requires statsmodels ≥ 0.14. |

#### Volatility Clustering — ARCH-LM

Engle's ARCH-LM test checks for autoregressive conditional heteroskedasticity (time-varying variance). The auxiliary regression regresses squared residuals on their own lags:

```
ε̂²_t = α₀ + α₁ε̂²_{t-1} + … + α_qε̂²_{t-q} + uₜ
```

LM = nR² ~ χ²(q). A significant result (p < 0.05) means the series exhibits ARCH effects — common in financial returns.

#### Structural Break Detection

| Test | What it does | Mathematical framework |
|------|-------------|----------------------|
| **Chow** | Fits y = a + bt on the full sample and two sub-samples split at each candidate breakpoint. F = [(RSS_full − RSS₁ − RSS₂)/k] / [(RSS₁ + RSS₂)/(n − 2k)]. | Scans candidate breakpoints at 10%-90% quantiles. |
| **CUSUM** | Cumulative sum of recursive OLS residuals. If the process is stable, the CUSUM path stays within ±√n boundaries. | Boundary crossings flag parameter instability. |
| **Zivot-Andrews** | Unit-root test that endogenously selects a single structural break date by minimizing the ADF t-statistic across all candidate break points. | Distinguishes "unit root with no break" from "stationary with a level/trend shift". |
| **Bai-Perron** | Optimal multiple breakpoint detection using dynamic programming + BIC. | Finds the partition of [1, n] into m+1 segments that minimizes total RSS, then selects m via BIC. |
| **Welch mean-shift** | Scans candidate split points and runs a Welch two-sample t-test on the means of the left vs. right sub-samples. | Reports the split with the most significant t-statistic. |

#### Plots
- Line plot per numeric column, STL component plot (observed / trend / seasonal / residual), ACF/PACF bar plots, structural break overlay plots.

---

### 5.3 Panel Data Tests

Panel analysis first runs **panel-level diagnostics**, then loops over each entity and runs the **full time-series suite** per entity.

#### Panel-Level Diagnostics

| Test | Description |
|------|------------|
| **Date ranges** | Per-entity start/end dates — reveals uneven coverage. |
| **Panel balance** | Compares actual (entity, date) pairs vs. theoretical full grid. Reports balance ratio = actual/expected. |
| **Variance decomposition** | For each numeric column, decomposes overall variance into **between-entity** variance (how entity means differ) and **within-entity** variance (fluctuation around each entity's mean). Reported as absolute values and percentages. |
| **Frequency analysis** | Per-entity frequency regularity + suspicious gap detection. Reports the dominant panel frequency. |

#### Entity-Level Time-Series

For each entity, the validator slices the data and runs: frequency analysis, missing dates, STL decomposition, Ljung-Box, stationarity (ADF/KPSS/PP), ARCH-LM, and structural break tests — with separate plot directories per entity.

#### Plots
- Per-entity series overlay, faceted panel (one subplot per entity), panel average line, snapshot bar chart (cross-entity comparison at the last date).

---

## 6. GUI Walkthrough

### Tabs

| Tab | Purpose |
|-----|---------|
| **Input** | Browse for a file, select data type (auto / cross-sectional / time-series / panel), configure output directory, set overrides (date column, entity column), and hit **Run**. |
| **Preview** | Treeview table showing the first rows of the loaded dataset. |
| **Log** | Scrollable real-time log of every pipeline step, warning, and error. Supports save-to-file and clear. |
| **Results** | Displays the raw JSON results tree after the run completes. |
| **History** | Last 10 run records (file, type, timestamp, status). |
| **Settings** | Output directory, override parameter management. |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open file browser |
| `F5` / `Ctrl+R` | Start validation |
| `Ctrl+Q` | Quit |

### Override Dialogs

When the pipeline encounters ambiguity (multiple date columns, unknown frequency, unknown STL period, entity column selection), a **modal dialog** pops up and blocks the worker thread until the user responds. The dialog definitions live in `overrides/registry.py`.

---

## 7. Example Workflow

### Scenario: Validating a Monthly GDP Panel Dataset

Suppose you have `gdp_panel.csv`:

```csv
country,date,gdp_billion,inflation_pct
USA,2020-01-01,21000,1.2
USA,2020-02-01,20800,1.3
...
Germany,2020-01-01,3800,0.8
...
```

**Step-by-step:**

1. **Launch**: `python app.py`
2. **Browse**: Click: Browse → select `gdp_panel.csv`.
3. **Data type**: Leave on "Auto" (the app will detect `panel` because dates repeat across countries).
4. **Run**: Press F5. The log tab shows:
   ```
   [Step] Reading data...
   Loaded dataframe: 480 rows x 4 cols (0.1 MB)
   [Step] Detecting date columns...
   [Step] Checking missing / duplicates...
     Missing: 0 cells (0.00%)
     Duplicates: 0 rows (0.00%)
   [Step] Converting string columns to categorical...
   Dataset type: panel
   ```
5. **Entity dialog**: A popup asks "Select entity ID column:" → choose `country`.
6. **Panel analysis runs**: balance check, variance decomposition, frequency analysis, then per-country time-series (stationarity, ARCH-LM, structural breaks, etc.).
7. **Report**: Opens `results/2026-04-07_14-30-00/report.html` in your browser — a self-contained HTML file with all tables, test results, and embedded PNG plots.

### Programmatic Usage (No GUI)

You can also drive the validator from a script:

```python
from main import StepwiseValidator
from overrides.manager import OverrideManager

mgr = OverrideManager()
mgr.set("entity_column", "country")

validator = StepwiseValidator(
    filepath="gdp_panel.csv",
    manual_type=None,          # auto-detect
    override_mgr=mgr,
    logger_callback=print,
    gui_override_callback=None  # will raise if ambiguity requires user input
)

results = validator.run_all_steps()
report_path = validator.generate_html_report()
print(f"Report: {report_path}")
```

---

## 8. Further Development Route

This section is a guide for developers who want to extend the validator with new tests or capabilities.

### 8.1 Adding a New Cross-Sectional Test

**Example: Adding a Durbin-Watson autocorrelation test on OLS residuals.**

**Step 1** — Create the module: `cross_sectional/durbin_watson.py`

```python
"""
durbin_watson.py

Durbin-Watson test for first-order autocorrelation in OLS residuals.
DW ≈ 2(1 − ρ̂₁), so DW ≈ 2 means no autocorrelation,
DW → 0 means positive, DW → 4 means negative autocorrelation.
"""

import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.stats.stattools import durbin_watson


def run_durbin_watson(df: pd.DataFrame, y_col: str, x_cols: list) -> dict:
    y = pd.to_numeric(df[y_col], errors="coerce")
    X = df[x_cols].apply(pd.to_numeric, errors="coerce")
    combined = pd.concat([y, X], axis=1).dropna()

    model = OLS(combined.iloc[:, 0], add_constant(combined.iloc[:, 1:])).fit()
    dw = durbin_watson(model.resid)

    return {
        "dw_statistic": float(dw),
        "interpretation": (
            "positive autocorrelation" if dw < 1.5
            else "no autocorrelation" if dw < 2.5
            else "negative autocorrelation"
        ),
    }
```

**Step 2** — Wire it into `main.py`:

```python
# At the top, add the import:
from cross_sectional.durbin_watson import run_durbin_watson

# Inside run_cross_sectional(), add:
try:
    self.results["durbin_watson"] = run_durbin_watson(self.df, y, X)
except Exception as e:
    self.results["durbin_watson"] = {"error": str(e)}
```

**Step 3** — (Optional) Add it to the HTML report in `reports/html_report_v2.py` by handling the `"durbin_watson"` key in the results dict.

That's it. The pattern is always: **create module → import in main.py → call inside the branch method → optionally update the report renderer.**

### 8.2 Adding a New Time-Series Test

Same pattern. Create a file in `time_series/`, accept `(df, date_col, value_col)` as the signature, return a dict. Wire it into the per-column loop inside `StepwiseValidator.run_time_series()`. If the test also needs to run for panel entities, add it to `_run_time_series_on_subset()` as well.

### 8.3 Adding a New Panel-Level Test

Create a file in `panel/`, accept `(df, entity_col, ...)`. Call it from `run_panel()` before the entity loop. Example ideas: Hausman test (fixed vs. random effects), Pesaran CD test (cross-sectional dependence), Levin-Lin-Chu panel unit-root test.

### 8.4 Adding a New Override Dialog

1. Add a template in `overrides/registry.py`:
   ```python
   "my_new_override": {
       "title": "My New Override",
       "fields": {
           "my_param": {
               "type": "dropdown",  # or "text" or "multiselect"
               "label": "Choose something:",
               "options": []
           }
       }
   }
   ```
2. In `main.py`, call `self.gui_override_callback({...})` when you need user input. The dialog blocks the worker thread and returns the user's selection as a dict.

### 8.5 Adding a New Plot Type

All plot functions follow the same contract: accept a DataFrame + column names + `save_dir` or `save_path`, create a matplotlib figure, call `plt.savefig(...)`, then `plt.close()`. Always close figures to avoid memory leaks on large runs.

### 8.6 Suggested Feature Ideas

| Feature | Module location | Difficulty |
|---------|----------------|-----------|
| Granger causality (pairwise) | `time_series/granger.py` | Medium |
| Cointegration (Engle-Granger / Johansen) | `time_series/cointegration.py` | Medium |
| Hausman test (FE vs RE) | `panel/hausman.py` | Medium-Hard |
| Pesaran CD test | `panel/cross_dep.py` | Medium |
| Levin-Lin-Chu panel unit root | `panel/panel_unitroot.py` | Hard |
| Interactive Plotly report | `reports/html_report_plotly.py` | Medium |
| CLI mode (argparse, no GUI) | `cli.py` | Easy |
| YAML/TOML config file support | `utils/config.py` | Easy |
| Export to LaTeX tables | `reports/latex_export.py` | Easy |
| Automated PDF report (matplotlib PdfPages) | `reports/pdf_report.py` | Medium |

### 8.7 Code Conventions

- Every test module returns a plain `dict` (no custom classes) so it serialises to JSON without friction.
- Wrap every test call in `try/except` and store `{"error": str(e)}` on failure — never let one broken test crash the entire pipeline.
- Use `sanitize_for_json()` from `main.py` before writing results to disk.
- For large-data paths, avoid `.apply(axis=1)` and nested Python loops — prefer vectorised pandas/numpy operations.
- Always call `plt.close(fig)` after saving a figure.

---

*Made by Molnár Marcell for Data Validator Pro v3.0 — April 2026*
