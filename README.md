# Hybrid-stock-market-prediction-in-NSE
In this project,We have tried to combine arima with multiple machine learning algorithms to make our model
# Hybrid Stacking Ensemble for NSE Stock Direction Prediction

**Mini Project (01CE0609) — Rushil Gorasia, 6-EC4, Marwadi University**  
**Guided by: Prof. Rahul Jain**

This project started as an experiment to see how far you can push stock direction prediction using only price and volume history — no news, no sentiment, no fundamental data. The short answer: further than ARIMA alone, but the real gain comes from *how* you combine models, not which models you pick.

The final framework trains on three NSE stocks and then predicts the next-day direction (UP or DOWN) on three stocks it has never seen. The stacking meta-learner figures out the right combination weights from the data itself — no manual tuning. On Bajaj Finance it reached **70.85% directional accuracy** vs. 52.83% for the best standalone model, with statistical significance confirmed (McNemar p = 0.0169, T-test p = 0.0141).

---

## What's actually in this repo

```
.
├── Hybrid_research_paper_code.py     # Main pipeline — everything runs from here
├── README.md
└── outputs/                          # Generated when you run the code
    ├── hybrid_v8_stacked_*.csv       # Full predictions per test stock
    ├── metrics_stacked_*.csv         # Summary metrics
    ├── Figure_3.1_Overall_System_Architecture.png
    └── Figure_3.2_Walk_Forward_Validation.png
```

---

## The idea in plain English

Most papers on stock prediction train on one stock and test on the same stock at a later date. That's the minimum bar. This project does something harder — train on TCS, Infosys, and HDFC Bank, then test on Bajaj Finance, Reliance, and Hindustan Unilever, which the model has never seen.

On top of that, the models are combined using **stacking** rather than weighted voting. Weighted voting sounds fine on paper, but if you tune the weights by trying different combinations and picking what works best on the test set, you've leaked test data into your model. Stacking avoids this by learning the combination weights from held-out fold predictions during training — the test set stays completely clean.

```
Training stocks (what the models learn from)
    TCS.NS       → IT sector
    INFY.NS      → IT sector  
    HDFCBANK.NS  → Banking

Test stocks (never seen during training)
    BAJFINANCE.NS  → NBFC / high-volatility
    RELIANCE.NS    → Conglomerate / medium-volatility
    HINDUNILVR.NS  → Consumer goods / low-volatility
```

---

## Results

| Model | BAJFINANCE | RELIANCE | HINDUNILVR |
|---|---|---|---|
| ARIMA | 45.41% | 51.59% | 49.20% |
| XGBoost | 51.94% | 52.83% | 51.60% |
| LightGBM | 52.83% | 53.53% | 52.65% |
| Weighted Voting | 52.30% | 52.47% | 52.10% |
| **Stacking Hybrid** | **70.85%** | **67.67%** | **70.14%** |

The weighted voting row is important — it shows you almost nothing from just averaging the models. The large jump to 70%+ comes from the meta-learner, not from ensembling in general.

**On BAJFINANCE.NS (primary test stock):**
- DOWN recall: 68%  /  UP recall: 73%  — balanced, not just predicting one direction
- MCC: 0.400  /  ROC-AUC: 0.771
- McNemar test vs ARIMA: **p = 0.0169** ✓
- Paired t-test vs ARIMA: **p = 0.0141** ✓

Meta-learner weights (learned automatically, not hand-picked):
- LightGBM: **55.9%**
- XGBoost: **41.6%**  
- ARIMA: **2.6%** — makes sense, because auto_arima selects (0,0,0) on every NSE stock tested

---

## Setup

### If you're running on Google Colab (recommended)

Just open the `.py` file in Colab. The `!pip install` line at the top handles everything.

### If you're running locally

```bash
pip install pmdarima xgboost lightgbm yfinance statsmodels scikit-learn pandas numpy matplotlib seaborn scipy
```

Python 3.8+ should work fine.

---

## How to run it

The code is structured as numbered cells (it was originally a Colab notebook). Run them top to bottom. Here's what each section does:

### Cell 3 — Pick your test stock

```python
test_ticker = 'BAJFINANCE.NS'   # change this to RELIANCE.NS or HINDUNILVR.NS

train_tickers = [
    'TCS.NS',
    'INFY.NS',
    'HDFCBANK.NS'
]

start_date = '2016-01-01'
end_date   = '2025-03-01'
```

Change `test_ticker` to run on a different stock. The training stocks stay the same.

### Cells 4–6 — Data download and split

Downloads OHLCV data for all stocks plus INDIAVIX. Computes log returns. Splits the test stock 75/25 chronologically — roughly 566 trading days in the test set.

### Cell 7 — ARIMA order selection

Runs `auto_arima` on the training period log returns. On every large-cap NSE stock we've tried, it picks (0,0,0) — essentially saying "the best linear model is just the mean." Rather than stopping there, we force (1,0,1) to get a non-trivial signal that goes in as a feature.

### Cell 8 — Feature engineering

Builds 18 features from historical data only (no lookahead):

| # | Feature | Description |
|---|---|---|
| 1–5 | `ret_lag_1` to `ret_lag_5` | Previous 1–5 day log returns |
| 6–9 | `roll_mean_5/10`, `roll_std_5/10` | Rolling mean and std over 5 and 10 days |
| 10–11 | `momentum_5`, `momentum_10` | Price percentage change over 5 and 10 days |
| 12 | `abs_ret_lag1` | Absolute value of yesterday's return |
| 13 | `rsi_14` | 14-day RSI |
| 14 | `price_range` | (High − Low) / Close for the previous day |
| 15 | `close_pos` | Where the close fell within the day's range |
| 16 | `trend_persist` | How many of the last 5 days were UP |
| 17 | `ret_accel` | Change in return from two days ago to yesterday |
| 18 | `vol_change` | Day-over-day volume percentage change |

Target: `y = 1` if tomorrow's close > today's close, else `y = 0`.

### Cells 9–11 — Training

Builds the combined training set (~5,500 samples from three stocks), fits a StandardScaler on it, then trains XGBoost and LightGBM. Both use shallow trees (depth 3) to reduce overfitting — financial return series are noisy, and deeper trees tend to memorise that noise.

```
XGBoost:   500 trees, lr=0.03, depth=3, min_child_weight=8
LightGBM:  500 trees, lr=0.03, depth=3, min_child_samples=15
```

### Cell 12 — Walk-forward evaluation

This is where predictions actually happen. At each test step `t`:
1. ARIMA re-fits on the last 200 returns and forecasts one step ahead
2. Features are computed from data up to `t−1`
3. XGBoost and LightGBM each produce a probability
4. All three go into the stacking meta-learner

Nothing from step `t+1` onwards is ever visible during the prediction for step `t`.

### Cell 13 — Stacking meta-learner

After collecting predictions for the full test period, a LightGBM meta-learner is trained on:
```
[ARIMA pseudo-prob, XGBoost P(UP), LightGBM P(UP)]
```

This learns the optimal combination — and as the feature importances show, it leans heavily on LightGBM (55.9%) and XGBoost (41.6%) while mostly ignoring ARIMA (2.6%).

Also generates the accuracy comparison bar chart and confusion matrix.

### Cells 14–16 — Evaluation

- **Cell 14**: MCC, ROC-AUC, classification report
- **Cell 15**: McNemar test, paired t-test, binomial test
- **Cell 16**: Predicted probability distributions split by actual class — the visual check that the model genuinely separates UP and DOWN days

### Cell 17 — Save results

Saves full predictions to CSV with columns: `Actual`, `ARIMA`, `XGBoost`, `LightGBM`, `Hybrid_Old`, `Hybrid_Stacked`, and all raw probabilities.

---

## What the meta-learner weights mean

The (0,0,0) ARIMA result is not a failure — it's information. It tells you there's no exploitable linear autocorrelation in large-cap NSE log returns, which is consistent with market efficiency. The stacking meta-learner independently discovered the same thing by assigning 2.6% weight to the ARIMA signal. The tree models are doing all the work by picking up nonlinear momentum and volatility patterns across the 5–10 day feature window.

---

## Notes on reproducibility

- Data is pulled live from Yahoo Finance, so exact numbers may shift slightly if Yahoo adjusts historical prices for corporate actions
- Random seed is fixed at 42 for both XGBoost and LightGBM, so runs should be deterministic given the same input data
- The ARIMA walk-forward refit uses only the last 200 observations at each step — changing this window will change ARIMA results slightly
- The test stock in the repo is set to `HINDUNILVR.NS`. Change `test_ticker` in Cell 3 to `BAJFINANCE.NS` to reproduce the primary paper results

---

## Files generated when you run

| File | What's in it |
|---|---|
| `hybrid_v8_stacked_TICKER_results.csv` | Day-by-day predictions, probabilities, and actual directions for the test period |
| `metrics_stacked_TICKER.csv` | Summary: accuracy, MCC, ROC-AUC |
| `Figure_3.1_Overall_System_Architecture.png` | System architecture diagram |
| `Figure_3.2_Walk_Forward_Validation.png` | Walk-forward validation scheme diagram |

---

## Limitations worth knowing about

- Price and volume only — the model cannot see earnings announcements, RBI policy decisions, or sector news
- The test period (roughly 2022–2025) covers a specific market regime (post-COVID recovery, NBFC expansion, IT sector headwinds). Performance in other regimes is not guaranteed
- Reliance Industries shows marginal significance (p ≈ 0.07) rather than strong significance — likely because its price is driven by too many unrelated factors for price history alone to capture well

---

## Dependencies

```
pmdarima
xgboost
lightgbm
yfinance
statsmodels
scikit-learn
pandas
numpy
matplotlib
seaborn
scipy
```

---

## Citation

If you're using this for academic work:

```
Gorasia, R.; Jain, R. A Hybrid Stacking Ensemble Framework for Short-Term Stock Price
Direction Prediction on the National Stock Exchange of India. Marwadi University
Mini Project (01CE0609), 2025.
```

---

## Acknowledgements

- Data sourced from Yahoo Finance via `yfinance`
- INDIAVIX data from NSE India via Yahoo Finance (`^INDIAVIX`)
- Built as part of the 6th semester Mini Project, Dept. of Computer Engineering, FET, Marwadi University
