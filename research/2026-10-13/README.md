# Implied Volatility Information in Variance Forecasting

Research update recorded October 13, 2026. This extends the earlier surface and return-prediction archive; it does not replace those experiments.

## Question and design

Does 30-day SPX ATM implied variance add information beyond log-HAR for SPY variance forecasts, and does lower forecast loss imply correct calibration?

The historical input archive contains approximately 15.6 million option records over 2010–2023. The daily SPY target proxy combines a Garman–Klass OHLC estimate with squared overnight log returns (cash-dividend adjusted). Horizon labels are the log of the annualized mean of the future daily proxies, excluding the signal day. They are not squared close-to-close returns or high-frequency realized variance.

HAR uses 1-, 5- and 22-session historical variance averages; HAR+IV adds log 30-day ATM implied variance. Chronological rolling estimation only admits matured labels. The original 2023 temporal holdout is a historical holdout: later calibration experiments reused it and are exploratory.

## Recorded findings

| Comparison | Result | Interpretation |
|---|---|---|
| Original holdout log-variance MSE, HAR+IV versus HAR | 24.3% lower at 5 sessions; 35.6% lower at 21 sessions | Incremental variance-forecast information, not return alpha |
| Five-session development-year comparisons | Lower MSE in all 11 calendar years, 2012–2022 | 2012 is a partial development year; calibration's common sample covers 10 years |
| Direct-exponentiation mean bias | −10.2% at 5 sessions; +18.3% at 21 sessions | Ratio of forecast mean to proxy mean minus one |
| Shrunk recursive positive calibration | Five-session bias +0.8%; 21-session bias +29.2% | Exploratory short-horizon mean-bias improvement; not full calibration or a confirmed loss winner |
| Matched-sample 21-session QLIKE advantage over HAR | 32.3% with native IV; 23.2% after one-session delay | Robustness to stale inputs, not proof of actual historical vendor delivery |
| Exclude weekend-expiry nodes | Headline MSE improvements remain about 24.3% / 35.6% | Sensitivity to ambiguous expiry definitions |

The five-session smearing correction moved mean bias from approximately −10.2% to +13.7%, demonstrating possible overcorrection. More recent weighting, multiple-window combinations and alternative calibration were tested under fixed settings. Their relative merits depend on horizon and loss; none establishes a universally effective correction. The 21–22-session calibration problem remains unresolved.

Variance risk premia and Jensen/retransformation effects are relevant mechanisms, but this study does not identify either as the cause of the reported forecast bias. ATM IV squared is not automatically the model-free risk-neutral expected variance.

## Public source and evidence

- `code/`: archived numerical implementations and substantive tests for positive calibration, QLIKE training, recent weighting, window combinations, and label-maturity fences.
- `results/`: selected aggregate metrics and comparisons; no market rows, row-level predictions or vendor files.
- `protocols/`: public copies of recorded specifications, with machine-specific paths and private input hashes removed. These are historical design records, not a new preregistration.

Run the data-independent method checks from this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s code -p 'test_*.py' -v
```

The public code exposes the analysis methods. Full historical reproduction additionally needs authorized optionsDX/CRSP inputs and preceding-stage forecast timelines in the expected schemas. Input contracts remain in the repository's `data/templates/`, `WRDS_BUILD.md` and earlier research archive. Unpublished forecast timelines and local artifacts are deliberately not included. Do not expect the historical CLI runs to execute using this public archive alone.

## Evidence limits and attribution

All follow-up method comparisons use already viewed history. Resampling is conditional on the fitted forecasts; it does not refit every model. Mean calibration and Mincer–Zarnowitz calibration differ, and multiplicity-adjusted loss comparisons must be interpreted separately. Actual vendor file availability and A.M./P.M. settlement identities remain unverified. New-period confirmation is outstanding.

Developed by Linfeng Zhao with substantial OpenAI Codex assistance. Data sources: optionsDX SPX options and CRSP SPY. Methods build on HAR, forecast calibration and retransformation literature; no novel volatility estimator or proven trading strategy is claimed.
