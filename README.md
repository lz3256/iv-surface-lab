<img src="assets/social-preview.jpg" width="560" alt="IV Surface Lab cover">

# IV Surface Lab

**Separate volatility forecasting from return prediction — and audit every time boundary.**

A reproducible research engine for implied-volatility surfaces, HAR benchmarks, skew changes, variance-risk-premium proxies, rolling forecasts and cost diagnostics. The repository also preserves the research protocols and aggregate findings from exploratory SPX-options / SPY experiments.

## Work completed

The implementation covers a shared surface/price/calendar pipeline, interpolation and node-quality checks, and rolling return/variance forecasts against HAR and price-history benchmarks. Real-data development and holdout comparisons document weak return-prediction evidence. Later residualized-skew and same-expiry extensions explicitly disclose reuse of the original sample.

## Research highlights

- A shared data schema for surface nodes, prices, calendars and events.
- Training restricted to labels known at each forecast time.
- HAR and nested return/variance models, block bootstrap and multiple-comparison reporting.
- Surface-node quality checks, fixed-tenor interpolation audits and dividend-aware labels.
- Later exploratory extensions for residualized skew and same-expiry endpoint changes.

## What the recorded research found

The initial real-data experiment did not find stable incremental SPY return-prediction value from changes in 30-day 25-delta skew. The specified one-day comparison improved holdout MSE by only 0.087%, with p=0.590.

IV levels showed useful variance-forecasting information relative to HAR in those experiments, but stronger significance claims depend on the multiple-comparison family. Forecasting volatility must not be presented as forecasting market direction.

The later v3 work reuses the 2010–2023 sample. Its former holdout is no longer an untouched test set, and the v3 findings are exploratory.

## Run a synthetic demo

Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
```

The command creates a new timestamped directory under `runs/` and prints the report path. Demo data contain deliberately implanted relationships and do not represent empirical market evidence.

Run the core checks:

```bash
python -m unittest discover -s tests -v
```

## Research archive

- [Initial empirical results](research/v1/研究结果.md)
- [Initial data audit](research/v1/数据核验.md)
- [v2 research notes](research/v2/研究说明.md)
- [v3 exploratory findings](research/v3/研究说明.md)
- [v3 fixed protocol](research/v3/PROTOCOL.md)
- [Data preparation guide](WRDS_BUILD.md)
- [Full engine guide](GUIDE.zh.md)

The public archive contains model code, protocols and selected aggregate comparison tables. It excludes vendor raw data, processed market inputs and row-level prediction snapshots. Reproducing the real-data extensions requires supplying the corresponding inputs from your own data access; the synthetic demo runs independently.

## Structure

`ivsurf/` is the v3.2 research engine; `tests/` contains its checks; `data/templates/` documents input schemas; and `research/v1`, `research/v2`, `research/v3` preserve the experiment-specific code and reports.

## Credits

Project owner: [Linfeng Zhao (@lz3256)](https://github.com/lz3256).

Developed with substantial assistance from OpenAI Codex for implementation, analysis, debugging and documentation. Recorded experiments and limitations are described in the linked reports.

Recorded empirical experiments use optionsDX SPX option-chain data and CRSP SPY data. Vendor raw data and processed market inputs are excluded from the public repository. The synthetic demo is clearly labeled and does not represent empirical performance.
