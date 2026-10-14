<img src="assets/social-preview.jpg" width="560" alt="IV Surface Lab cover">

# IV Surface Lab

**Implied-variance information, forecast calibration, and the limits of cross-horizon corrections.**

A reproducible research engine for implied-volatility surfaces, HAR benchmarks, skew changes, variance-risk-premium proxies, rolling forecasts and cost diagnostics. The repository also preserves the research protocols and aggregate findings from exploratory SPX-options / SPY experiments.

## Latest research: October 13, 2026

**[Implied Volatility Information in Variance Forecasting](research/2026-10-13/README.md)** studies 15.6M SPX option records over 2010–2023 and SPY Garman–Klass variance proxies augmented with squared overnight returns.

- Adding 30-day implied variance reduced original holdout log-variance MSE by **24.3% / 35.6% at 5 / 21 trading sessions**; five-session MSE was lower in all 11 development calendar years.
- Lower losses coexisted with horizon-dependent level bias. Exploratory shrunk recursive calibration narrowed five-session mean bias from **−10.2% to +0.8%**, but worsened 21-session calibration.
- A one-session IV delay retained a **23.2% 21-session QLIKE advantage over HAR** (32.3% without delay). Excluding weekend-expiry nodes barely changed headline MSE improvements.

The update includes method code, data-independent checks, historical protocols, and aggregate comparison tables. Later calibration work reuses the historical sample; it does not supply a fresh holdout or establish trading alpha. The original skew-return research and its weak return-prediction evidence remain in the archive below.

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

- [October 2026 calibration and robustness update](research/2026-10-13/README.md)

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
