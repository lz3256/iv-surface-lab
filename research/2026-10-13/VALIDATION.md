# Publication validation — October 13, 2026

10 substantive method tests passed. They cover calibration recovery, independent optimizer comparisons, units, label-maturity fences, recency weighting and common-sample timing.

```bash
python -m unittest discover -s research/2026-10-13/code -p 'test_*.py' -v
```

All published Python sources passed syntax checks. Archived aggregate tables were copied from the completed recorded experiments; publication did not retrain models or rerun historical market evaluations. Private host paths, raw market records, row-level predictions, trained weights and credentials are excluded. Public path adaptation preserves the numerical method definitions.
