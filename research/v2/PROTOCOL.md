# v2 exploratory patch protocol

Written after inspecting v1 results and before any v2 results. All 2010–2023 observations, including the former holdout, are now reused exploratory data. The old development/former-holdout partition is retained only for side-by-side diagnostics. Nothing here creates a new confirmation sample.

Scope of this patch: H2 feature transformation, training-only skew-change winsorization, stronger fixed ridge, and third-Friday/expiry-bracket diagnostics. No claim to have repaired AM/PM settlement identification: contract roots and actual settlement timestamps are absent. No fabricated FOMC/CPI release calendar is introduced in this patch.

- Use v1 daily inputs and exclusions unchanged. Preserve missing sessions.
- H1 and HAR generation remain v1 settings, including HAR alpha=10. H2 positive variance features change to log(rv_22), log(iv_var_30), log(har_fwd_21), floor=1e-10. Past returns and term slope remain unchanged.
- R0: rolling historical mean. R1: ret_1, ret_5, log_rv_22, log_iv_var_30, term_slope. R2: R1 + log_har_fwd_21. R3: R2 + skew_change.
- At each refit, select only valid feature rows whose target information time is strictly before the fit signal time. Then fit skew-change 1%/99% quantile bounds on those training rows, clip both training and later forecast values using those frozen bounds, and compute training means/scales after clipping. No full-sample winsorization.
- H2 penalty is fixed alpha = 1.0 × actual training sample count. Equivalently use mean squared training loss + lambda ||beta||² with lambda=1 on standardized slopes and an unpenalized intercept. No CV or parameter sweep. This is a chosen exploratory stress test, not an assertion of optimal regularization.
- Minimum 504 valid past samples, window latest 1008 valid samples, refit every 21 session rows. Horizons 1/5/21. Same original dividend-adjusted targets and availability assumptions.
- Report all R0/R1/R2/R3 MSEs on common observations. Compare R3 vs R2 and R3 vs R0 for all three horizons, preserving the calendar for paired bootstrap. Also compare R3 against v1 R3 on matched dates. Report within-stage Holm over all nine comparisons, with no primary-confirmatory label. Bootstrap: 1000, block max(20,horizon), seed42.
- Preserve former development and former holdout tables separately. Do not choose a winner or select new parameters using the former holdout.
- Expiry diagnostics: group skew change by signed calendar distance to nearest third Friday; also inspect date+30-day proximity to third Friday and changes in interpolating expiry brackets. A calendar-monthly endpoint proxy is never a verified AM flag. Group differences do not identify settlement causality or rule out monthly macro/flow effects.
- Outlier diagnostics reconstruct v1 actual refit-window standardization and coefficients; distinguish raw feature maxima from training-window z-scores. Decompose the loss relative to R0 by year/month and largest loss days.
- Six extreme May 2019 gaps remain quarantined. They cannot be attributed to AM/PM merely because AM/PM is unverified.

Next work after this patch: source verified contract roots/settlement definitions; build separately documented event-calendar variants using historical scheduled/actual release times; freeze a revised protocol before inspecting genuinely new data. New-data availability, licensing and untouched status must be verified before any purchase or confirmation claim.
