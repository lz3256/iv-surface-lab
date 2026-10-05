# v3 fixed exploratory extension

Written after v1/v2 results and before v3 outcome calculations. All 2010–2023 data are reused. Former development/holdout are descriptive partitions, not untouched samples. No purchase or inspection of 2024–2026 outcomes.

## Daily models

Preserve v2 R0–R3 forecasts, refit dates, controls and original train eligibility. No new hyperparameter search. Controls are ret1, ret5, log RV22, log IV30 variance, term slope, log HAR21. At each stored v2 refit, use its last 1008 eligible known-label observations (minimum 504).

R3u_raw = saved R2 + beta*z. On the training rows only: winsorize skew change at 1/99%; standardize controls; project clipped skew on an intercept and controls by unpenalized least squares; z is the residual. beta=z'y/(z'z), with no shrinkage. Apply frozen bounds/projection/beta to forecasts until next v2 refit. This beta equals the training partial-OLS coefficient via FWL. There are estimated nuisance parameters; it is not a guaranteed minimum-variance estimator. Keep this precise additive prediction distinct from a standalone one-variable return model or fully unpenalized control forecast.

R3u_raw_matched and R3u_same use identical subsets of the original refit's training rows where both raw and same-endpoint signals exist (minimum 504). Fit separate training-only clipping, residual projections and unpenalized beta, then add each to the same saved R2 baseline. Test on identical dates. No moving the training window farther back to compensate for missingness. Rank deficiency uses SVD least squares; residual variance <=1e-16 makes that fit unavailable, never a fabricated zero coefficient.

Horizons 1/5/21; same labels, calendar and 21-session v2 refits. Save beta, residual variance, rank, bounds, projection parameters, training first/last/max label time, forecast z and incremental prediction. Report four comparisons per horizon per stage: raw vs R2, raw_matched vs R2, same vs R2, same vs raw_matched. Holm over all 12 tests per stage; report all, no best-variant selection.

## Same endpoints

For each put25/call25 leg, map yesterday's normal interpolation pair to actual vendor expiry-date identifiers. On today and the immediately previous exchange session, interpolate total variance at 30 days using that identical pair, with each day's own vendor DTE and delta-node IV. Subtract the two skew levels. Keep today's normal nodes for controls and reset the reference pair each session; do not cumulatively integrate the adjusted differences.

Require both days' original main-surface quality checks to pass, both endpoint quotes on both days, and the pair to bracket 30 on both days. No extrapolation, zero padding or bridging missing sessions. A single exact-30 endpoint yesterday generally fails today and is missing. Preserve reasons and exclusion counts. These are vendor expiry aggregates, not verified SPX/SPXW contract roots; delta strikes can change, maturity weights still roll, and the method does not eliminate all possible settlement, liquidity or interpolation effects.

Compare calendar group means/variances for raw and adjusted signals on the same dates, and raw on all valid dates. Inspect normal endpoint switch days involving a third-Friday/previous-Thursday calendar proxy separately from other switch days and unchanged brackets. The proxy is not an AM identifier. Report subperiods 2010–2012, 2013–2019, 2020–2023 and target-date+30 proximity. No causal claim or post hoc choice of groups.

## Precision and threshold

Declare delta=0.001 (0.1 percentage point of incremental R²) for daily horizon=1. Delta is an illustrative economic relevance threshold, not a transaction-cost-adjusted tradability threshold. Sharpe sqrt(252*delta) is an idealized independent-homoskedastic optimal-use heuristic, not a general mapping from conditional loss gain over R2 to realizable Sharpe.

Incremental R² = mean(loss_baseline-loss_added)/mean(loss_baseline). First report v1/v2 saved 95% loss-gain intervals divided by their fixed sample baseline MSE, as a transparent unit conversion. Then for all daily comparisons, bootstrap the paired ratio jointly (including its denominator) on the uncompressed session grid: circular blocks max(20,h), 5000 repetitions, seed42. Report percentile 95% interval and one-sided 95% upper bound (95th percentile). A centered bootstrap test uses the same ratio distribution; also report Holm for zero-gain tests. No rounding to [0,0].

For h=1: upper bound < delta supports only that this specified forecast algorithm's gain is below the chosen threshold, conditional on the resampling assumptions. A 95% interval wholly inside [-delta,+delta] is a conservative practical-equivalence screen, not an assertion of absent latent information. Simultaneous Bonferroni upper bounds over the four daily comparisons within each stage are also reported. CI width is precision, not a minimum detectable effect or power calculation. Do not label these bounds as a formal MDE; no power simulation is claimed.

## Separate event exploration

Collect CPI release dates/times from official BLS historical schedules and scheduled FOMC meeting/statement dates from official Federal Reserve archives/calendars, restricted to 2010–2023. Exclude emergency/conference-call meetings; preserve source records and missing/ambiguous dates. Verify intraday timing with official statements where available; do not assume all historical FOMC releases were at 14:00.

Calendar build before outcomes: 168 CPI releases and 111 scheduled FOMC releases. 63 FOMC statements print a clock time. For 23 dates from March 2013 through 2015 whose HTML only says immediate release, use the Federal Reserve's published historical description of the March-2013 switch to 14:00, separately flagged as a timing-rule source. Leave 25 earlier FOMC times unverified and exclude them; they precede the first available R3u forecasts anyway. Store the official URLs and source-response hashes, without fetching any 2024–2026 outcome data.

Use h=1 open-to-next-open windows containing the release. CPI at 08:30 maps to entry at the previous exchange-session open; FOMC scheduled daytime statements map to event-day open. Signal must be available before entry and before release. This uses older EOD skew for CPI; it is not a post-CPI-open reaction trade. Calendar membership is a retrospective scheduled-date assumption, not independently verified point-in-time schedule availability.

Use the saved out-of-sample raw residual z from R3u_raw as the abnormal-skew predictor. At each event, train only on prior same-type events whose labels are known, latest 80, minimum24. On that event training sample fit return-minus-saved-R2 on intercept plus z, unpenalized. Baseline = R2 + past event mean error; added = baseline + beta*(z-current training mean z). Thus the comparison does not mistake an event mean premium for skew's slope. Report CPI/FOMC separately on all available 2010–2023 evaluation dates, without optimizing event windows or using the former holdout to tune.

For event uncertainty use paired circular blocks of 4 successive event observations, 5000 reps, seed42; Holm over the two event-type zero-gain tests. Event samples and dependence differ from daily models; daily 0.1%/Sharpe heuristic is not applied to event-only observations. Event labels, train/end/release timing and minimum sample rules are audited. This is an additional exploratory hypothesis, not a new confirmation exercise.
