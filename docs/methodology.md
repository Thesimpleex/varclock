# Methodology

This note collects the derivations and conventions behind varclock. Notation follows the README.

## 1. Model-free variance

Let $S_T$ be the terminal price, $F=\mathbb E[S_T]$ the forward and $K_0\le F$ any reference strike. For twice-differentiable payoffs, a log contract is replicated by out-of-the-money options,

$$
-2\,\mathbb E\!\left[\ln\frac{S_T}{F}\right]
= 2\,e^{RT}\!\left[\int_0^{K_0}\frac{P(K)}{K^2}\,dK+\int_{K_0}^{\infty}\frac{C(K)}{K^2}\,dK\right]-\left(\frac{F}{K_0}-1\right)^2+\text{(third-order terms)},
$$

where the quadratic term corrects for the discrete choice of $K_0$ (Demeterfi et al. 1999, appendix; Cboe's formula keeps the second-order term). If prices have continuous paths, $-2\,\mathbb E[\ln(S_T/F)]=\mathbb E\int_0^T v_t\,dt$, the expected integrated variance. For jumps the log contract measures something slightly different: for a single Gaussian log-jump with variance $\delta^2$ at a known date, $S_T$ is lognormal and the replicated total variance is exactly $\sigma^2T+\delta^2$ whatever the jump mean. This is the closed-form check of the clock fit.

The integral becomes the Cboe sum $\frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2} e^{RT}Q(K_i)$ over the selected strikes. Two errors enter: truncation of the wings, which vanishes as the strike range widens, and discretisation of the integral, which is second order in the strike spacing. Test results: the error falls by a factor of 4.00 for every halving of the spacing, for a flat smile and for Fourier-priced Heston chains, where the target is $\theta T+(v_0-\theta)(1-e^{-\kappa T})/\kappa$. The discretisation error is the same for both models to six digits (6.67e-05 at a spacing of 2), because its leading term comes from the kink at $K_0$ and does not depend on the model; the Heston test therefore adds the check that the converged level equals the closed form, not a second convergence rate.

## 2. Cboe conventions as implemented

**Contracts.** AM-settled SPX options (09:30 on the third Friday) and PM-settled SPXW options (the close, 16:00 or 13:00 on early-closing days). VIX and VIX9D use SPX plus the end-of-week SPXW series, without the SPXW expiring on a date that also has an AM-settled SPX series; VIX3M uses SPX plus the end-of-month SPXW series; VIX1D uses all SPXW.

**Terms.** Bracket method (VIX, 30 days; VIX3M, 93 days): the near term is the latest expiration with at most the target number of days, the next term the following one. Nearest-term method (VIX9D, after excluding series with fewer than two days; VIX1D): the nearest remaining expiration, and the one after it. VIX1D uses the same-day series as near term while it lives; after 16:00 only the next-day series remains, whose 390 business minutes are below the 405-minute target, and the index equals $100\sqrt{\sigma_2^2}$.

**Time.** Calendar indices count whole elapsed minutes to the deemed expiration over $365\times1440=525{,}600$. VIX1D counts minutes inside sessions that run from 09:30 to the close plus 15 minutes, over $252\times405=102{,}060$; from the 16:15 close to the next 16:00 expiry that is 390 minutes. Elapsed real minutes are used across daylight-saving changes (the methodology does not say): the difference is taken between UTC instants, so a term that crosses the change in November is an hour longer than its wall-clock span. A test pins this for the weeks around 1 November 2026 and 8 March 2026. On the 2026-10-02 snapshot the convention moves the VIX by 0.004 points (15.313 instead of 15.318).

**Rates.** Treasury par yields (CMT) of the previous business day at the 12 maturities of 30, 60, 91, 182, 365, 730, 1,095, 1,825, 2,555, 3,650, 7,300 and 10,950 days. A natural cubic spline is clipped between the two neighbouring yields; below 30 days it is clipped between two lines through the first node whose slopes come from the next higher and next lower yield. The bond-equivalent yield $y$ becomes $r=\ln\!\big[(1+y/2)^2\big]$. The day count is the number of calendar days from the curve date to the expiration date; this reproduces the rates printed in Cboe's samples ($0.031664\%$, $0.028797\%$, $0.2898\%$, $0.5808\%$). The sample for the one-day index prints $0.0393\%$ and $0.0390\%$, which correspond to day counts one larger; the effect on that index is below $10^{-9}$ and the test documents it.

**Forward and $K_0$.** Among strikes with two-sided call and put quotes (positive bid and ask on both sides, not crossed), the one with the smallest $|C-P|$ of mids, the lowest strike on ties, gives $F=K^*+e^{RT}(C-P)$; $K_0$ is the largest strike at or below $F$. The requirement of positive bids at the forward strike is a robustness choice for scraped data; Cboe's own feed never violates it near the money.

**Strike selection.** Puts below and calls above $K_0$, outward; options with zero bid or zero ask are skipped, and two consecutive such strikes end the walk. Crossed quotes are treated as missing. $\Delta K_i$ is half the distance between the neighbouring *selected* strikes (one-sided at the ends), so skipping a strike widens its neighbours' $\Delta K$.

**Interpolation.** As in the README. For VIX1D, if the near term has under 60 business minutes left the official calculation freezes its variance; a snapshot cannot, so varclock warns instead.

## 3. The forward-variance strip

The strip uses the longest run of SPXW expirations on consecutive trading days. The first expiration contributes its total variance; every later increment is the difference of two totals, $\Delta w_j=w_j-w_{j-1}$, and the cumulative totals handed to the clock fit are the totals themselves. Each increment carries a half-width equal to the sum of the bid-ask half-widths of the two totals, where the half-width of a total is $\sum_i w_i(\text{ask}_i-\text{bid}_i)/2$ plus the first-order effect of the forward's bid-ask uncertainty on the $(F/K_0-1)^2$ term. An expiration whose forward strike lacks a valid quote is skipped and recorded.

**Strike coverage.** Cboe's truncation stops each strip at its own pair of zero bids, so adjacent expirations cover different strike ranges. Two treatments are implemented. *Own range* (default): each total uses the expiration's selected strikes, and the wing variance that truncation leaves out stays out of every total. *Common window* (`coverage="common"`): for adjacent expirations the strike windows are intersected ($[\max(\text{lowest}),\min(\text{highest})]$ over the two selected ranges) and both variances are recomputed on that window with the original forward and $K_0$ and one-sided $\Delta K$ at the new ends; the cumulative totals are the running sums of these increments. The common window removes the later expiration's wing variance beyond the earlier expiration's range from each increment, and these losses add up along the strip: the cumulative variance is biased down by an amount that grows with the horizon, and the weekend term, fitted as a constant on top of a rising base, is biased up. This was found with a truncated synthetic truth (Black-76 chains with a known weekend weight, options below a tick priced with a zero bid): at a tick of 0.05 index points own-range totals are within 0.6% of the truth and the weekend weight within 1%, while the common window is off by 4.3% and +35%. On the real snapshots the weekend weight is 0.07 (Cboe) and 0.11 (Yahoo) on own ranges and 0.11 and 0.15 on the common window. The common window remains available as a sensitivity run.

## 4. The variance clock

**Model.** With sessions $s=1,\dots,S$ between the snapshot and the last expiration and $j(s)$ the expirations,

$$
w_j=\sum_{s\le j}v_s,\qquad v_s=f_s\,b(s)+W\,\mathbb 1[\text{weekend before }s]+H\,n^{\text{hol}}_s+\sum_e J_e\,x_{e,s}.
$$

A "weekend before $s$" means that the previous trading day is separated from $s$ by a Saturday and a Sunday; $n^{\text{hol}}_s$ counts weekday market holidays in the gap. Events are assigned to the session whose variance they move: the same day if released before the close, otherwise the next trading day (a 20:00 election night, a 16:05 earnings report, a Saturday report).

**Estimation.** Parameters are $b$'s coefficients, $W$, $H$ (only if the horizon contains a holiday) and the $J_e$, all constrained to be non-negative. For the mean-reverting family $b(s)=b_0e^{-ks}+b_\infty(1-e^{-ks})$ with $b_0,b_\infty\ge0$, $k$ is chosen on a grid by the smallest weighted residual norm (a one-dimensional profile); for the spline family $b$ is piecewise linear on equally spaced nodes with non-negative node values and a second-difference penalty scaled by 2% of the average base per session. Rows are weighted by $1/\sqrt{h_j^2+(0.02\,w_j)^2}$, with $h_j$ the quote half-width of the total, so that no expiration is trusted beyond 2% for reasons the quotes do not show. The problem is solved by the Lawson-Hanson algorithm (`scipy.optimize.nnls`).

**Why the base is not flat.** The term structure of implied volatility is not flat over a few weeks; a flat base would put the whole rise into event terms. On the 2026-10-02 snapshot the base volatility rises from 9.9% to 17.1% across 27 sessions.

**Bootstrap.** Each replicate (i) redraws every option price inside its bid-ask band, uniformly in the margin, with the positions within one expiration linked by a Gaussian copula of correlation $\rho=0.5$ (a common factor $z_j\sim N(0,1)$ per expiration: $u_{ji}=\Phi(\sqrt\rho\,z_j+\sqrt{1-\rho}\,\varepsilon_{ji})$), and the forward from the at-the-money quotes, recomputing the totals on the fixed strike selection, (ii) adds residuals resampled with replacement from the standardized residuals of the fit, around the fitted curve, and (iii) refits, including the profile over $k$. Reported intervals are 5th and 95th percentiles. With independent draws the quote noise of a total averaged out over hundreds of strikes (0.1% to 0.3% of the total against a quote band of 1.8% to 3.3%); with $\rho=0.5$ it is 0.7% to 1.4%, and the quote and residual components are of similar size. $\rho$ is a modelling assumption, not an estimate. The intervals describe estimation uncertainty given the model family; the sensitivity runs in the README show how much more the family itself matters.

**Identification.** One bump parameter per event is identified by a spike of the increments relative to the smooth base; with 27 expirations and 6 to 9 free parameters it is weakly identified when the base can bend locally, which is why the spline runs give smaller election and FOMC bumps than the mean-reverting base.

## 5. Event study

For an event on session $t$ with previous close $t-1$: implied variance $\hat v_t=(\text{VIX1D}_{t-1}/100)^2\,M_t/102{,}060$ with $M_t$ the business minutes from the 16:15 close of $t-1$ to the close of $t$ (390 on ordinary sessions); realised variance $r_t^2$ with $r_t=\ln(S_t/S_{t-1})$ from the Cboe SPX history. Statistics are means over events with percentile bootstrap intervals (10,000 draws, seed 0; events are resampled one by one, the ordinary-day sample, a run of consecutive days, by a circular block bootstrap with blocks of 20 sessions): the ratio of mean realised to mean implied variance, the mean difference (premium) in squared percent, the mean of $r_t^2/\hat v_t$, and the event-day to ordinary-day multiples of implied and realised variance (events and ordinary-day blocks resampled). Ordinary days are all trading days in the sample without a CPI, FOMC or employment event. The close of the index (16:15) and the S&P 500 close (16:00) differ by 15 minutes. After a weekend or holiday the business-minute convention puts the closed days into the next session's implied variance.

## 6. Arbitrage-free bands

For the options selected by the Cboe rule, the variables are the undiscounted call values $u_i$ and the forward $F$ (puts follow from parity $p_i=u_i-(F-K_i)$). Constraints: each quote $\text{bid}_i/D\le u_i\le \text{ask}_i/D$ (calls above $K_0$, puts below, both at $K_0$), $u$ decreasing and convex in $K$, slope at least $-1$, $u_i\ge F-K_i$, and $K_0\le F\le K_0+$ gap. The variance estimate is linear in $(u,F)$ apart from $(F/K_0-1)^2$, which is convex in $F$: the chord over the admissible interval bounds it from above (used for the minimum) and the tangent at the observed forward from below (used for the maximum), so both bounds are valid outer bounds. The discount factor $D$ is fixed at its Treasury value. If the quotes admit no arbitrage-free curve, every quote is widened by the smallest common amount (found by bisection) that does, and this amount is reported.

## 7. Adapters

Yahoo Finance (`yfinance`) returns bid and ask per contract; the contract symbol identifies SPX and SPXW. Its quotes stop at the 16:00 equity close, so the valuation time of a Yahoo snapshot taken after the close is 16:00 rather than 16:15. The evidence is the forward implied by the quotes: 7,722.80, the 16:00 index close, against 7,729.40 in the Cboe feed at 16:15. The Cboe delayed-quotes JSON gives the full chain in one request. Both snapshots are cached with their download time; after hours the valuation time is the last session end, during the session the download time minus 15 minutes.
