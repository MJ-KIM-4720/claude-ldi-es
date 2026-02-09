# CLAUDE.md

## Project Overview

**ES-Constrained Liability-Driven Investment (LDI) Model**

Academic research extending Jo et al. (2025) VaR-LDI model to Expected Shortfall (ES) constraints using the option-based approach of Kraft & Steffensen (2013). Target journal: JEDC.

The core contribution: ES eliminates gambling incentives that VaR creates for underfunded pension funds, by providing partial linear protection in the tail (g = cy, c > 1) instead of abandoning protection entirely (g = y).

## Repository Structure

```
ldi/                    # Main package
├── __init__.py
├── params.py           # All parameters + derived quantities (r_tilde, sigma_Y, Pi_star, etc.)
├── bs_utils.py         # Shared Black-Scholes functions (put, digital put, deltas)
├── es_model.py         # ES constrained model
├── var_model.py        # VaR constrained model (Jo et al. 2025)
└── compare.py          # Cross-sectional/time-series comparison & plotting
run_all.py              # Entry point: `python run_all.py`
```

## Key Parameters (CRITICAL)

```
R = 0.02  (real interest rate)
r = 0.04  (nominal risk-free rate)
```

**DO NOT swap R and r.** With correct values: Merton total ≈ 84%, r_tilde = -0.023, sigma_Y = 0.0711. Swapping gives unrealistic Merton = 650%.

Default constraint parameters: `alpha = 0.05` (VaR), `epsilon = 0.10` (ES), `T = 5`, `k = 1.0`, `gamma = 3.0`.

## Model API

Both ES and VaR models expose the same interface:

```python
from ldi import es_model as ES, var_model as VaR, params as P

# Cross-sectional: each fund solves its own threshold
A = ES.cross_sectional_A(y0=0.8, eps=0.10)    # returns scalar
A = VaR.cross_sectional_A(y0=0.8, alpha=0.05)

# Time-series: fund solves threshold once, A varies as Y evolves
k_eps, c, binding = ES.solve_threshold(y0=1.0)
A = ES.adjustment_factor(Y=0.9, k_eps=k_eps, c=c, tau=4.0)

k_alpha, binding = VaR.solve_threshold(y0=1.0)
A = VaR.adjustment_factor(Y=0.9, k_alpha=k_alpha, tau=4.0)

# Optimal portfolio weights: pi* = A · Pi_star
pi_S, pi_I = ES.optimal_portfolio(Y, k_eps, c, tau)
```

## Mathematical Notes

- **ES constraint:** `c · Put(y0, k_eps) = epsilon` where `c = k / k_eps`
- **VaR threshold:** `k_alpha = y0 · exp(m_P·T + sigma_Y·sqrt(T)·Phi^{-1}(alpha))` (P-measure)
- **Adjustment factor:** `A = (Y / Psi) · (dPsi/dy)` — multiplies Merton weights
- **ES key property:** A ≤ 1 always (structural, because g_ES ≥ Y everywhere)
- **VaR key property:** A > 1 possible for underfunded funds (gambling incentive from digital option)
- All Black-Scholes pricing uses liability-adjusted rate `r_tilde = r - (beta_0 + beta_1 * mu_I)`

## Conventions

- **Language:** Python 3.10+, numpy, scipy, matplotlib
- **Cross-sectional analysis** = different pension funds at t=0 with varying y0, each solving own threshold
- **Time-series analysis** = single fund over time, threshold fixed at t=0, Y evolves stochastically
- Figures saved to `outputs/` at 150 dpi
- Use `brentq` for ES threshold solving, closed-form for VaR threshold
- All monetary values are in funding ratio units (F = X/L, dimensionless)

## Common Tasks

- **Change parameters:** Edit `ldi/params.py` — derived quantities auto-compute on import
- **Add sensitivity analysis:** Add function in `ldi/compare.py`, follow `plot_eps_sensitivity` pattern
- **Add new constraint model:** Create `ldi/new_model.py` mirroring `es_model.py` API
- **Monte Carlo simulation:** Use P-measure drift `m_P` and vol `A · sigma_Y` for fund dynamics

## Known Results

| y0   | VaR A | ES A  | Interpretation                          |
|------|-------|-------|-----------------------------------------|
| 0.1  | 1.34  | 0.62  | VaR gambles, ES conservative            |
| 0.9  | 0.61  | 0.70  | VaR trough (lock-in), ES moderate       |
| 1.0  | 0.67  | 0.85  | Both constrained, ES less so            |
| 1.5  | 1.00  | 1.00  | Both non-binding → Merton              |

## References

- Jo, Kim, Jang (2025) — VaR + LDI + inflation risk (Applied Economics Letters)
- Kraft & Steffensen (2013) — Option-based VaR/ES (European J. Operational Research)
- Basak & Shapiro (2001) — VaR + ES constraints (Review of Financial Studies)
- Gabih, Grecksch, Wunderlich (2005) — Expected Loss constraint (Stochastic Analysis and Applications)
