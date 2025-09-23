
# Financial Literacy Structural Estimation (Rust/NFXP + Hotz–Miller NPL)

This package contains a runnable script and a small analysis notebook that follow the lecture-style
dynamic discrete choice setup for financial participation and risky-share choice.

## Files
- `finlit_structural.py` – estimation script (NFXP and NPL).
- `finlit_npl_full.py` – estimation script (fully NPL).
- `finlit_analysis.ipynb` – quick plots for policy and CCPs after running the script.
- (You provide) `synthetic_finlit_portfolio.csv` – 2-wave panel (id, wave).
  Required cols: `id, wave, income_eur, liquid_assets_eur, literacy_index_z, age, participate_risky, risky_share`.

## Run

```bash
# --> finlit_structural.py
# NFXP (full-solution MLE)
python finlit_structural.py --data synthetic_finlit_portfolio.csv --method nfxp

# NPL (Hotz–Miller pseudo-likelihood, sigma fixed)
python finlit_structural.py --data synthetic_finlit_portfolio.csv --method npl


# --> finlit_npl_full.py
# Run NPL with acceleration (default) and warm starts
python finlit_npl_full.py --data synthetic_finlit_portfolio.csv

# Tweak settings if you like
python finlit_npl_full.py --data synthetic_finlit_portfolio.csv \
  --Ka 5 --Ky 5 --KL 3 --Kage 3 \
  --beta 0.96 --rf 0.01 --muR 0.05 --sdR 0.15 --sigma 2.0 \
  --F0_start 25 --phi_start 10

# Turn off Anderson acceleration (for debugging):
python finlit_npl_full.py --data synthetic_finlit_portfolio.csv --no_accel


# --> finlit_structural_unified.py
# NFXP (full-solution MLE, Rust 1987)
python finlit_structural_unified.py --data synthetic_finlit_portfolio.csv --method nfxp

# NPL (Hotz–Miller) with warm starts + Anderson acceleration
python finlit_structural_unified.py --data synthetic_finlit_portfolio.csv --method npl
```

Common options:
- `--Ka --Ky --KL --Kage` (state bin counts)
- `--beta --rf --muR --sdR` (discount, returns)
- `--sigma` (CRRA, used by NFXP; NPL holds sigma fixed)

## Outputs
- `finlit_estimates_*.csv` – parameter table.
- `V_hat_*.npy`, `CCP_hat_*.npy` – value function and conditional choice probs by state.
- `policy_*.csv` – most likely action per state.

## Notes
- Savings `a'_d` is not observed; the script uses an **action-specific savings-rate proxy** from the panel
  (next-period liquid assets / current resources), trimmed to [0, 0.9]. Replace with a proper `a'` grid
  if you prefer a fuller structural savings choice.
- Transitions `π(x'|x,a)` are estimated nonparametrically from wave 1→2 using Laplace smoothing.
- Flow utility is *expected* utility over risky return via 6-point Gauss–Hermite.
