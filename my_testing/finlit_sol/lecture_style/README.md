
# Financial Literacy DDC — Rust (NFXP) + Hotz–Miller (NPL)

This repo contains a lecture-style structural model of financial participation and risky-share choice,
implemented both as **NFXP** (full-solution MLE) and **NPL** (Hotz–Miller pseudo-likelihood).

## Folder layout
- `model_finlit.py` – core model (states, actions `(d, s, k)`, GH integration, transitions, Bellman).
- `Solve_NFXP.py` – Rust (1987) NFXP estimation.
- `Solve_NPL.py` – Hotz–Miller / NPL estimation (σ fixed).
- `finlit_run_stats.ipynb` – notebook to run the code and produce lecture-style statistics & plots.

You provide the panel CSV (2 waves): `synthetic_finlit_portfolio.csv` with columns:
```
id, wave, income_eur, liquid_assets_eur, literacy_index_z, age, participate_risky, risky_share
```
Assumptions: `wave ∈ {1,2}`, `participate_risky ∈ {0,1}`, `risky_share ∈ [0,1]` (0 if no participation).

## Quick start

```bash
# NFXP (full-solution MLE)
python Solve_NFXP.py --data synthetic_finlit_portfolio.csv

# NPL (Hotz–Miller)
python Solve_NPL.py --data synthetic_finlit_portfolio.csv
```

Common flags:
- State bins: `--Ka 5 --Ky 5 --KL 3 --Kage 3`
- Primitives: `--beta 0.96 --rf 0.01 --muR 0.05 --sdR 0.15`
- Grids: `--risk_grid "0,0.25,0.5,0.75,1.0"` and `--save_grid "0,0.1,0.2,0.3,0.4,0.5"`
- NPL starts: `--F0_start 25 --phi_start 10`

## Outputs (guaranteed)
- `finlit_estimates_<method>.csv` – parameter table.
- `V_hat_<method>.npy` – value function by state.
- `CCP_hat_<method>.npy` – conditional choice probabilities.
- `policy_<method>.csv` – most likely observed action (d,k) per state (`policy_obs_idx`).
- (NPL) `CCP_emp.npy` – a simple diagnostic proxy.

## Notebook
Open **`finlit_run_stats.ipynb`** to:
- run NFXP or NPL from the notebook,
- show estimates,
- compare **observed vs model** participation & risky-share distributions,
- plot a couple of lecture-style bar charts.

## Troubleshooting
- If you don’t see `CCP_hat_*.npy` or `policy_*.csv`, re-run the solver. They are always saved.
- Dataclass default errors: we use immutable tuples for grids.
- If your CSV has different names, adapt the loader in `model_finlit.load_panel`.

