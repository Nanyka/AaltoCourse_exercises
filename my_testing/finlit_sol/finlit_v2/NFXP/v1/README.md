
# NFXP — Financial Literacy (CRRA + Budget) — Minimal Working Example

This folder contains a **ready-to-run NFXP** estimation for a dynamic portfolio‑share choice model, using your data file
`toy_panel_with_random_x copy.csv`.

## Model (short)

- **State**: assets `a` (discretized on a grid). Future income is summarized by the sample mean `ȳ` from your data.
- **Action**: portfolio share `x ∈ {0, 0.25, 0.5, 0.75, 1}` invested in a risky asset.
- **Budget/Transition**: end‑of‑period wealth is
  \[ w' = a·r_p(x) + y, \quad r_p(x) = x·R + (1-x)·R_f \]
  This becomes next period’s asset state.
- **Preferences**: CRRA over **end‑of‑period wealth** \(u(w') = \frac{w'^{1-σ}-1}{1-σ}\) (or `log` at σ=1).
- **Financial literacy**: captured by an **effort/cost** \( \text{cost}(x) = κ·x \). (Change this to indicator or hump-shaped cost if preferred.)
- **Shocks**: i.i.d. Extreme Value Type‑I -> **logit CCPs**.
- **NFXP**: outer loop: maximize likelihood of observed discretized `x` via `scipy.optimize.minimize`. Inner loop: value iteration with Gauss‑Hermite quadrature to integrate over risky returns \(R \sim \log\mathcal{N}(μ, σ_R)\).

> The structure mirrors Rust (1987): **outer MLE**, **inner fixed point** for the DP. It is intentionally compact and easy to modify.

## Files

- `nfxp_model.py` — model class, value iteration, CCPs (inner fixed point).
- `estimate.py` — data loader and outer loop (MLE).
- `Demo.ipynb` — small notebook that loads your CSV and runs a quick estimation.

## How to run

### 1) From the command line
```bash
python estimate.py --csv "/mnt/data/toy_panel_with_random_x copy.csv" --outdir ./outputs --na 50 --gh 10 --maxiter 400
```

### 2) In a Python session
```python
from estimate import run_estimation
res, (beta, sigma, kappa) = run_estimation("/mnt/data/toy_panel_with_random_x copy.csv", outdir="./outputs", n_a=40, maxiter=200, gh_order=8)
print(beta, sigma, kappa)
```

### 3) Tips if it’s slow
- Use fewer asset grid points: `--na 20`
- Lower Gauss–Hermite order: `--gh 5`
- Start with smaller `--maxiter 150`
- Later, switch optimizer to `"Powell"` or provide good initial values.

## Interpreting outputs

- `outputs/estimates.csv`: β (discount), σ (CRRA), κ (literacy cost intensity).
- `outputs/V_hat.npy`: value function on the asset grid.
- `outputs/CCP_hat.npy`: conditional choice probabilities on the asset grid.
- `outputs/policy.csv`: greedy policy \(x^*(a)\) on the asset grid.

## Customize the literacy channel

Inside `nfxp_model.py`:
```python
cost = spec.kappa * x                # linear in x  (default)
# cost = spec.kappa * (x > 0)       # entry/participation cost
# cost = spec.kappa * (x*(1-x))     # hump-shaped complexity penalty
```

## Caveats (and easy extensions)

- Income dynamics are summarized by \(ȳ\). If you want AR(1) income, augment the state with a small y‑grid and add a transition matrix.
- We fix `R_f`, `μ`, and `σ_R`. You can estimate them too: include in θ and add simple bounds (be mindful of identification).
- We discretize observed `x` to nearest grid point for the likelihood. You can instead model continuous `x` by adding a **measurement error** or a **kernel** link to the discrete grid.
- If you have direct consumption data, swap the period utility to CRRA over \(c\) and let `w'` feed only the continuation value.

## Reference

- Rust, J. (1987). **“Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher.”** *Econometrica*.

---

**Author**: auto‑generated NFXP scaffold (ChatGPT). Edit freely to match your dissertation spec.


## Update: CRRA over **consumption**
The model now treats **consumption** as the argument of utility:
- resources before consumption: `res = a * r_p(x) + y`
- savings rate `s ∈ {0.6, 0.75, 0.9}`
- consumption `c = (1 - s) * res`
- next assets `a' = s * res`
- period utility `u(c)` is CRRA (log when σ=1)

Because your dataset observes only the portfolio share `x` (not `s`), the likelihood uses the **marginal CCP over x**:
\[
P(x\_t \mid a\_t) \;=\; \sum_{s} P(x\_t, s \mid a\_t).
\]

To run the consumption-based estimator:
```bash
python /mnt/data/nfxp_finlit/estimate.py \
  --csv "/mnt/data/toy_panel_with_random_x copy.csv" \
  --outdir "/mnt/data/nfxp_finlit/outputs_cons" \
  --na 30 --gh 5 --maxiter 150
```
(You can increase `--na`, `--gh`, `--maxiter` after you verify it works on your machine.)



## Budget-embedded sophistication cost (implemented)

We now model sophistication as a **monetary cost** in the budget constraint:
- A **fixed participation cost** \(k_{\text{fix}}\) when \(x>0\) (e.g., course fees, advisor retainers).
- An **ad valorem fee** \(\tau\) (AUM-style) applied to gross resources when \(x>0\).

Budget with risky share \(x\) and savings \(s\):
\[
\text{res}_{\text{gross}} = a \cdot r_p(x) + y,\quad r_p(x)=xR+(1-x)R_f
\]
\[
\text{res}_{\text{net}} =
\begin{cases}
(1-\tau)\,\text{res}_{\text{gross}} - k_{\text{fix}}, & \text{if } x>0,\\\\
\text{res}_{\text{gross}}, & \text{if } x=0.
\end{cases}
\]
\[
c=(1-s)\,\text{res}_{\text{net}},\qquad a' = s\,\text{res}_{\text{net}}.
\]

This replaces the previous utility-penalty. Likelihood still marginalizes over \(s\) because your data observes \(x\) only.

**Run:**
```bash
python /mnt/data/nfxp_finlit/estimate.py \
  --csv "/mnt/data/toy_panel_with_random_x copy.csv" \
  --outdir "/mnt/data/nfxp_finlit/outputs_budget" \
  --na 30 --gh 6 --maxiter 200
```

### Literature anchors (for modeling choices)
- **Participation / fixed costs in portfolio choice**: Haliassos & Bertaut (1995, *JME*), Vissing‑Jørgensen (2002, *Brookings Papers*), Polkovnichenko (2007, *Review of Finance*).
- **AUM / proportional advisory fees** embedded in the budget: Campbell & Viceira (2002, *Strategic Asset Allocation*), French (2008, *JEP*) discusses the magnitude of active management fees, which can be modeled as ad valorem drags.
- **Education / information acquisition as monetary outlay**: Lusardi & Mitchell (2014, *JEL*) survey links between financial knowledge investments and portfolio outcomes.

(Use these as starting points; tailor functional forms to your data, e.g., letting \(k_{\text{fix}}\) and \(\tau\) depend on observables or vary by household.)

