
# FinLit CCP (Ψ–QML) — Multi-Action with Risky-Share Bins

This package implements a *structural* lecture-style CCP estimator with multi-action risky-share bins,
using your dataset `/mnt/data/synthetic_finlit_portfolio.csv`.

**Action set:** D(x) = { d=0 (no participation) } ∪ { d=1..Kx (participation with risky-share bin k) }.

**Utility:** From the budget with CRRA and a literacy-dependent fixed cost:
- u(c) = c^(1-σ)/(1-σ)  (or log c if σ=1)
- F(L) = F0 - φ * L

**Expected flow utility:** integrates over risky returns with Gauss–Hermite quadrature.

**Lecture operators (Arcidiacono–Miller):**
- φ(P;θ): V_σ = [I - β Π_mix(P)]^{-1} * Σ_d P_d ⊙ [ U_d(θ) + e_d(P) ]
- λ(V;θ): softmax of v(x,d) = U_d(θ) + β Π(d) V
- Ψ(P;θ) = λ( φ(P;θ), θ )

**Estimation (QML):** maximize Σ log Ψ( P̂, θ )(d|x).

Run:
    python run_psiqml.py
Edit `run_psiqml.py` to change Kx, bins, or starting values.
