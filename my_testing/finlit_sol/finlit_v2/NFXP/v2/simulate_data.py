
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from nfxp_model_v2 import NFXPFinLitConsBudgetAgeType, SpecType

def _sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

@dataclass
class SimConfig:
    N: int = 500
    T: int = 8
    NA_grid: int = 24
    gh_order: int = 5
    a_min: float = 5_000.0
    a_max: float = 80_000.0
    age_start: int = 30
    y_bar: float = 25_000.0

def solve_type_grids(cfg: SimConfig, base_const: Dict, theta: Tuple[float,float,float,float]):
    a_grid = np.linspace(cfg.a_min, cfg.a_max, cfg.NA_grid)
    age_grid = np.arange(cfg.age_start, cfg.age_start + cfg.T + 5)
    x_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    s_grid = np.array([0.6, 0.8, 0.95])
    model = NFXPFinLitConsBudgetAgeType(a_grid, age_grid, cfg.y_bar, x_grid, s_grid, gh_order=cfg.gh_order)
    g0, g1, d0, d1 = theta
    spec = SpecType(**base_const, gamma0=g0, gamma1=g1, delta0=d0, delta1=d1)
    V, CCP, polx, pols = model.value_iteration_one_type(spec)
    return model, spec, V, CCP, polx, pols

def simulate_panel(cfg: SimConfig,
                   base_const: Dict,
                   theta1: Tuple[float,float,float,float],
                   theta2: Optional[Tuple[float,float,float,float]] = None,
                   eta: Optional[np.ndarray] = None,
                   rng_seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    # beta = base_const.get("beta", 0.95)
    # sigma = base_const.get("sigma", 2.0)
    model1, spec1, V1, CCP1, polx1, pols1 = solve_type_grids(cfg, base_const, theta1)
    if theta2 is not None:
        model2, spec2, V2, CCP2, polx2, pols2 = solve_type_grids(cfg, base_const, theta2)
    else:
        model2 = spec2 = V2 = CCP2 = polx2 = pols2 = None

    ids = np.arange(1, cfg.N+1)
    age0 = np.full(cfg.N, cfg.age_start, dtype=float)
    a0 = rng.uniform(cfg.a_min, cfg.a_max, size=cfg.N)
    ln_a0 = np.log(np.maximum(a0, 1e-8))

    if theta2 is not None and eta is not None:
        Z = np.column_stack([np.ones(cfg.N), age0, ln_a0])
        p1 = _sigmoid(Z @ eta)
        types = rng.binomial(1, p1) + 1
    else:
        p1 = np.ones(cfg.N)
        types = np.ones(cfg.N, dtype=int)

    rows = []
    for i in range(cfg.N):
        a = a0[i]; g = age0[i]; t = 0
        for tt in range(cfg.T):
            if types[i] == 1:
                model, CCP, spec = model1, CCP1, spec1
            else:
                model, CCP, spec = model2, CCP2, spec2

            ai = int(np.argmin(np.abs(model.agrid - a)))
            gi = int(np.argmin(np.abs(model.agegrid - g)))
            probs = CCP[ai, gi, :, :].reshape(-1)
            probs = probs / np.maximum(probs.sum(), 1e-300)
            idx = rng.choice(len(probs), p=probs)
            xi, si = np.unravel_index(idx, (len(model.xgrid), len(model.sgrid)))
            x = float(model.xgrid[xi]); s = float(model.sgrid[si])

            Zret = rng.normal(0.0, 1.0)
            R = np.exp(spec.mu_lnR + spec.sigma_lnR * Zret)
            gross = a * (x * R + (1.0 - x) * spec.R_f) + cfg.y_bar
            if x > 1e-12:
                kfix = np.exp(spec.gamma0 + spec.gamma1 * g)
                tau = 0.9 / (1.0 + np.exp(-(spec.delta0 + spec.delta1 * g)))
                res = (1.0 - tau) * gross - kfix
            else:
                res = gross
            res = max(res, 1e-12)
            c = (1.0 - s) * res
            a_next = s * res

            rows.append({"id": int(ids[i]), "t": int(t), "a": float(a), "y": float(cfg.y_bar),
                         "x": float(x), "age": float(g), "s": float(s), "c": float(c), "type": int(types[i])})
            a = a_next; g = g + 1.0; t += 1
    df = pd.DataFrame(rows).sort_values(["id","t"]).reset_index(drop=True)
    return df
