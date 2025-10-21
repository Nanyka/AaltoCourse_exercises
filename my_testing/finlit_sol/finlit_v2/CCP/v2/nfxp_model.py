"""
nfxp_model.py
-------------
Dynamic portfolio-choice model with CRRA utility and **budget-embedded financial
literacy costs** (fixed participation + ad-valorem fee). This is the inner
problem used by the EM estimator in `estimate_em_obs.py`.

States and actions
==================
- State s = (a, g): assets `a` on a grid, age `g` on a grid (deterministic aging).
- Controls d = (x, s): risky portfolio share `x` in [0,1] on a grid, savings
  rate `s` in (0,1) on a grid.

Budget with literacy costs
==========================
Let R_f be the safe gross return and R the risky gross return (lognormal).
Gross resources before literacy costs:
    res_gross = a * [ x*R + (1-x)*R_f ] + y_bar
If x > 0, agents pay:
    - fixed cost k_fix(a,g)  (interpreted as a toll to participate)
    - ad-valorem fee tau(a,g) (interpreted as a percent wedge)
Net resources after costs:
    res_net = (1 - tau) * res_gross - k_fix   if x > 0
            = res_gross                       if x = 0
Consumption and next-period assets given savings rate s:
    c = (1-s) * res_net
    a' = s * res_net

Preferences
===========
CRRA utility over consumption c with curvature sigma and discount beta.
Taste shocks are EV1 at the action level, so we aggregate Q-values with
log-sum-exp to obtain V and derive logit CCPs.

Numerical notes
===============
- Risky return integration: Gauss–Hermite quadrature over Z ~ N(0,1) with
  R = exp(mu_lnR + sigma_lnR * Z).
- Asset projection: nearest neighbor (fast and robust for coarse grids).
- EV1 aggregation: V(s) = log sum_{x,s} exp(Q(s,x,s)).
- CCPs: softmax over (x,s) using the same Q.

This class is intentionally minimal and documented for clarity. You can
extend k_fix and tau to depend on ln(a), education, etc.
"""

from dataclasses import dataclass
import numpy as np
from numpy.polynomial.hermite import hermgauss


@dataclass
class SpecType:
    """
    Container for all parameters *for a given type*.

    Attributes
    ----------
    beta, sigma : float
        Discount factor and CRRA curvature.
    R_f : float
        Safe gross return.
    mu_lnR, sigma_lnR : float
        Parameters of the lognormal risky return in logs.
    gamma0, gamma1 : float
        Parameters for fixed cost k_fix(a,g) = exp(gamma0 + gamma1 * g).
        (You can generalize to exp(gamma0 + gamma1*ln a + gamma2*g) as needed.)
    delta0, delta1 : float
        Parameters for ad-valorem fee tau(a,g) = 0.9 * logistic(delta0 + delta1 * g).
    """
    beta: float
    sigma: float
    R_f: float
    mu_lnR: float
    sigma_lnR: float
    gamma0: float
    gamma1: float
    delta0: float
    delta1: float


class NFXPFinLitConsBudgetAgeType:
    """
    NFXP solver for the dynamic portfolio-choice problem with literacy costs
    embedded in the *budget constraint*.

    Parameters
    ----------
    agrid : array-like
        Asset grid (in levels). Must be non-negative and increasing.
    agegrid : array-like
        Age grid (deterministic aging: g_{t+1} = next point, capped at last).
    y_bar : float
        Deterministic income level used in the first-pass implementation.
    xgrid : array-like
        Discrete grid for risky share x in [0,1].
    sgrid : array-like
        Discrete grid for savings rate s in (0,1).
    gh_order : int, default 5
        Gauss–Hermite quadrature order for risky return integration.
    ev1_scale : float, default 1.0
        EV1 scale parameter (kept at 1 for standard logit).

    Methods
    -------
    value_iteration_one_type(spec, max_iter=300, tol=1e-6)
        Solve the DP for a single type; return V, CCP, and policies.
    ccp_x_marginal(a_series, age_series, CCP)
        Compute P(x | state) by summing CCP(x,s|state) over s along a path.
    """

    def __init__(self, agrid, agegrid, y_bar, xgrid, sgrid, gh_order=5, ev1_scale=1.0):
        self.agrid = np.asarray(agrid, dtype=float)
        self.agegrid = np.asarray(agegrid, dtype=float)
        self.y_bar = float(y_bar)
        self.xgrid = np.asarray(xgrid, dtype=float)
        self.sgrid = np.asarray(sgrid, dtype=float)
        self.gh_order = int(gh_order)
        self.ev1_scale = float(ev1_scale)

        # Gauss–Hermite nodes and weights for Z ~ N(0,1)
        nodes, weights = hermgauss(self.gh_order)
        self.z_nodes = np.sqrt(2.0) * nodes
        self.z_w = weights / np.sqrt(np.pi)
        self.z_w = self.z_w / np.sum(self.z_w)  # normalize to sum to 1

        self.NA = len(self.agrid)
        self.NG = len(self.agegrid)
        self.NX = len(self.xgrid)
        self.NS = len(self.sgrid)

    # ---------- Preferences ----------
    def _u(self, c, sigma):
        """
        CRRA utility with safe guard for c <= 0.

        If sigma == 1, return log utility.
        """
        c = np.maximum(c, 1e-12)
        if abs(sigma - 1.0) < 1e-12:
            return np.log(c)
        return (np.power(c, 1.0 - sigma) - 1.0) / (1.0 - sigma)

    # ---------- Literacy costs (customize here) ----------
    def _kfix(self, a, g, spec: SpecType):
        """Fixed cost function k_fix(a,g). Currently: exp(gamma0 + gamma1 * g)."""
        return np.exp(spec.gamma0 + spec.gamma1 * g)

    def _tau(self, a, g, spec: SpecType):
        """Ad-valorem fee tau(a,g). Currently: 0.9 * logistic(delta0 + delta1 * g)."""
        z = spec.delta0 + spec.delta1 * g
        return 0.9 / (1.0 + np.exp(-z))  # cap below 1 for feasibility

    # ---------- Core solver ----------
    def value_iteration_one_type(self, spec: SpecType, max_iter=300, tol=1e-6):
        """
        Value iteration for a single type.

        Returns
        -------
        V   : array, shape (NA, NG)
            Value function on the (a,g) grid.
        CCP : array, shape (NA, NG, NX, NS)
            Logit choice probabilities for each (x,s) at each state.
        polx: array, shape (NA, NG)
            Argmax risky share at each state.
        pols: array, shape (NA, NG)
            Argmax savings rate at each state.
        """
        V = np.zeros((self.NA, self.NG))
        Q = np.zeros((self.NA, self.NG, self.NX, self.NS))

        for it in range(max_iter):
            V_new = np.empty_like(V)
            for gi, g in enumerate(self.agegrid):
                # next age (absorbing at the last index)
                next_gi = min(gi + 1, self.NG - 1)
                for ai, a in enumerate(self.agrid):
                    # Loop over risky share grid and integrate over returns
                    for xi, x in enumerate(self.xgrid):
                        exp_vals_s = np.zeros(self.NS)
                        for k in range(len(self.z_w)):
                            Rk = np.exp(spec.mu_lnR + spec.sigma_lnR * self.z_nodes[k])
                            w = self.z_w[k]
                            # Loop over savings grid
                            for si, s in enumerate(self.sgrid):
                                # Gross resources before costs
                                gross = a * (x * Rk + (1.0 - x) * spec.R_f) + self.y_bar
                                # Apply literacy costs only if participating (x>0)
                                if x > 1e-12:
                                    res = (1.0 - self._tau(a, g, spec)) * gross - self._kfix(a, g, spec)
                                else:
                                    res = gross
                                res = max(res, 1e-12)
                                c = (1.0 - s) * res
                                a_next = s * res
                                # Nearest-neighbor projection for next asset
                                aj = int(np.argmin(np.abs(self.agrid - a_next)))
                                cont = V[aj, next_gi]
                                exp_vals_s[si] += w * (self._u(c, spec.sigma) + spec.beta * cont)
                        Q[ai, gi, xi, :] = exp_vals_s

                    # EV1 aggregation over (x,s): V = log-sum-exp(Q)
                    m = np.max(Q[ai, gi, :, :])
                    V_new[ai, gi] = m + np.log(np.sum(np.exp(Q[ai, gi, :, :] - m)))

            # Convergence check
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new

        # Derive CCPs (softmax over (x,s))
        CCP = np.zeros_like(Q)
        for gi in range(self.NG):
            for ai in range(self.NA):
                m = np.max(Q[ai, gi, :, :])
                den = np.exp(Q[ai, gi, :, :] - m).sum()
                CCP[ai, gi, :, :] = np.exp(Q[ai, gi, :, :] - m) / max(den, 1e-300)

        # Greedy policies (argmax of Q)
        polx = np.zeros((self.NA, self.NG))
        pols = np.zeros((self.NA, self.NG))
        for gi in range(self.NG):
            for ai in range(self.NA):
                idx = np.unravel_index(np.argmax(Q[ai, gi, :, :]), (self.NX, self.NS))
                polx[ai, gi] = self.xgrid[idx[0]]
                pols[ai, gi] = self.sgrid[idx[1]]

        return V, CCP, polx, pols

    # ---------- Likelihood helper ----------
    def ccp_x_marginal(self, a_series, age_series, CCP):
        """
        Compute P(x | state) along a realized path by summing CCP(x,s|state) over s.

        Parameters
        ----------
        a_series, age_series : array-like
            Sequences of assets and ages for each observation.
        CCP : np.ndarray of shape (NA, NG, NX, NS)
            Choice probabilities on the grid.

        Returns
        -------
        Px : array, shape (T, NX)
            For each row t, the vector of probabilities over x-grid.
        """
        a_series = np.asarray(a_series)
        age_series = np.asarray(age_series)
        out = np.zeros((len(a_series), self.NX))
        for t in range(len(a_series)):
            a = a_series[t]
            g = age_series[t]
            ai = np.argmin(np.abs(self.agrid - a))
            gi = np.argmin(np.abs(self.agegrid - g))
            Px = CCP[ai, gi, :, :].sum(axis=1)  # sum over s
            Px = Px / max(Px.sum(), 1e-300)
            out[t, :] = Px
        return out
