
import numpy as np
from dataclasses import dataclass
from numpy.polynomial.hermite import hermgauss

def gh_nodes_weights(n=8):
    nodes, weights = hermgauss(n)
    weights = weights / np.sqrt(np.pi)
    return nodes, weights

@dataclass
class SpecType:
    # Shared preference parameters across types can be passed externally.
    beta: float
    sigma: float
    R_f: float
    mu_lnR: float
    sigma_lnR: float
    # Type-specific budget-embedded cost parameters (functions of ln a)
    gamma0: float   # for k_fix(a) = exp(gamma0 + gamma1 ln a)
    gamma1: float
    delta0: float   # for tau(a)   = tau_min + tau_range * sigmoid(delta0 + delta1 ln a)
    delta1: float
    tau_min: float = 1e-4
    tau_range: float = 0.10

class NFXPFinLitConsBudgetAgeType:
    """
    DP with CRRA over consumption, budget-embedded costs, age as a state,
    and support for computing CCPs per type. Mixture weights handled outside.

    States: (a, age_idx) with deterministic aging age_next = min(age_idx+1, n_age-1)
    Actions: x in x_grid, s in s_grid
    Budget for a given type:
        res_gross = a * r_p(x) + y_bar, with r_p(x) = x*R + (1-x)*R_f
        res_net   = (1 - tau(a)) * res_gross - k_fix(a)   if x>0
                    res_gross                              if x=0
        c = (1-s) * res_net,     a' = s * res_net
    u(c) CRRA, EV1 shocks -> logit CCPs over (x,s).
    """
    def __init__(self, a_grid, age_grid, y_bar, x_grid, s_grid, gh_order=6):
        self.a = a_grid
        self.na = len(a_grid)
        self.age = age_grid.astype(int)
        self.nage = len(age_grid)
        self.y = float(y_bar)
        self.xgrid = x_grid
        self.ndx = len(x_grid)
        self.sgrid = s_grid
        self.nds = len(s_grid)
        self.nodes, self.weights = gh_nodes_weights(gh_order)

    @staticmethod
    def crra(c, sigma):
        eps = 1e-10
        c = np.maximum(c, eps)
        if abs(sigma - 1.0) < 1e-8:
            return np.log(c)
        return (np.power(c, 1.0 - sigma) - 1.0) / (1.0 - sigma)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fees(self, a_vec, spec: SpecType):
        ln_a = np.log(np.maximum(a_vec, 1e-8))
        k_fix = np.exp(spec.gamma0 + spec.gamma1 * ln_a)  # (na,)
        tau   = spec.tau_min + spec.tau_range * self.sigmoid(spec.delta0 + spec.delta1 * ln_a)
        return k_fix, tau

    def value_iteration_one_type(self, spec: SpecType, tol=1e-6, maxit=500):
        # Precompute by age-independent terms
        R_vals = np.exp(spec.mu_lnR + spec.sigma_lnR * self.nodes)  # (nR,)
        # Fees depend on a but not age
        k_fix_a, tau_a = self.fees(self.a, spec)  # (na,), (na,)

        # V over (a_idx, age_idx)
        V = np.zeros((self.na, self.nage))

        for it in range(maxit):
            V_old = V.copy()

            def V_interp(a_prime, age_next_idx):
                # linear interp in a, fixed next-age slice
                a_min, a_max = self.a[0], self.a[-1]
                ap = np.clip(a_prime, a_min, a_max)
                return np.interp(ap, self.a, V[:, age_next_idx])

            # For each age slice compute Bellman update using continuation in next age
            for g in range(self.nage):
                age_next = min(g + 1, self.nage - 1)
                Q = np.zeros((self.na, self.ndx, self.nds))

                for j, x in enumerate(self.xgrid):
                    rp = x * R_vals + (1 - x) * spec.R_f       # (nR,)
                    res_gross = np.outer(self.a, rp) + self.y  # (na, nR)
                    part = (x > 0).astype(float)

                    k_mat = k_fix_a[:, None] * part
                    tau_mat = tau_a[:, None] * part

                    res_net = res_gross * (1.0 - tau_mat) - k_mat
                    res_net = np.maximum(res_net, 1e-10)

                    for k, s in enumerate(self.sgrid):
                        c_next = (1.0 - s) * res_net
                        a_next = s * res_net

                        U = self.crra(c_next, spec.sigma)
                        # interpolate continuation on next age slice
                        Vcont = np.array([V_interp(w, age_next) for w in a_next.ravel()]).reshape(a_next.shape)
                        EU = (U + spec.beta * Vcont) @ self.weights
                        Q[:, j, k] = EU

                # log-sum-exp over joint actions
                Q2 = Q.reshape(self.na, -1)
                row_max = Q2.max(axis=1, keepdims=True)
                V[:, g] = (row_max + np.log(np.exp(Q2 - row_max).sum(axis=1, keepdims=True))).ravel()

            if np.max(np.abs(V - V_old)) < tol:
                break

        # Produce CCPs and policy for each age slice
        CCP = np.zeros((self.na, self.nage, self.ndx, self.nds))
        pol_x = np.zeros((self.na, self.nage))
        pol_s = np.zeros((self.na, self.nage))

        for g in range(self.nage):
            Q = np.zeros((self.na, self.ndx, self.nds))
            age_next = min(g + 1, self.nage - 1)

            def V_interp(a_prime):
                a_min, a_max = self.a[0], self.a[-1]
                ap = np.clip(a_prime, a_min, a_max)
                return np.interp(ap, self.a, V[:, age_next])

            for j, x in enumerate(self.xgrid):
                rp = x * R_vals + (1 - x) * spec.R_f
                res_gross = np.outer(self.a, rp) + self.y
                part = (x > 0).astype(float)

                k_mat = k_fix_a[:, None] * part
                tau_mat = tau_a[:, None] * part

                res_net = res_gross * (1.0 - tau_mat) - k_mat
                res_net = np.maximum(res_net, 1e-10)

                for k, s in enumerate(self.sgrid):
                    c_next = (1.0 - s) * res_net
                    a_next = s * res_net
                    U = self.crra(c_next, spec.sigma)
                    Vcont = np.array([V_interp(w) for w in a_next.ravel()]).reshape(a_next.shape)
                    EU = (U + spec.beta * Vcont) @ self.weights
                    Q[:, j, k] = EU

            Q2 = Q.reshape(self.na, -1)
            row_max = Q2.max(axis=1, keepdims=True)
            denom = np.exp(Q2 - row_max).sum(axis=1, keepdims=True)
            CCP_flat = np.exp(Q2 - row_max) / denom
            CCP[:, g, :, :] = CCP_flat.reshape(self.na, self.ndx, self.nds)

            pol_flat = CCP_flat.argmax(axis=1)
            pol_x[:, g] = self.xgrid[(pol_flat // self.nds)]
            pol_s[:, g] = self.sgrid[(pol_flat % self.nds)]

        return V, CCP, pol_x, pol_s

    def ccp_x_marginal(self, a_obs, age_obs, CCP):
        # CCP over x only = sum_s CCP(a_idx, age_idx, x, s)
        # Map a_obs to nearest index, age_obs to exact index
        a_idx = np.searchsorted(self.a, a_obs, side='left')
        a_idx = np.clip(a_idx, 0, self.na - 1)
        # age grid expected to include observed ages; fallback to nearest
        age_idx = np.searchsorted(self.age, age_obs, side='left')
        age_idx = np.clip(age_idx, 0, self.nage - 1)
        return CCP[a_idx, age_idx, :, :].sum(axis=2)
