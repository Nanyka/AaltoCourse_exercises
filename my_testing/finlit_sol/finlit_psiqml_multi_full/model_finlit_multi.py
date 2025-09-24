# model_finlit_multi.py  (stable medians-in-levels, capped; robust φ)
import numpy as np
import pandas as pd

EULER_G = 0.5772156649015329

def gh_nodes(n):
    from numpy.polynomial.hermite_e import hermegauss
    z, w = hermegauss(n)
    # For N(0,1): E[f(Z)] = sum w_i / sqrt(pi) * f( sqrt(2)*z_i )
    return np.sqrt(2.0)*z, w/np.sqrt(np.pi)

class FinLitMulti:
    def __init__(self, **kwargs):
        self.setup(**kwargs)

    def setup(self,
              data_path,
              Kx=3,
              share_grid=None,
              bins_assets=4, bins_income=3, bins_lit=3, bins_age=3,
              smooth=0.5,
              beta=0.95,
              rf=0.01,
              quad_n=5,
              cap_quantile=0.995,      # NEW: cap reps at 99.5% level
              ridge_phi=1e-10):        # NEW: tiny ridge in φ linear system
        self.data_path = data_path
        self.beta = float(beta)
        self.rf = float(rf)
        self.smooth = float(smooth)
        self.quad_n = int(quad_n)
        self.cap_q = float(cap_quantile)
        self.ridge_phi = float(ridge_phi)

        self.n_a = int(bins_assets)
        self.n_y = int(bins_income)
        self.n_l = int(bins_lit)
        self.n_g = int(bins_age)

        self.Kx = int(Kx)
        if share_grid is None:
            self.share_grid = (np.arange(1, self.Kx+1) - 0.5)/self.Kx
        else:
            self.share_grid = np.asarray(share_grid, float)
            self.Kx = len(self.share_grid)

        self._loaded = False

    # ---------- helpers ----------
    @staticmethod
    def _safe_cuts(series, nbins, pad_high=0.0):
        """Quantile cuts with *no* aggressive padding (we’ll cap reps separately)."""
        q = np.linspace(0, 1, nbins+1)
        cuts = np.quantile(series, q).astype(float)
        for i in range(1, len(cuts)):
            if cuts[i] <= cuts[i-1]:
                cuts[i] = cuts[i-1] + 1e-8
        # No big extension; just tiny epsilon at ends
        cuts[0] = float(np.min(series)) - 1e-8
        cuts[-1] = float(np.max(series)) + pad_high
        return cuts

    @staticmethod
    def _cut_to_codes(series, cuts):
        cat = pd.Categorical(pd.cut(series, bins=cuts, include_lowest=True, ordered=True))
        codes = cat.codes.astype(int)
        nb = len(cuts)-1
        return np.where(codes < 0, nb-1, codes)

    @staticmethod
    def _row_normalize(M, eps=1e-12):
        M = np.asarray(M, float)
        rs = M.sum(axis=1, keepdims=True)
        bad = (rs <= eps).ravel()
        if np.any(bad):
            M[bad, :] = 1.0
            rs = M.sum(axis=1, keepdims=True)
        return M / rs

    @staticmethod
    def _bin_level_medians(levels, bin_codes, nbins, cap_high=None):
        """Representative *level* per bin: within-bin median, capped."""
        rep = np.zeros(nbins, dtype=float)
        if cap_high is None:
            cap_high = np.quantile(levels, 0.995)
        for b in range(nbins):
            mask = (bin_codes == b)
            if np.any(mask):
                val = float(np.median(levels[mask]))
            else:
                # Fallback: use midpoint of unconditional quantile slice
                lo = np.quantile(levels, b/nbins)
                hi = np.quantile(levels, (b+1)/nbins)
                val = float(0.5*(lo+hi))
            rep[b] = np.clip(val, 0.0, cap_high)
        rep[~np.isfinite(rep)] = 0.0
        return rep

    # ---------- data → first-stage objects ----------
    def load_and_bin(self):
        df = pd.read_csv(self.data_path)
        need = ["id","wave","participate_risky","risky_share",
                "liquid_assets_eur","income_eur","literacy_index_z","age","country"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"Dataset missing columns: {miss}")

        d1 = df[df.wave==1].copy()
        d2 = df[df.wave==2].copy()
        m = pd.merge(d1, d2, on="id", suffixes=("_1","_2"))
        if m.empty:
            raise ValueError("No overlapping ids across waves 1 and 2.")

        # build *log* bins on wave 1 (stable)
        m["logA_1"] = np.log(m["liquid_assets_eur_1"] + 1.0)
        m["logY_1"] = np.log(m["income_eur_1"] + 1.0)

        a_cuts = self._safe_cuts(m["logA_1"].values, self.n_a, pad_high=0.0)
        y_cuts = self._safe_cuts(m["logY_1"].values, self.n_y, pad_high=0.0)
        l_cuts = self._safe_cuts(m["literacy_index_z_1"].values, self.n_l, pad_high=0.0)
        g_cuts = self._safe_cuts(m["age_1"].values, self.n_g, pad_high=0.0)

        a1 = self._cut_to_codes(m["logA_1"].values, a_cuts)
        y1 = self._cut_to_codes(m["logY_1"].values, y_cuts)
        l1 = self._cut_to_codes(m["literacy_index_z_1"].values, l_cuts)
        g1 = self._cut_to_codes(m["age_1"].values, g_cuts)

        # same binning for wave 2 (do NOT re-estimate cuts)
        m["logA_2"] = np.log(m["liquid_assets_eur_2"] + 1.0)
        m["logY_2"] = np.log(m["income_eur_2"] + 1.0)
        a2 = self._cut_to_codes(m["logA_2"].values, a_cuts)
        y2 = self._cut_to_codes(m["logY_2"].values, y_cuts)
        l2 = self._cut_to_codes(m["literacy_index_z_2"].values, l_cuts)
        g2 = self._cut_to_codes(m["age_2"].values, g_cuts)

        # State index
        def idx(a, y, l, g, n_y, n_l, n_g):
            return (((a*n_y + y)*n_l + l)*n_g + g).astype(int)
        x1 = idx(a1,y1,l1,g1,self.n_y,self.n_l,self.n_g)
        x2 = idx(a2,y2,l2,g2,self.n_y,self.n_l,self.n_g)
        self.n = int(self.n_a*self.n_y*self.n_l*self.n_g)

        # Observed multi-action: d=0 no-participation, d=1..Kx bins for risky_share if s=1
        s1 = m["participate_risky_1"].astype(int).values
        rs = m["risky_share_1"].fillna(0.0).clip(0,1).values
        rs[s1==0] = 0.0
        kx = np.digitize(rs, bins=np.linspace(0,1,self.Kx+1), right=True)
        kx = np.clip(kx, 0, self.Kx)
        d_obs = np.where(s1==0, 0, kx)

        # Panel for likelihood mapping
        self.panel = pd.DataFrame({"id": m["id"].values, "x": x1, "d": d_obs, "xprime": x2})
        self.x_obs = self.panel["x"].values
        self.d_obs = self.panel["d"].values

        # ---------- Representative *levels* by bin: medians, capped ----------
        A1_levels = m["liquid_assets_eur_1"].values.astype(float)
        Y1_levels = m["income_eur_1"].values.astype(float)
        A_cap = float(np.quantile(A1_levels, self.cap_q))
        Y_cap = float(np.quantile(Y1_levels, self.cap_q))

        a_rep = self._bin_level_medians(A1_levels, a1, self.n_a, cap_high=A_cap)
        y_rep = self._bin_level_medians(Y1_levels, y1, self.n_y, cap_high=Y_cap)

        # Midpoints for literacy and age can remain bin midpoints (already in levels)
        l_mid = 0.5*(l_cuts[:-1]+l_cuts[1:])
        g_mid = 0.5*(g_cuts[:-1]+g_cuts[1:])

        # Expand to full state grid in the correct index order
        A = np.repeat(a_rep, self.n_y*self.n_l*self.n_g)
        Y = np.tile(np.repeat(y_rep, self.n_l*self.n_g), self.n_a)
        L = np.tile(np.repeat(l_mid, self.n_g), self.n_a*self.n_y)
        G = np.tile(g_mid, self.n_a*self.n_y*self.n_l)

        # Clip to be safe
        A = np.clip(A, 0.0, A_cap)
        Y = np.clip(Y, 0.0, Y_cap)

        self.state_mids = np.vstack([A,Y,L,G]).T  # [n x 4] finite and capped

        # First-stage CCPs & transitions
        self._build_first_stage(x1, d_obs, x2)
        self._loaded = True

    def _build_first_stage(self, x, d, xp):
        n, K = self.n, self.Kx + 1

        # CCPs (Laplace-smoothed)
        counts = np.zeros((n, K), dtype=float)
        for xi, di in zip(x, d):
            counts[xi, di] += 1.0
        counts += self.smooth
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        self.P_hat = counts / row_sums

        # Action-specific transitions (smoothed, row-normalized, sanitized)
        Pis = [np.zeros((n, n), dtype=float) for _ in range(K)]
        for xi, di, xpi in zip(x, d, xp):
            Pis[di][xi, xpi] += 1.0
        self.Pi = []
        for M in Pis:
            M = M + self.smooth
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
            rs = M.sum(axis=1, keepdims=True)
            rs[~np.isfinite(rs) | (rs <= 0.0)] = 1.0
            M = M / rs
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
            self.Pi.append(M)

        # Expected next asset level a'bar(x,d) = Σ_{x'} Π_d(x'|x) * a_mid(x')
        a_mid_by_state = np.ascontiguousarray(self.state_mids[:, 0], dtype=float)
        self.a_next_bar = np.zeros((n, K), dtype=float)
        for j in range(K):
            M = self.Pi[j]
            # safe mat-vec: einsum avoids BLAS warnings & is numerically stable
            a_bar = np.einsum('ij,j->i', M, a_mid_by_state)
            a_bar = np.nan_to_num(a_bar, nan=0.0, posinf=np.max(a_mid_by_state), neginf=0.0)
            self.a_next_bar[:, j] = np.clip(a_bar, 0.0, np.max(a_mid_by_state))

    # ---------- structural flow utility ----------
    def expected_flow_utility(self, theta):
        """θ = (sigma, F0, phi, mu_R, sigma_R)"""
        sigma, F0, phi, muR, sigR = [float(t) for t in theta]
        n, K = self.n, self.Kx+1
        U = np.zeros((n, K), dtype=float)
        A = self.state_mids[:,0]  # levels
        Y = self.state_mids[:,1]
        L = self.state_mids[:,2]

        z, w = gh_nodes(self.quad_n)

        # d=0
        c0 = Y + A*(1.0 + self.rf) - self.a_next_bar[:,0]
        c0 = np.maximum(c0, 1e-8)
        U[:,0] = np.log(c0) if sigma==1.0 else c0**(1.0 - sigma) / (1.0 - sigma)

        # d=1..Kx
        for j in range(1, K):
            xshare = self.share_grid[j-1]
            a_next = self.a_next_bar[:, j]
            EU = np.zeros(n, dtype=float)
            for zi, wi in zip(z, w):
                R = muR + sigR*zi
                gross = (1.0 - xshare)*(1.0 + self.rf) + xshare*(1.0 + R)
                c = Y + A*gross - a_next - (F0 - phi*L)
                c = np.maximum(c, 1e-8)
                u = (np.log(c) if sigma==1.0 else c**(1.0 - sigma) / (1.0 - sigma))
                EU += wi * u
            # sanitize just in case
            EU[~np.isfinite(EU)] = np.nanmedian(EU[np.isfinite(EU)]) if np.any(np.isfinite(EU)) else 0.0
            U[:, j] = EU

        U[~np.isfinite(U)] = 0.0
        return U

    # ---------- lecture operators ----------
    def phi(self, P, theta):
        beta = self.beta
        n, K = self.n, self.Kx + 1
        U = self.expected_flow_utility(theta)

        # EV1 correction
        P = np.clip(P, 1e-12, 1.0)
        e = EULER_G - np.log(P)

        # Π_mix = Σ_d [ Π(d) * P_d row-wise ]  (broadcast instead of diag(P) @ Π)
        Pi_mix = np.zeros((n, n), dtype=float)
        rhs = np.zeros(n, dtype=float)
        for j in range(K):
            # multiply each row of Π_j by P[:, j]
            Pi_mix += self.Pi[j] * P[:, j][:, None]
            rhs += P[:, j] * (U[:, j] + e[:, j])

        # Solve (I - β Π_mix + ridge I) V = rhs
        A = np.eye(n, dtype=float) - beta * Pi_mix
        if getattr(self, "ridge_phi", 1e-10) > 0:
            A[np.diag_indices(n)] += self.ridge_phi

        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            V = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            V, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        return V

    def lambdaa(self, V, theta):
        beta = self.beta
        n, K = self.n, self.Kx + 1
        U = self.expected_flow_utility(theta)
        V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

        Vd = np.empty((n, K), dtype=float)
        for j in range(K):
            # safe mat-vec
            cont = np.einsum('ij,j->i', self.Pi[j], V)
            cont = np.nan_to_num(cont, nan=0.0, posinf=0.0, neginf=0.0)
            Vd[:, j] = U[:, j] + beta * cont

        vmax = np.max(Vd, axis=1, keepdims=True)
        expv = np.exp(np.clip(Vd - vmax, -50.0, 50.0))
        Pm = expv / expv.sum(axis=1, keepdims=True)
        Pm = np.nan_to_num(Pm, nan=1.0 / (K), posinf=1.0 / (K), neginf=1.0 / (K))
        return Pm

    def psi(self, P, theta):
        return self.lambdaa(self.phi(P, theta), theta)

    # ---------- QML ----------
    def loglik_qml(self, theta, P_hat=None):
        if P_hat is None:
            P_hat = self.P_hat
        Pm = self.psi(P_hat, theta)
        p = Pm[self.x_obs, self.d_obs]
        return float(np.sum(np.log(np.clip(p, 1e-12, 1.0))))
