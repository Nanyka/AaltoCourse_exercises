
#!/usr/bin/env python3
import numpy as np
from run_estimation import generate_dataset, newton_maximize
from nrls_optimizer import nrls_maximize, loglik
from tictactoe_play_utils import FEATURE_NAMES

def main():
    theta_star = np.array([2.0,3.0,0.5,0.3,0.0,0.6])
    data, theta_star = generate_dataset(n_games=1500, theta_star=theta_star, seed=42)
    print(f"Data points: {len(data)} | theta*={theta_star}")

    theta0 = np.zeros(len(FEATURE_NAMES))
    th_newton, ll_newton = newton_maximize(data, theta0, lam=1e-6, verbose=False)
    print("\n[Newton–Armijo]")
    print("theta_hat:", th_newton)
    print("ll      :", ll_newton)

    bounds = [(-4,6),(-4,6),(-2,2),(-2,2),(-2,2),(-2,2)]
    f = lambda th: loglik(data, np.asarray(th), lam=1e-6)
    res = nrls_maximize(f, bounds, levels=(5,7,9), topk=6, shrink=0.4, verbose=False)
    print("\n[NRLS]")
    print("theta_hat:", res.theta)
    print("ll      :", res.value)
    print("evals   :", res.evaluations)

if __name__ == "__main__":
    main()
