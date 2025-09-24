
from Estimate_QML import estimate_qml

if __name__ == "__main__":
    config = {
        "data_path": "./synthetic_finlit_portfolio.csv",
        "Kx": 3,
        "bins_assets": 4, "bins_income": 3, "bins_lit": 3, "bins_age": 3,
        "smooth": 0.5,
        "beta": 0.95,
        "rf": 0.01,
        "quad_n": 5,
        "maxiter": 60,
        # starting values (tune if needed)
        "sigma0": 3.0, "F0_0": 500.0, "phi0": 250.0, "muR0": 0.06, "sigR0": 0.18,
    }
    res, model = estimate_qml(config)
    print("Converged:", res.success, "| Message:", res.message)
    print("theta_hat = (sigma, F0, phi, mu_R, sigma_R):", res.x)
    print("QML log-likelihood:", -res.fun)
